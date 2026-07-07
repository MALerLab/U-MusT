import os
import sys
import shutil
import subprocess
import warnings
from typing import Union, Optional
from pathlib import Path
from tempfile import TemporaryFile, NamedTemporaryFile, TemporaryDirectory
import xml.etree.ElementTree as ET

import numpy as np
import cv2

import partitura as pt
from partitura.io.musescore import MuseScoreNotFoundException, FileImportException

from .Linearizer import Linearizer
from .Delinearizer import Delinearizer
from .symbolic.MxlFile import MxlFile
from .symbolic.part_to_score import part_to_score


def delinearize_lmx(input_lmx:str) -> str:
  delinearizer = Delinearizer( errout=sys.stderr )
  delinearizer.process_text( input_lmx )
  score_etree = part_to_score( delinearizer.part_element )
  
  output_xml = str(
    ET.tostring(
      score_etree.getroot(),
      encoding="utf-8",
      xml_declaration=True
    ), 
    "utf-8"
  )

  return output_xml

def render_xml_with_musescore(
  xml:str,
  fmt='png', # png, pdf
  dpi:int=300,
  out=None,
  mscore_exec='musescore3',
):
  if shutil.which(mscore_exec) is None:
    warnings.warn(f"MuseScore executable '{mscore_exec}' not found; skipping score rendering")
    return None

  with (
    NamedTemporaryFile(suffix=".musicxml", mode="w+") as xml_f,
    TemporaryDirectory() as tmpdir
  ):
    
    if out is None:
      out = Path(tmpdir)
    else:
      out = Path(out)
    
    xml_f.write(xml)
    xml_f.flush()
    xml_f.seek(0)
    
    pt_score = pt.load_musicxml(xml_f.name)
    
    img_fh = out / f"score.{fmt}"
    xml_fh = out / "score.musicxml"
    pt.save_musicxml(pt_score, xml_fh)
    
    cmd = [
      mscore_exec,
      "-r",
      "{}".format(int(dpi)),
      "-o",
      os.fspath(img_fh),
      os.fspath(xml_fh),
      "-f",
    ]
    try:
      ps = subprocess.run(
        cmd, 
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
      )
      
      if ps.returncode != 0:
        raise Exception(
          "Command {} failed with code {}; stdout: {}; stderr: {}"
          .format(
            cmd,
            ps.returncode,
            ps.stdout.decode("UTF-8"),
            ps.stderr.decode("UTF-8"),
          )
        )
    
    except Exception as e:
      raise Exception(
        'Executing "{}" returned  {}.'
        .format(" ".join(cmd), e),
      )
    
    if fmt == "png":
      # gather all generated image files
      img_files = list(sorted(Path(out).glob(f"*.{fmt}")))
      
      # make background white
      o_i = cv2.imread(img_files[0], cv2.IMREAD_UNCHANGED)
      transparent_mask = o_i[:,:,3] == 0
      o_i[transparent_mask] = [255, 255, 255, 255]
      o_i = cv2.cvtColor(o_i, cv2.COLOR_BGRA2BGR)
      return o_i

def render_xml_with_lilypond(input_xml:str, multi_page=False) -> np.array:
  # save input_xml to a temporary file
  # to convert into a partitura score
  with NamedTemporaryFile(mode='w+', suffix='.xml', encoding='utf-8') as tmp_xml:
    tmp_xml.write( input_xml )  
    score = pt.load_score(tmp_xml.name)
  
  # render the partitura score using lilypond
  if multi_page:
    out_file_paths = render_lilypond_multipage(score)
    out_images = []
    
    for o_f_p in out_file_paths:
      o_f_p = Path(o_f_p)
      
      out_img = cv2.imread(o_f_p)
      out_images.append(out_img)
      
      o_f_p.unlink()
    
    return out_images
  
  else:
    out_file_path = pt.display.render_lilypond(score)
    out_file_path = Path(out_file_path)
  
    # read the rendered image
    rendered = cv2.imread(out_file_path)
    
    # remove the temp file
    out_file_path.unlink()
  
    return rendered


def render_lilypond_multipage(
  score_data,
  fmt:str="png",
) -> list[str]:
  """
  Render a score-like object using Lilypond

  Parameters
  ----------
  score_data : Score
  fmt : {'png', 'pdf'}
    Output image format

  Returns
  -------
  out : list of output image file paths
  """
  if fmt not in ("png", "pdf"):
    warnings.warn("warning: unsupported output format")
    return None

  img_sfx = ".{}".format(fmt)

  with (
    TemporaryFile() as xml_fh, 
    NamedTemporaryFile(suffix=img_sfx, delete=False) as img_fh
  ):
    # save part to musicxml in file handle xml_fh
    pt.save_musicxml(score_data, xml_fh)
    # rewind read pointer of file handle before we pass it to musicxml2ly
    xml_fh.seek(0)

    img_stem = img_fh.name[: -len(img_sfx)]

    # convert musicxml to lilypond format (use stdout pipe)
    cmd1 = ["musicxml2ly", "-o-", "-"]
    try:
      ps1 = subprocess.run(
        cmd1, stdin=xml_fh, stdout=subprocess.PIPE, check=False
      )
      if ps1.returncode != 0:
        warnings.warn(
          "Command {} failed with code {}".format(cmd1, ps1.returncode),
          stacklevel=2,
        )
        return None
    except FileNotFoundError as f:
      warnings.warn(
        'Executing "{}" returned  {}.'.format(" ".join(cmd1), f),
        ImportWarning,
        stacklevel=2,
      )
      return None

    # convert lilypond format (read from pipe of ps1) to image, and save to
    # temporary filename
    cmd2 = [
      "lilypond",
      "--{}".format(fmt),
      "-dprint-pages",
      "-o{}".format(img_stem),
      "-",
    ]
    try:
      ps2 = subprocess.run(cmd2, input=ps1.stdout, check=False)
      if ps2.returncode != 0:
        warnings.warn(
          "Command {} failed with code {}".format(cmd2, ps2.returncode),
          stacklevel=2,
        )
        return None
    except FileNotFoundError as f:
      warnings.warn(
        'Executing "{}" returned {}.'.format(" ".join(cmd2), f),
        ImportWarning,
        stacklevel=2,
      )
      return
    
    # remove the empty page image
    Path(img_fh.name).unlink()
    
    img_paths = list(sorted(
      Path(img_fh.name).parent.glob(f'{Path(img_stem).stem}-page*.{fmt}')
    ))

    return img_paths