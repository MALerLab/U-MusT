import os
import sys
import shutil
import subprocess
import warnings
from typing import Union, Optional
from pathlib import Path
from tempfile import TemporaryFile, TemporaryDirectory
import xml.etree.ElementTree as ET

import numpy as np
import cv2

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
  mscore_exec=None,
):
  if mscore_exec is None:
    candidates = ["mscore3", "musescore3", "mscore", "musescore"]
    mscore_exec = next((c for c in candidates if shutil.which(c)), None)
    if mscore_exec is None:
      warnings.warn(f"No MuseScore executable found among {candidates}; skipping score rendering")
      return None
  elif shutil.which(mscore_exec) is None:
    warnings.warn(f"MuseScore executable '{mscore_exec}' not found; skipping score rendering")
    return None

  with TemporaryDirectory() as tmpdir:

    if out is None:
      out = Path(tmpdir)
    else:
      out = Path(out)

    img_fh = out / f"score.{fmt}"
    xml_fh = out / "score.musicxml"
    xml_fh.write_text(xml)

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

      first_page = None
      for img_file in img_files:
        # make background white
        o_i = cv2.imread(os.fspath(img_file), cv2.IMREAD_UNCHANGED)
        transparent_mask = o_i[:,:,3] == 0
        o_i[transparent_mask] = [255, 255, 255, 255]
        o_i = cv2.cvtColor(o_i, cv2.COLOR_BGRA2BGR)

        if first_page is None:
          first_page = o_i

        gray = cv2.cvtColor(o_i, cv2.COLOR_BGR2GRAY)
        if (gray < 200).mean() >= 1e-4:
          return o_i

      warnings.warn("Rendered score came out blank on every page")
      return first_page

