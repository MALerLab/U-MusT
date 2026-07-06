import os
import subprocess
from pathlib import Path
from typing import Union, Optional

from tqdm.auto import tqdm

import pdfplumber
import json
import tempfile
import shutil

from .utils import PathLike

def convert_musicxml_to_mscx(
  xml_path:PathLike,
  out_dir:PathLike,
  script_path, 
  style_path:Union[PathLike,None]=None,
  virtual_display=':99',
):
  if not isinstance(out_dir, Path):
    out_dir = Path(out_dir)
  if not isinstance(xml_path, Path):
    xml_path = Path(xml_path)
  
  env = os.environ.copy()
  env.update({
    'DISPLAY': virtual_display,
    'QT_QPA_PLATFORM': 'xcb',
    'QT_X11_NO_MITSHM': '1',
    'XDG_RUNTIME_DIR': '/tmp'
  })
  
  process_configs = [
    ['Xvfb', virtual_display, '-screen', '0', '2560x1440x24', '-ac'],
    [script_path]
  ]
  
  processes = spawn_processes(process_configs, env)
  
  out_path = (out_dir / f"{xml_path.stem}.mscx")
  
  if style_path is not None:
    cmd = [script_path, "-S", style_path, "-o", str(out_path), str(xml_path)]
  else:
    cmd = [script_path, "-o", str(out_path), str(xml_path)]
  
  try:
    # convert MusicXML to .mscx file
    ps = subprocess.run(
      cmd, env=env,
      stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )
    
    if ps.returncode != 0:
      raise Exception(
        "Command {} failed with code {}. MuseScore " "error messages:\n {}"
        .format(cmd, ps.returncode, ps.stderr.decode("UTF-8"))
      )
  
  except Exception as e:
    raise Exception(
      'Executing "{}" \nreturned  {}.'.format(" ".join(cmd), e)
    )
  
  terminate_processes(processes)
  
  return out_path


def render_mscx(
  mscx_path:Union[PathLike,None],
  out_path:PathLike,
  env,
  dpi:Optional[int]=300,
  mscore_exec:str='./mscore',
  style_path:Union[PathLike,None]=None,
) -> PathLike:
  """
  render .mscx file as .pdf using MuseScore

  Parameters
  ----------
  mscx_path : PathLike or None
    MuseScore file path to be rendered
  out_path : Path to output PDF file
  dpi : int, optional
    Image resolution. 
    This option is ignored when `fmt` is 'pdf'. 
    Defaults to 90.

  Returns
  -------
  out : PathLike
    Path to the output PDF file if rendering was successful, 
    otherwise None.
  """

  img_fh = Path(out_path)
  
  if style_path is not None:
    cmd = [
      mscore_exec,
      "-S", style_path, 
      "-r",
      "{}".format(int(dpi)),
      "-o",
      str(img_fh),
      str(mscx_path),
    ]
  else:
    cmd = [
      mscore_exec,
      "-r",
      "{}".format(int(dpi)),
      "-o",
      str(img_fh),
      str(mscx_path),
    ]
  
  try:
    ps = subprocess.run(
      cmd, env=env,
      stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    
    if ps.returncode != 0:
      raise Exception(
        "Command {} failed with code {}; stdout: {}; stderr: {}"
        .format(
          " ".join(cmd),
          ps.returncode,
          ps.stdout.decode("UTF-8"),
          ps.stderr.decode("UTF-8"),
        )
      )
    
    return img_fh if img_fh.exists() else None
  
  except Exception as e:
    raise Exception(
      'Executing "{}" returned  {}.'
      .format(" ".join(cmd), e),
    )
  
  return None

def convert_mscx_to_pdf(
  mscx_path:PathLike,
  out_path:PathLike,
  script_path,
  display_id=':99',
  style_path:Union[PathLike,None]=None,
):
  
  env = os.environ.copy()
  env.update({
      'DISPLAY': display_id,
      'QT_QPA_PLATFORM': 'xcb',
      'QT_X11_NO_MITSHM': '1',
      'XDG_RUNTIME_DIR': '/tmp'
  })
  
  process_configs = [
    ['Xvfb', display_id, '-screen', '0', '2560x1440x24', '-ac'],
    [script_path]
  ]
  
  processes = spawn_processes(process_configs, env)
    
  _ = render_mscx(
        mscx_path=mscx_path,
        out_path=out_path,
        env=env,
        dpi=300,
        mscore_exec=script_path,
        style_path=style_path,
      )
  
  # Clean up Xvfb
  terminate_processes(processes)

def render_mxl(
  mxl_path:Union[PathLike,None],
  out_path:PathLike,
  env,
  dpi:Optional[int]=300,
  mscore_exec:str='./mscore',
  style_path:Union[PathLike,None]=None,
) -> PathLike:
  """
  render .mxl file as .pdf using MuseScore

  Parameters
  ----------
  mxl_path : PathLike or None
    MusicXML file path to be rendered
  out_path : Path to output PDF file
  dpi : int, optional
    Image resolution. 
    This option is ignored when `fmt` is 'pdf'. 
    Defaults to 90.

  Returns
  -------
  out : PathLike
    Path to the output PDF file if rendering was successful, 
    otherwise None.
  """

  img_fh = Path(out_path)
  
  # Build MuseScore CLI command with xvfb-run
  if style_path is not None:
    cmd = [
      "xvfb-run", "-a",
      mscore_exec,
      "-n",
      "-S", str(Path(style_path).resolve()),
      "-r",
      "{}".format(int(dpi)),
      "-o",
      str(img_fh),
      str(mxl_path),
    ]
  else:
    cmd = [
      "xvfb-run", "-a",
      mscore_exec,
      "-n",
      "-r",
      "{}".format(int(dpi)),
      "-o",
      str(img_fh),
      str(mxl_path),
    ]
  
  try:
    ps = subprocess.run(
      cmd, env=env,
      stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    
    if ps.returncode != 0:
      raise Exception(
        "Command {} failed with code {}; stdout: {}; stderr: {}"
        .format(
          " ".join(cmd),
          ps.returncode,
          ps.stdout.decode("UTF-8"),
          ps.stderr.decode("UTF-8"),
        )
      )
    
    return img_fh if img_fh.exists() else None
  
  except Exception as e:
    raise Exception(
      'Executing "{}" returned  {}.'
      .format(" ".join(cmd), e),
    )
  
  return None

def convert_mxl_to_pdf(
  mxl_path:PathLike,
  out_path:PathLike,
  script_path,
  display_id=':99',
  style_path:Union[PathLike,None]=None,
):
  
  env = os.environ.copy()
  env.update({
      'DISPLAY': display_id,
      'QT_QPA_PLATFORM': 'xcb',
  })
    
  _ = render_mxl(
        mxl_path=mxl_path,
        out_path=out_path,
        env=env,
        dpi=300,
        mscore_exec=script_path,
        style_path=style_path,
      )
    
def split_pdf(
    pdf_path:PathLike,
    out_dir:PathLike,
    score_name:str,
):
  if not isinstance(pdf_path, Path):
    pdf_path = Path(pdf_path)
  if not isinstance(out_dir, Path):
    out_dir = Path(out_dir)
  if not pdf_path.exists():
    print(f"PDF file does not exist:{str(pdf_path)}")
    return
  
  out_dir.mkdir(parents=True, exist_ok=True)
    
  pdf = pdfplumber.open(str(pdf_path))

  for page in pdf.pages:
    page_number = str(page.page_number).zfill(4)
    image = page.to_image(resolution=300)
    
    image_path = out_dir / f'{score_name}_{page_number}.png'
    image.save(str(image_path))
