from datetime import datetime
import argparse
import subprocess
import shutil
from pathlib import Path
from typing import Union

import json

from tqdm.auto import tqdm

get_ts = lambda: datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
dformat = lambda d: json.dumps(d, indent=2)
dprint = lambda d: print(dformat(d))

PathLike = Union[Path, str]


def get_argument_parser(args=None):
  parser = argparse.ArgumentParser()

  if args is None:
    args = [
      ('-d', '--base_dir', True, str),
    ]

  for short, full, required, type_ in args:
    parser.add_argument(
      short,
      full,
      required=required,
      type=type_
    )
  
  return parser


# <-------------------- Deprecated -------------------->

def delete_old_musicxml_files(metadata, score_dir):
  """
  Delete all MusicXML files in the data directory.
  """
  for ossq_id, infos in metadata.items():
    ossq_id = f'sq{ossq_id}'
    
    mscore_dir = score_dir / infos['path'] 
    musicxml_path = mscore_dir / f"{ossq_id}.musicxml"
    
    image_paths = (mscore_dir / 'images' / 'original').glob('*.png')
    image_paths = list(sorted(image_paths))
    
    assert musicxml_path.exists(), f"old score: {str(musicxml_path)} does not exist"
    assert all([ p.exists() for p in image_paths ]), f"old score: some images do not exist"
    
    # delete old files
    musicxml_path.unlink()
    for p in image_paths:
      p.unlink()


def merge_external_files(metadata, score_dir, out_dir):
  score_pbar = tqdm(metadata.items())
  
  for ossq_id, infos in score_pbar:
    ossq_id = f'sq{ossq_id}'
    
    score_pbar.set_description(f"{ossq_id}")
    
    new_mscore_dir = out_dir / infos['path']
    new_musicxml_path = new_mscore_dir / f'{ossq_id}.musicxml'
    new_image_paths = (new_mscore_dir / 'images' / 'original').glob('*.png')
    new_image_paths = list(sorted(new_image_paths))
    
    assert new_musicxml_path.exists(), f"{str(new_musicxml_path)} does not exist"
    assert all([ p.exists() for p in new_image_paths ]), f"some images do not exist"
    
    mscore_dir = score_dir / infos['path'] 
    musicxml_path = mscore_dir / f"{ossq_id}.musicxml"
    image_dir = mscore_dir / 'images' / 'original'
    
    # copy new files
    shutil.copy(new_musicxml_path, musicxml_path)
    for src_img in new_image_paths:
      shutil.copy(src_img, image_dir / src_img.name)


def delete_old_lmxe(metadata, score_dir):
  score_pbar = tqdm(metadata.items())
  for mscore_id, infos in score_pbar:
    ossq_id = f'sq{mscore_id}'
    score_pbar.set_description(f"{ossq_id}")
    
    mscore_dir = score_dir / infos['path'] 
    lmxe_paths = (mscore_dir / 'lmxe').glob('*.lmxe')
    
    for p in lmxe_paths:
      p.unlink()
