
import numpy as np
from umust.midi_utils.preprocess.preprocess_maestro import create_note_event_and_note_from_midi
from tqdm.auto import tqdm
from umust.lmx_utils.linearize import linearize
from pathlib import Path
import pandas as pd
import json

asap_dataset_path = Path('dataset/asap-dataset')
assert asap_dataset_path.exists()

metadata = pd.read_csv(asap_dataset_path / 'metadata.csv')
annotations = json.load(open(asap_dataset_path / 'asap_annotations.json'))

error_messages = []
note_events_dir = asap_dataset_path.parent / 'maestro/asap_note_events'
lmx_dir = asap_dataset_path.parent / 'maestro/lmx'
assert note_events_dir.exists()
assert lmx_dir.exists()

for i, row in tqdm(metadata.iterrows(), total=len(metadata)):
  midi_performance_path = asap_dataset_path / row['midi_performance']
  annt_file_path = asap_dataset_path / row['performance_annotations']
  xml_path = asap_dataset_path / 'musescore3_xmls' / row['xml_score'].replace('/', '_')
  # xml_path = asap_dataset_path / 
  assert midi_performance_path.exists()
  assert annt_file_path.exists()
  assert xml_path.exists(), f'{xml_path} does not exist'
  
  annt = annotations[row['midi_performance']]
  if not annt['score_and_performance_aligned']:
    print(f'{row["midi_performance"]} is not score and performance aligned')
    continue
  perf_id = row['midi_performance'].replace('.mid', '').replace('/', '_')
  if not isinstance(annt['downbeats_score_map'], list) or len(annt['downbeats_score_map']) != len(annt['performance_downbeats']):
    print(f'{row["midi_performance"]} has invalid downbeats_score_map')
    continue
  
  lmx_fn = lmx_dir / f"{row['composer']}_{row['title']}.lmx"
  if not lmx_fn.exists():
    try:
      lmx_fn.parent.mkdir(parents=True, exist_ok=True)
      print(f"Linearizing {xml_path}")
      lmx = linearize(str(xml_path))
      print(f"Saved {lmx_fn}")
      with open(lmx_fn, 'w') as f:
        f.write(lmx)
    except Exception as e:
      print(f"Error linearizing {xml_path}: {e}")
      error_messages.append([xml_path, str(e)])
      continue

  notes, note_events = create_note_event_and_note_from_midi(midi_performance_path, perf_id, ignore_pedal=False)
  perf_measure_map = [(p, d) for p, d in zip(annt['performance_downbeats'], annt['downbeats_score_map'])]
  note_events['measure_map'] = perf_measure_map
  if type(row['start']) is float and row['start'] > 0:
    note_events['audio_start'] = row['start']
  if type(row['end']) is float and row['end'] > 0:
    note_events['audio_end'] = row['end']
  note_events_file = note_events_dir / f'{perf_id}_note_events.npy'
  note_events_file.parent.mkdir(parents=True, exist_ok=True)
  # np.save(notes_file, notes, allow_pickle=True, fix_imports=False)
  np.save(note_events_file, note_events, allow_pickle=True, fix_imports=False)
  
for message in error_messages:
  print(message)
