def slice_by_measure(measure_map:tuple, start_measure:int, num_measures:int, margin=0.1):
  measure_ids = [m[1] for m in measure_map]
  
  try:
    start_idx = measure_ids.index(start_measure)
    end_idx = measure_ids.index(start_measure + num_measures)
  except ValueError:
    return None, None
  if end_idx != start_idx + num_measures:
    # The measure map is not continuous
    return None, None
  start_sec = measure_map[start_idx][0]
  end_sec = measure_map[end_idx][0]
  return start_sec-margin, end_sec-margin/2

def get_measure_boundary_from_lmx(lmx:str):
  measure_boundaries = [i for i, t in enumerate(lmx.split(' ')) if t == 'measure']
  return measure_boundaries