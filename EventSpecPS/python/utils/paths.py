import os


def resolve_data_path(data_dir: str, value: str) -> str:
  """Resolve legacy repo-relative and new scene-relative data paths."""
  if not value:
    return ""
  if os.path.isabs(value):
    return value

  candidates = [os.path.join(data_dir, value), value]
  base = os.path.abspath(data_dir)
  for _ in range(8):
    base = os.path.dirname(base)
    candidates.append(os.path.join(base, value))

  for candidate in candidates:
    if os.path.exists(candidate):
      return os.path.abspath(candidate)
  return os.path.abspath(os.path.join(data_dir, value))


def prepare_output_dirs(output_dir: str):
  output_dir = os.path.abspath(output_dir)
  cache_dir = os.path.join(output_dir, "cache")
  vis_dir = os.path.join(output_dir, "visualizations")
  os.makedirs(cache_dir, exist_ok=True)
  os.makedirs(vis_dir, exist_ok=True)
  return output_dir, cache_dir, vis_dir
