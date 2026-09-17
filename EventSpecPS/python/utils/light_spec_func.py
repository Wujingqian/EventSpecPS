import numpy as np

def rainbow_spec_fuc(time, event_trigger_time, n_bands, cycle_per_round, rainbow_reverse=True):
  t = time
  K = int(n_bands)
  pos = t / event_trigger_time * np.float32(cycle_per_round)
  if rainbow_reverse:
    u = np.minimum(pos % 2.0, 2.0 - (pos % 2.0)) * (K - 1) / K
  else:
    u = pos % 1.0
  s = u * K
  i0 = np.floor(s).astype(int);
  frac = s - np.floor(s)
  i1 = np.clip(i0 + 1, 0, K - 1) if rainbow_reverse else (i0 + 1) % K
  S = np.zeros((t.shape[0], K), np.float32);
  rows = np.arange(t.shape[0])
  S[rows, i0] += (1 - frac).astype(np.float32);
  S[rows, i1] += frac.astype(np.float32)
  return S

def circle_light_dir_fuc(time: float, latitude: float, event_trigger_time: float) -> np.ndarray:
  longitude = time / event_trigger_time * 2.0 * np.pi
  return np.array([
    np.cos(latitude) * np.sin(longitude),
    np.cos(latitude) * np.cos(longitude),
    np.broadcast_to(np.sin(latitude), longitude.shape)
  ], dtype=np.float32)