import os
import math
import hashlib
import configparser
import numpy as np
import argparse
from scipy.stats import ortho_group

RENDER_RESOLUTION = 256
HYPERSPECTRAL_N_BANDS = 8

def dataset_root(dataset, key, mode):
  if key:
    return f"{dataset}_{key}_{mode}"
  return f"{dataset}_{mode}"


def generate_config(render_id, dataset, mode, hyperspectral, rng, key="", geometry_root="data"):
  seed = rng.integers(65536)
  transform_v = np.zeros((3, 4), dtype=np.float32)
  transform_v[:, :3] = ortho_group.rvs(3, random_state=rng) * rng.uniform(0.5, 1.5)
  if dataset == "blobs":
    if mode == "training":
      scene_id = rng.integers(1, 8 + 1)
    elif mode == "eval":
      scene_id = rng.integers(9, 10 + 1)
  elif dataset == "sculptures":
    if mode == "training":
      scene_id = rng.integers(0, 12 + 1)
    elif mode == "eval":
      scene_id = rng.integers(12, 15 + 1)
  config = configparser.ConfigParser(interpolation=None)
  config["main"] = {
    "log_level": "info",
    "loader": "render",
  }
  if hyperspectral:
    mode = mode + "_hyperspectral"
    save_video_ext = "xz"
  else:
    save_video_ext = "vif"
  root = dataset_root(dataset, key, mode)
  save_video = f"data/{root}/{render_id:06}/frames.{save_video_ext} " + \
               f"data/{root}/{render_id:06}/normal.{save_video_ext}"
  scan_pattern = rng.choice(["circle"]) #, "hypotrochoid", "diligent"
  circle_latitude = rng.integers(30, 60 + 1)  # change to 50, 61
  circle_latitude = circle_latitude
  hypotrochoid_r_big = rng.uniform(0.2, 0.3)
  hypotrochoid_r_small = rng.uniform(0.6, 0.7)
  event_threshold_mean = rng.uniform(0.05, 0.25)
  event_threshold_mean = 0.10
  event_threshold_std = rng.uniform(0.05, 0.35) * event_threshold_mean
  event_threshold = rng.uniform(0.80, 1.20) * event_threshold_mean #
  event_refractory = rng.integers(300, 2500) #rng.integers(300, 2500) # us
  event_refractory = 300
  duration = rng.uniform(0.2, 0.4) # 6 / 30 seconds to 12 / 30 seconds
  texture_resolution = rng.choice([16, 32, 64])
  config["ps"] = {
    "scan_pattern": "circle_with_calibration" if scan_pattern == "circle" else scan_pattern,
    "circle_light_diameter": 2 * math.cos(circle_latitude / 180 * math.pi),
    "circle_object_distance": math.sin(circle_latitude / 180 * math.pi),
    "circle_view_width": 0.,
    "hypotrochoid_r_big": hypotrochoid_r_big,
    "hypotrochoid_r_small": hypotrochoid_r_small,
    "event_threshold": event_threshold,
    "event_refractory": event_refractory,
    "event_refractory_threshold_min": event_refractory,
    "event_refractory_threshold_max": 8192,      # us
    "record_time": 131072,                       # us, about 1-2 rounds
    "record_n_bin": 16,
    "vis_half_life": 2048,                       # us, about 1/16 rounds
    "ls_ps_half_life": 32768,                    # us, about one round
    "show_ls_ps": "none",
    "ps_fcn_per_n_bin": 0,
    "ps_fcn_python": "python/ps_fcn_eval.py",
    "cnn_ps_per_n_bin": 0,
    "cnn_ps_python": "python/cnn_ps_eval.py",
    "cnn_ps_half_life": 32768,                   # us, about one round
  }
  config["loader_render"] = {
    "obj_file": os.path.join(geometry_root, f"{dataset}_processed", f"{scene_id:06}.obj"),
    "client_connect": "/tmp/libredr_client.sock",
    "client_unix": True,
    "client_tls": False,
    # Transform for the beginning
    "transform_v": " ".join(map(str, transform_v.flatten().tolist())),
    # Apply this transform to the left of transform_v per frame
    "transform_v_frame": "1. 0. 0. 0. | 0. 1. 0. 0. | 0. 0. 1. 0.",
    "scan_pattern": scan_pattern,
    "circle_latitude": circle_latitude,
    "hypotrochoid_r_big": hypotrochoid_r_big,
    "hypotrochoid_start_big": 0.,
    "hypotrochoid_r_small": hypotrochoid_r_small,
    "hypotrochoid_start_small": 0.,
    "n_frames": 600,
    "n_rounds": 6,
    "duration": duration,
    "width": RENDER_RESOLUTION,
    "height": RENDER_RESOLUTION,
    "hyperspectral_enable": hyperspectral,
    "hyperspectral_n_bands": HYPERSPECTRAL_N_BANDS,
    "n_texture_bases": 8,
    "n_texture_keys": 4,
    "rainbow_cycle_per_round": 3.,
    "rainbow_reverse": False,
    "texture_resolution": texture_resolution,
    "specular_enable": True,
    "event_threshold_mean": event_threshold_mean,
    "event_threshold_std": event_threshold_std,
    "event_refractory": event_refractory * 1e-6,
    "show_video": "cv",
    # "load_video": save_video,
    "save_video": save_video,
    "save_normal": f"data/{root}/{render_id:06}/normal.xz",
    "save_texture": f"data/{root}/{render_id:06}/texture.xz",
    "save_event": f"data/{root}/{render_id:06}/event_internal.xz " + \
                  f"data/{root}/{render_id:06}/event_trigger",
    "seed": seed,
  }
  config["loader_event_reader"] = {
    "width": RENDER_RESOLUTION,
    "height": RENDER_RESOLUTION,
    "load_event": f"data/{root}/{render_id:06}/event.xz data/{root}/{render_id:06}/event_trigger",
    "playback_speed": 0.,
    "flush_interval": 2048, # us
  }
  os.makedirs(f"data/{root}/{render_id:06}/", exist_ok=True)
  with open(f"data/{root}/{render_id:06}/render.ini", "w") as f:
    config.write(f)

def main():
  parser = argparse.ArgumentParser(description="Generate render configs")
  parser.add_argument("key", nargs="?", default="", help="Path key inserted into data folder name")
  parser.add_argument("--geometry-root", default="data",
                      help="Directory containing blobs_processed/ and sculptures_processed/ (default: data)")
  args = parser.parse_args()
  key = args.key

  for dataset, mode, n_render in [("blobs",      "training", 20),
                                  ("blobs",      "eval",     20),
                                  ("sculptures", "training", 100),
                                  ("sculptures", "eval",     20)]:
    for hyperspectral in [True, False]:
      for render_id in range(n_render):
        seed = int.from_bytes(hashlib.md5(str((dataset, mode, render_id)).encode()).digest()[:4])
        rng = np.random.default_rng(seed=seed)
        generate_config(render_id, dataset, mode, hyperspectral, rng, key=key,
                        geometry_root=args.geometry_root)

if __name__ == "__main__":
  main()
