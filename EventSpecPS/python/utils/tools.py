import os
import lzma
import torch
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from ..utils.light_spec_func import circle_light_dir_fuc, rainbow_spec_fuc

def data_load_hyper(normal_gt_path, buffer_path, texture_path, frames_path, hyper_c, height=256, width=256):
  normal_gt = load_normal_xz(normal_gt_path, height, width)
  buffer = load_event_array(buffer_path)
  texture_diffuse = load_texture_xz(texture_path, hyper_c, height, width)
  frames_data = load_frames_xz(frames_path, hyper_c, height, width)
  return buffer, normal_gt, texture_diffuse, frames_data

def load_event_array(path):
  with open(path, "rb") as f:
    data = lzma.decompress(f.read())
  events = np.frombuffer(data, dtype=np.float32)

  if events.size % 4 != 0:
    raise ValueError("Event array size is not divisible by 4. Corrupted?")

  events = events.reshape(-1, 4).copy()

  return events

def load_texture_xz(path, hyper_c, height=256, width=256):
  if path == "":
    return torch.zeros((hyper_c, height, width), dtype=torch.float32)
  with open(path, "rb") as f:
    data = lzma.decompress(f.read())
  temp_array = np.frombuffer(data, dtype=np.float32)
  texture_data = temp_array.reshape(-1, height, width)
  render_diffuse = texture_data[3:3+hyper_c,:,:]
  return torch.tensor(render_diffuse, dtype=torch.float32)

def load_normal_xz(path, height=256, width=256):
  if path == "":
    return torch.zeros((3, height, width), dtype=torch.float32)
  with open(path, "rb") as f:
    data = lzma.decompress(f.read())
  normal_gt = np.frombuffer(data, dtype=np.float32).reshape(-1, 3, height, width)[0]
  return torch.tensor(normal_gt, dtype=torch.float32)

def load_frames_xz(path: str, n_bands: int, height: int, width: int) -> torch.Tensor:
  if path == "":
    return torch.zeros((1, n_bands, height, width), dtype=torch.float32)
  with open(path, "rb") as f:
    data = lzma.decompress(f.read())
  arr = np.frombuffer(data, dtype=np.float32)
  elems_per_t = n_bands * height * width
  n_frame = arr.size // elems_per_t
  return torch.from_numpy(arr.reshape(n_frame, n_bands, height, width).copy())

def visualize_array(x, gt, flag="normal", data_path=None):
  if flag == "normal":
    def transform(x):
      return (x + 1.0) * 0.5
    title = "Normal Map"

  elif flag == "ref":
    def transform(x):
      x = x[..., :3]
      return x
    title = "Ref visualization"

  else:
    raise ValueError(f"Unknown flag: {flag}")

  transform_clip = lambda x: np.clip(transform(x), 0.0, 1.0)
  vis_x, vis_gt = transform_clip(x), transform_clip(gt)
  plt.figure(figsize=(12, 5))
  plt.subplot(1, 2, 1)
  plt.imshow(vis_x)
  plt.axis("off")
  plt.title(title)

  plt.subplot(1, 2, 2)
  plt.imshow(vis_gt)
  plt.axis("off")
  plt.show(block=False)

  if data_path:
    save_path = os.path.join(data_path, f"{flag}_vis.png")
    vis_u8 = (np.concatenate([vis_x, vis_gt], axis=1) * 255).astype(np.uint8)
    cv.imwrite(save_path, vis_u8[..., ::-1])  # RGB -> BGR
    print(f"Saved visualization to {save_path}")

def visualize_error_maps(normal_err, ref_err, data_path=None):
  plt.figure(figsize=(12, 5))
  plt.subplot(1, 2, 1)
  plt.imshow(normal_err, vmin=0, vmax=30)
  plt.axis('off')

  plt.subplot(1, 2, 2)
  plt.imshow(ref_err, vmin=0, vmax=30)
  plt.axis('off')

  plt.tight_layout()
  plt.show(block=False)
  plt.pause(0.1)

  if data_path:
    save_path = os.path.join(data_path, "error_maps.png")
    vis_u8 = (np.concatenate([normal_err, ref_err], axis=1) * 255).astype(np.uint8)
    cv.imwrite(save_path, vis_u8[..., ::-1])

def generate_light_directions(time, pattern_params):
  if pattern_params["spec_pattern"] == "rainbow":
    light_spec = rainbow_spec_fuc(time, pattern_params["event_trigger_time"],
                                pattern_params["hyperspectral_n_bands"], pattern_params["rainbow_cycle_per_round"],
                                pattern_params["rainbow_reverse"])
    light_spec_prev = rainbow_spec_fuc(time + pattern_params["event_refractory"],
                                     pattern_params["event_trigger_time"],
                                     pattern_params["hyperspectral_n_bands"], pattern_params["rainbow_cycle_per_round"],
                                     pattern_params["rainbow_reverse"])
  else:
    raise ValueError(f"Unsupported scan pattern: {pattern_params['scan_pattern']}")
  if pattern_params["light_pattern"] == "circle":
    light_dir = circle_light_dir_fuc(time, pattern_params["circle_latitude"],
                                     pattern_params["event_trigger_time"]).T
    light_dir_prev = circle_light_dir_fuc(time + pattern_params["event_refractory"],
                                          pattern_params["circle_latitude"],
                                          pattern_params["event_trigger_time"]).T
  else:
    raise ValueError(f"Unsupported scan pattern: {pattern_params['scan_pattern']}")

  return light_dir, light_dir_prev, light_spec, light_spec_prev, np.exp(pattern_params["event_threshold"])
