import os
import math
import argparse
import configparser
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import torch
import torchvision
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from ..utils.tools import data_load_hyper, visualize_array, visualize_error_maps, generate_light_directions
from ..utils.paths import resolve_data_path, prepare_output_dirs

DEVICE = torch.device("cuda")
LOW_RANK_K = 8


def current_git_commit():
  try:
    return subprocess.check_output(
      ["git", "rev-parse", "HEAD"],
      cwd=Path(__file__).resolve().parents[3],
      text=True,
      stderr=subprocess.DEVNULL,
    ).strip()
  except (OSError, subprocess.CalledProcessError):
    return None

def build_event_pairs(events, pattern_params, rows, cols, cache_path):
  if cache_path is not None and os.path.isfile(cache_path):
    print("cache loaded")
    return torch.load(cache_path, map_location="cpu")
  xs, ys, ts = [], [], []
  light_dir_high_list, light_dir_low_list = [], []
  light_spec_high_list, light_spec_low_list = [], []
  toT = lambda x: torch.as_tensor(np.concatenate(x, axis=0))
  pbar = tqdm(range(rows), desc="Processing Rows")
  for row in pbar:
    for col in range(cols):
      sub_mask = (events[:, 2] == row) & (events[:, 3] == col)
      pixel_events = events[sub_mask, :]
      if pixel_events.shape[0] < 2:
        continue
      light_dir, light_dir_prev, light_spec, light_spec_prev, epC = \
        generate_light_directions(pixel_events[:,0], pattern_params)

      light_dir_high_pos = light_dir[1:, :][pixel_events[1:, 1] == 1]
      light_dir_low_pos = light_dir_prev[:-1, :][pixel_events[1:, 1] == 1]
      light_dir_low_neg = light_dir[1:, :][pixel_events[1:, 1] == -1]
      light_dir_high_neg = light_dir_prev[:-1, :][pixel_events[1:, 1] == -1]
      light_dir_high = np.concatenate([light_dir_high_pos, light_dir_high_neg], axis=0)
      light_dir_low = np.concatenate([light_dir_low_pos, light_dir_low_neg], axis=0)

      light_spec_high_pos = light_spec[1:, :][pixel_events[1:, 1] == 1]
      light_spec_low_pos = light_spec_prev[:-1, :][pixel_events[1:, 1] == 1]
      light_spec_low_neg = light_spec[1:, :][pixel_events[1:, 1] == -1]
      light_spec_high_neg = light_spec_prev[:-1, :][pixel_events[1:, 1] == -1]
      light_spec_high = np.concatenate([light_spec_high_pos, light_spec_high_neg], axis=0)
      light_spec_low = np.concatenate([light_spec_low_pos, light_spec_low_neg], axis=0)

      n_pairs = light_spec_high.shape[0]
      xs.append(np.full((n_pairs,), row, dtype=np.int64))
      ys.append(np.full((n_pairs,), col, dtype=np.int64))
      light_dir_high_list.append(light_dir_high), light_dir_low_list.append(light_dir_low)
      light_spec_high_list.append(light_spec_high), light_spec_low_list.append(light_spec_low)

  out = {
    "x": toT(xs),
    "y": toT(ys),
    "light_dir_high": toT(light_dir_high_list),
    "light_dir_low": toT(light_dir_low_list),
    "light_spec_high": toT(light_spec_high_list),
    "light_spec_low": toT(light_spec_low_list),
    "C": torch.tensor(epC),
  }
  torch.save(out, cache_path)
  return out


def build_event_pairs_fast(events, pattern_params, rows, cols, cache_path, num_workers=8):
  """
  Faster version of build_event_pairs:
  - Group events by pixel only once via sort/unique.
  - Process pixel groups in parallel.
  Interface is kept compatible with build_event_pairs.
  """
  if cache_path is not None and os.path.isfile(cache_path):
    print("cache loaded")
    return torch.load(cache_path, map_location="cpu")

  if events.shape[0] == 0:
    empty_i64 = torch.empty((0,), dtype=torch.int64)
    empty_f32_3 = torch.empty((0, 3), dtype=torch.float32)
    empty_f32_k = torch.empty((0, pattern_params["hyperspectral_n_bands"]), dtype=torch.float32)
    out = {
      "x": empty_i64,
      "y": empty_i64,
      "light_dir_high": empty_f32_3,
      "light_dir_low": empty_f32_3,
      "light_spec_high": empty_f32_k,
      "light_spec_low": empty_f32_k,
      "C": torch.tensor(np.exp(pattern_params["event_threshold"]), dtype=torch.float32),
    }
    torch.save(out, cache_path)
    return out

  x = events[:, 2].astype(np.int64)
  y = events[:, 3].astype(np.int64)
  valid = (x >= 0) & (x < rows) & (y >= 0) & (y < cols)
  if not np.all(valid):
    events = events[valid]
    x = x[valid]
    y = y[valid]

  if events.shape[0] == 0:
    empty_i64 = torch.empty((0,), dtype=torch.int64)
    empty_f32_3 = torch.empty((0, 3), dtype=torch.float32)
    empty_f32_k = torch.empty((0, pattern_params["hyperspectral_n_bands"]), dtype=torch.float32)
    out = {
      "x": empty_i64,
      "y": empty_i64,
      "light_dir_high": empty_f32_3,
      "light_dir_low": empty_f32_3,
      "light_spec_high": empty_f32_k,
      "light_spec_low": empty_f32_k,
      "C": torch.tensor(np.exp(pattern_params["event_threshold"]), dtype=torch.float32),
    }
    torch.save(out, cache_path)
    return out

  key = x * cols + y
  order = np.argsort(key, kind="mergesort")
  events_sorted = events[order]
  key_sorted = key[order]

  unique_key, starts, counts = np.unique(key_sorted, return_index=True, return_counts=True)

  def process_chunk(begin, end):
    xs_local, ys_local = [], []
    ldh_local, ldl_local = [], []
    lsh_local, lsl_local = [], []
    for i in range(begin, end):
      cnt = int(counts[i])
      if cnt < 2:
        continue
      st = int(starts[i])
      pixel_events = events_sorted[st:st + cnt]
      light_dir, light_dir_prev, light_spec, light_spec_prev, _ = \
        generate_light_directions(pixel_events[:, 0], pattern_params)

      pos_mask = (pixel_events[1:, 1] == 1)
      neg_mask = (pixel_events[1:, 1] == -1)
      n_pairs = int(pos_mask.sum() + neg_mask.sum())
      if n_pairs == 0:
        continue

      light_dir_high_pos = light_dir[1:, :][pos_mask]
      light_dir_low_pos = light_dir_prev[:-1, :][pos_mask]
      light_dir_low_neg = light_dir[1:, :][neg_mask]
      light_dir_high_neg = light_dir_prev[:-1, :][neg_mask]
      light_dir_high = np.concatenate([light_dir_high_pos, light_dir_high_neg], axis=0)
      light_dir_low = np.concatenate([light_dir_low_pos, light_dir_low_neg], axis=0)

      light_spec_high_pos = light_spec[1:, :][pos_mask]
      light_spec_low_pos = light_spec_prev[:-1, :][pos_mask]
      light_spec_low_neg = light_spec[1:, :][neg_mask]
      light_spec_high_neg = light_spec_prev[:-1, :][neg_mask]
      light_spec_high = np.concatenate([light_spec_high_pos, light_spec_high_neg], axis=0)
      light_spec_low = np.concatenate([light_spec_low_pos, light_spec_low_neg], axis=0)

      pix = int(unique_key[i])
      row, col = divmod(pix, cols)
      xs_local.append(np.full((n_pairs,), row, dtype=np.int64))
      ys_local.append(np.full((n_pairs,), col, dtype=np.int64))
      ldh_local.append(light_dir_high.astype(np.float32, copy=False))
      ldl_local.append(light_dir_low.astype(np.float32, copy=False))
      lsh_local.append(light_spec_high.astype(np.float32, copy=False))
      lsl_local.append(light_spec_low.astype(np.float32, copy=False))
    return xs_local, ys_local, ldh_local, ldl_local, lsh_local, lsl_local

  n_groups = unique_key.shape[0]
  n_workers = max(1, int(num_workers))
  n_workers = min(n_workers, n_groups)

  boundaries = np.linspace(0, n_groups, n_workers + 1, dtype=np.int64)
  future_to_chunk_size = {}
  results = []
  with ThreadPoolExecutor(max_workers=n_workers) as ex:
    for w in range(n_workers):
      b = int(boundaries[w])
      e = int(boundaries[w + 1])
      if b < e:
        fu = ex.submit(process_chunk, b, e)
        future_to_chunk_size[fu] = e - b
    with tqdm(total=n_groups, desc="Processing Pixel Groups (fast)") as pbar_fast:
      for fu in as_completed(future_to_chunk_size):
        results.append(fu.result())
        pbar_fast.update(int(future_to_chunk_size[fu]))

  xs, ys = [], []
  ldh, ldl = [], []
  lsh, lsl = [], []
  for xs_l, ys_l, ldh_l, ldl_l, lsh_l, lsl_l in results:
    xs.extend(xs_l)
    ys.extend(ys_l)
    ldh.extend(ldh_l)
    ldl.extend(ldl_l)
    lsh.extend(lsh_l)
    lsl.extend(lsl_l)

  if len(xs) == 0:
    empty_i64 = torch.empty((0,), dtype=torch.int64)
    empty_f32_3 = torch.empty((0, 3), dtype=torch.float32)
    empty_f32_k = torch.empty((0, pattern_params["hyperspectral_n_bands"]), dtype=torch.float32)
    out = {
      "x": empty_i64,
      "y": empty_i64,
      "light_dir_high": empty_f32_3,
      "light_dir_low": empty_f32_3,
      "light_spec_high": empty_f32_k,
      "light_spec_low": empty_f32_k,
      "C": torch.tensor(np.exp(pattern_params["event_threshold"]), dtype=torch.float32),
    }
    torch.save(out, cache_path)
    return out

  out = {
    "x": torch.from_numpy(np.concatenate(xs, axis=0)),
    "y": torch.from_numpy(np.concatenate(ys, axis=0)),
    "light_dir_high": torch.from_numpy(np.concatenate(ldh, axis=0)),
    "light_dir_low": torch.from_numpy(np.concatenate(ldl, axis=0)),
    "light_spec_high": torch.from_numpy(np.concatenate(lsh, axis=0)),
    "light_spec_low": torch.from_numpy(np.concatenate(lsl, axis=0)),
    "C": torch.tensor(np.exp(pattern_params["event_threshold"]), dtype=torch.float32),
  }
  torch.save(out, cache_path)
  return out

def main():
  parser = argparse.ArgumentParser(description="EventSpecPS: joint normal and multispectral reflectance reconstruction")

  parser.add_argument("--data", type=str, required=True, help="Data folder path")
  parser.add_argument("--output-dir", type=str, default=None, help="Output run directory (default: data folder)")
  parser.add_argument("--max_pixel_iters", type=int, default=1600, help="Maximum iterations per pixel")
  parser.add_argument("--normal-smooth-weight", type=float, default=1e-10)
  parser.add_argument("--albedo-smooth-weight", type=float, default=1e-12)
  parser.add_argument("--constant-light", type=float, default=0.5)
  parser.add_argument("--normal-blur-kernel", type=int, default=21)
  parser.add_argument("--normal-blur-sigma", type=float, default=5.0)
  parser.add_argument("--albedo-blur-kernel", type=int, default=21)
  parser.add_argument("--albedo-blur-sigma", type=float, default=5.0)
  parser.add_argument("--random-seed", type=int, default=None)
  parser.add_argument("--cache-path", type=str, default=None)
  parser.add_argument("--device", default="cuda")
  parser.add_argument("--skip-visualizations", action="store_true")
  parser.add_argument("--quiet", action="store_true")

  args = parser.parse_args()

  if args.normal_smooth_weight < 0 or args.albedo_smooth_weight < 0:
    raise ValueError("Smoothness weights must be non-negative")
  if args.constant_light < 0:
    raise ValueError("constant-light must be non-negative")
  for name, kernel in (
    ("normal-blur-kernel", args.normal_blur_kernel),
    ("albedo-blur-kernel", args.albedo_blur_kernel),
  ):
    if kernel <= 0 or kernel % 2 == 0:
      raise ValueError(f"{name} must be a positive odd integer")
  if args.normal_blur_sigma <= 0 or args.albedo_blur_sigma <= 0:
    raise ValueError("Blur sigma values must be positive")
  global DEVICE
  DEVICE = torch.device(args.device)
  if DEVICE.type == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("CUDA was requested but is not available")
  if args.random_seed is not None:
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    if DEVICE.type == "cuda":
      torch.cuda.manual_seed_all(args.random_seed)

  data_path = args.data
  output_dir, cache_dir, vis_dir = prepare_output_dirs(args.output_dir or data_path)
  render_config_path = os.path.join(data_path, "render.ini")
  config = configparser.ConfigParser()
  config.read(render_config_path)

  scan_pattern = config.get('loader_render', 'scan_pattern')
  hyperspectral_enable = config.getboolean('loader_render', 'hyperspectral_enable')
  normal_path = resolve_data_path(data_path, config.get('loader_render', 'save_normal'))
  texture_path = resolve_data_path(data_path, config.get('loader_render', 'save_texture'))
  event_info = config.get('loader_render', 'save_event')
  frames_path = resolve_data_path(data_path, config.get('loader_render', 'save_video').split(' ')[0])
  event_path, trigger_path = event_info.split(' ')[0], event_info.split(' ')[1]
  event_path = resolve_data_path(data_path, event_path)
  trigger_path = resolve_data_path(data_path, trigger_path)

  with open(trigger_path, 'rb') as f:
    trigger_data = f.read()
  event_trigger = np.frombuffer(trigger_data, dtype=np.float32)
  event_refactory_time = np.float32(config.getfloat('ps', 'event_refractory')) * 1e-6
  event_threshold = np.float32(config.getfloat('ps', 'event_threshold'))
  pattern_params = {}

  assert hyperspectral_enable
  hyperspectral_n_bands = config.getint('loader_render', 'hyperspectral_n_bands')
  n_rounds = config.getint('loader_render', 'n_rounds')
  pattern_params.update({
    "spec_pattern": "rainbow",
    "light_pattern": scan_pattern,
    "event_trigger_time": event_trigger[1] - event_trigger[0],
    "hyperspectral_n_bands": hyperspectral_n_bands,
    "event_threshold": event_threshold,
    "event_refractory": event_refactory_time,
    "rainbow_cycle_per_round": np.float32(config.getfloat('loader_render', 'rainbow_cycle_per_round')),
    "rainbow_reverse": config.getboolean('loader_render', 'rainbow_reverse'),
    "n_rounds": n_rounds
  })

  width = config.getint('loader_render', 'width')
  height = config.getint('loader_render', 'height')
  events, normal_gt, ref_gt, frame_data = \
    data_load_hyper(normal_path, event_path, texture_path, frames_path, hyperspectral_n_bands, height, width)
  events[:, 3] -= event_trigger[0]
  event_mask = (events[:, 3] > 0) & (events[:, 3] < n_rounds * pattern_params['event_trigger_time'])
  events = events[event_mask, :]
  if scan_pattern == "circle":
    circle_light_diameter = np.float32(config.getfloat('ps', 'circle_light_diameter'))
    circle_object_distance = np.float32(config.getfloat('ps', 'circle_object_distance'))
    circle_latitude = math.atan2(circle_object_distance, circle_light_diameter / 2)  # * 180 / math.pi
    pattern_params.update({
      "circle_latitude": circle_latitude,
    })
  else:
    raise ValueError(f"Unsupported scan pattern: {scan_pattern}")

  events = events[:, [3, 2, 0, 1]] #[t, p, x, y]
  normal_gt, ref_gt = normal_gt.permute(1,2,0), ref_gt.permute(1,2,0)
  rows = height
  cols = width
  normal_raw = nn.Parameter(torch.zeros(rows, cols, 3, device=DEVICE), requires_grad=True)
  albedo_raw = nn.Parameter(torch.ones(rows, cols, LOW_RANK_K, device=DEVICE), requires_grad=True)
  ref_color = nn.Parameter(torch.ones(LOW_RANK_K, hyperspectral_n_bands, device=DEVICE), requires_grad=True)
  with torch.no_grad():
    albedo_raw.data += 0.1 * torch.randn_like(albedo_raw)
    albedo_raw.data /= LOW_RANK_K
    albedo_raw.clamp_(min=1e-12)
    ref_color.data += 0.1 * torch.randn_like(ref_color)
    ref_color.clamp_(min=1e-12)
    F.normalize(ref_color, dim=1, eps=1e-12, out=ref_color)
  num_epochs = args.max_pixel_iters

  cache_path = os.path.abspath(args.cache_path) if args.cache_path else os.path.join(cache_dir, "cache.pt")
  data = build_event_pairs_fast(events, pattern_params, rows, cols, cache_path, num_workers=8)

  opt = torch.optim.AdamW(
    [
      {"params": [normal_raw], "lr": 1e-2},
      {"params": [albedo_raw], "lr": 3e-2},
      {"params": [ref_color], "lr": 3e-2},
    ],
    betas=(0.5, 0.95)
  )
  scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[800, 1200, 1400], gamma=0.1)

  pbar = tqdm(range(num_epochs), desc="Training Epochs", disable=args.quiet)
  for epoch in pbar:
    x = data["x"].to(DEVICE).long()
    y = data["y"].to(DEVICE).long()
    Lh = data["light_dir_high"].to(DEVICE)
    Ll = data["light_dir_low"].to(DEVICE)
    Sh = data["light_spec_high"].to(DEVICE)
    Sl = data["light_spec_low"].to(DEVICE)
    C = data["C"].to(DEVICE)
    normal_raw_normalized = F.normalize(normal_raw, dim=-1, eps=1e-12)
    N = normal_raw_normalized[x, y]
    albedo = albedo_raw[x, y]
    spec_response_h = torch.matmul(Sh, ref_color.t())
    spec_response_l = torch.matmul(Sl, ref_color.t())
    I_high = torch.relu((Lh * N).sum(dim=1)) * (spec_response_h * albedo).sum(dim=1) + args.constant_light
    I_low = torch.relu((Ll * N).sum(dim=1)) * (spec_response_l * albedo).sum(dim=1) + args.constant_light
    @torch.no_grad()
    def calc_weight(L, N):
      weight = torch.cat([L[..., :-1], L[..., -1:] + 1.0], dim=-1)
      weight = F.normalize(weight, dim=-1, eps=1e-12)
      weight = F.relu(1. - F.relu((weight * N).sum(dim=-1)).pow(8))
      return weight
    weight = torch.minimum(calc_weight(Lh, N), calc_weight(Ll, N))
    res = weight * (I_high - C * I_low) / (I_low + 1e-3)
    loss_event = F.mse_loss(res, torch.zeros_like(res))


    normal_blur = torchvision.transforms.functional.gaussian_blur(
      normal_raw_normalized.permute(2, 0, 1),
      args.normal_blur_kernel,
      args.normal_blur_sigma,
    ).permute(1, 2, 0)
    albedo_blur = torchvision.transforms.functional.gaussian_blur(
      albedo_raw.permute(2, 0, 1),
      args.albedo_blur_kernel,
      args.albedo_blur_sigma,
    ).permute(1, 2, 0)
    loss_smooth = \
      args.normal_smooth_weight * (normal_raw_normalized - normal_blur).pow(2).sum() + \
      args.albedo_smooth_weight * (albedo_raw - albedo_blur).pow(2).sum()
    loss = loss_event + loss_smooth
    # loss = loss_event
    opt.zero_grad()
    loss.backward()
    opt.step()
    #scheduler.step()

    with torch.no_grad():
      normal_raw[..., 2].clamp_(min=1e-12)
      F.normalize(normal_raw, dim=-1, eps=1e-12, out=normal_raw)
      albedo_raw.clamp_(min=1e-12)
      ref_color.clamp_(min=1e-12)
      F.normalize(ref_color, dim=1, eps=1e-12, out=ref_color)
      normal_pred = normal_raw.detach().cpu()
      ref_pred = (albedo_raw[..., None] * ref_color[None, None, ...]).sum(dim=2).detach().cpu()
      # normal_err = (normal_pred * normal_gt).sum(dim=2)
      # normal_err = torch.rad2deg(torch.acos(normal_err.clamp(min=-1.0, max=1.0)))
      # ref_err = (F.normalize(ref_pred, dim=2) * F.normalize(ref_gt, dim=2)).sum(dim=2)
      # ref_err = torch.rad2deg(torch.acos(ref_err.clamp(min=-1.0, max=1.0)))


  # visualize_error_maps(normal_err.numpy(), ref_err.numpy(), data_path)
  with torch.no_grad():
    normal_gt_cpu = normal_gt.detach().cpu()
    ref_gt_cpu = ref_gt.detach().cpu()
    valid_mask = normal_gt_cpu[..., 2] > 1e-6

    normal_pred_n = F.normalize(normal_pred, dim=2, eps=1e-12)
    normal_gt_n = F.normalize(normal_gt_cpu, dim=2, eps=1e-12)
    cos_sim = (normal_pred_n * normal_gt_n).sum(dim=2).clamp(min=-1.0, max=1.0)
    normal_err = torch.rad2deg(torch.acos(cos_sim))
    normal_err_mean = normal_err[valid_mask].mean() if valid_mask.any() else torch.tensor(float("nan"))

    ref_err = (ref_pred - ref_gt_cpu).pow(2).mean(dim=2).sqrt()
    ref_err_mean = ref_err[valid_mask].mean() if valid_mask.any() else torch.tensor(float("nan"))

  print(f"Normal MAE (deg, valid): {float(normal_err_mean):.6f}")
  print(f"Ref RMSE (valid): {float(ref_err_mean):.6f}")
  if not args.skip_visualizations:
    visualize_array(normal_pred, normal_gt.cpu().numpy(), "normal", vis_dir)
    visualize_array(ref_pred / 5.0, ref_gt.cpu().numpy(), "ref", vis_dir)
  np.save(os.path.join(output_dir, "final_ref.npy"), ref_pred)
  np.save(os.path.join(output_dir, "final_normal.npy"), normal_pred)
  manifest = {
    "schema_version": 1,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "method": "eventspecps_eccv",
    "module": "EventSpecPS.python.map_solver.event_map_solver",
    "source_acquisition": os.path.abspath(data_path),
    "max_pixel_iters": args.max_pixel_iters,
    "normal_smooth_weight": args.normal_smooth_weight,
    "albedo_smooth_weight": args.albedo_smooth_weight,
    "constant_light": args.constant_light,
    "normal_blur_kernel": args.normal_blur_kernel,
    "normal_blur_sigma": args.normal_blur_sigma,
    "albedo_blur_kernel": args.albedo_blur_kernel,
    "albedo_blur_sigma": args.albedo_blur_sigma,
    "low_rank_k": LOW_RANK_K,
    "random_seed": args.random_seed,
    "device": str(DEVICE),
    "cache_path": os.path.abspath(cache_path),
    "cache_reused": args.cache_path is not None,
    "visualizations_skipped": args.skip_visualizations,
    "git_commit": current_git_commit(),
  }
  with open(os.path.join(output_dir, "run_manifest.json"), "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2)

if __name__ == "__main__":
  main()
