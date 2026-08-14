#!/usr/bin/env python3
# coding: utf-8
"""
LiDAR projection visualizer for RGB or depth.
=============================================
Randomly samples --max_pairs from matched image-PCD pairs and projects LiDAR
points with a calibrated 4x4 LiDAR->camera extrinsic matrix.

Supported modes:
  - rgb: project onto RGB images using color_camera_intrinsics.txt
  - depth: project onto depth images using depth_camera_intrinsics.txt
  - auto: infer from the extrinsics JSON if possible, otherwise RGB

The default matrix source is:
    /home/xuyh/mapanything/output/lidar_calibration/calibration_summary.json

Usage:
    python visualize_projection.py \
        --seq_root /add02/users/xuyh/seq1 \
        --output_dir /home/xuyh/mapanything/output/ \
        --max_pairs 500 \
        --img_size 448
"""

import os
import re
import glob
import cv2
import argparse
import warnings
import random
from typing import Tuple, List, Dict

import numpy as np
from PIL import Image
import open3d as o3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

DEFAULT_LIDAR_EXTRINSICS_PATH = "/home/xuyh/mapanything/output/lidar_calibration/calibration_summary.json"


def _parse_camera_intrinsics_file(file_path: str) -> Tuple[np.ndarray, Tuple[int, int]]:
    K = None
    width = height = None
    if not file_path or not os.path.exists(file_path):
        return np.eye(3, dtype=np.float32), (0, 0)
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        wm = re.search(r"width:\s*(\d+)", content)
        hm = re.search(r"height:\s*(\d+)", content)
        if wm:
            width = int(wm.group(1))
        if hm:
            height = int(hm.group(1))
        km = re.search(r"K:\s*\[\[(.*?)\]\]", content, re.DOTALL)
        if km:
            nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", km.group(1))
            if len(nums) >= 9:
                K = np.array([float(x) for x in nums[:9]], dtype=np.float32).reshape(3, 3)
    except Exception:
        pass
    if K is None:
        try:
            data = np.loadtxt(file_path)
            if data.shape == (3, 3):
                K = data.astype(np.float32)
            elif data.size == 9:
                K = data.reshape(3, 3).astype(np.float32)
        except Exception:
            K = np.eye(3, dtype=np.float32)
    return K.astype(np.float32), (int(width or 0), int(height or 0))


def _load_lidar_extrinsics_matrix(extrinsics_path: str = "") -> Tuple[np.ndarray, str]:
    default_T = np.eye(4, dtype=np.float32)
    candidates = []
    if extrinsics_path:
        candidates.append(extrinsics_path)

    for path in candidates:
        if not path or not os.path.exists(path):
            continue
        try:
            ext = os.path.splitext(path)[1].lower()
            if ext == ".npy":
                T = np.load(path).astype(np.float32)
            elif ext in (".txt", ".csv", ".dat"):
                T = np.loadtxt(path).astype(np.float32)
            elif ext == ".json":
                import json

                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    T = np.array(data, dtype=np.float32)
                elif isinstance(data, dict):
                    if "rgb" in data and isinstance(data["rgb"], dict) and "T_lidar_to_rgb" in data["rgb"]:
                        T = np.array(data["rgb"]["T_lidar_to_rgb"], dtype=np.float32)
                    elif "depth" in data and isinstance(data["depth"], dict) and "T_lidar_to_depth" in data["depth"]:
                        T = np.array(data["depth"]["T_lidar_to_depth"], dtype=np.float32)
                    else:
                        for key in ("T_lidar_to_cam", "T_lidar_to_rgb", "T_lidar_to_depth", "T"):
                            if key in data:
                                T = np.array(data[key], dtype=np.float32)
                                break
                        else:
                            T = np.array([], dtype=np.float32)
                else:
                    T = np.array([], dtype=np.float32)
            else:
                T = np.array([], dtype=np.float32)
            if T.shape == (4, 4):
                return T, os.path.abspath(path)
        except Exception:
            continue

    return default_T, ""


def _scale_intrinsics(K: np.ndarray, src_size: Tuple[int, int], dst_size: Tuple[int, int]) -> np.ndarray:
    src_w, src_h = src_size
    dst_w, dst_h = dst_size
    K_scaled = K.astype(np.float32).copy()
    if src_w > 0 and src_h > 0:
        sx = float(dst_w) / float(src_w)
        sy = float(dst_h) / float(src_h)
        K_scaled[0, 0] *= sx
        K_scaled[0, 2] *= sx
        K_scaled[1, 1] *= sy
        K_scaled[1, 2] *= sy
    return K_scaled


def _resolve_depth_scale(depth: np.ndarray, requested_scale: float, raw_dtype=None) -> float:
    if requested_scale and requested_scale > 0:
        return float(requested_scale)
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        return 1.0
    max_v = float(np.max(depth[valid]))
    median_v = float(np.median(depth[valid]))
    if (raw_dtype is not None and np.issubdtype(raw_dtype, np.integer)) or max_v > 1000.0 or median_v > 50.0:
        return 1000.0
    return 1.0


def _depth_to_display_rgb(depth: np.ndarray, depth_scale: float, target_size: Tuple[int, int]) -> np.ndarray:
    raw_dtype = np.asarray(depth).dtype
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    scale = _resolve_depth_scale(depth, depth_scale, raw_dtype=raw_dtype)
    depth_m = depth / scale
    finite = np.isfinite(depth_m) & (depth_m > 0)
    if np.any(finite):
        vis_scale = float(np.percentile(depth_m[finite], 95.0))
        if vis_scale <= 1e-8:
            vis_scale = float(np.max(depth_m[finite]) + 1e-8)
    else:
        vis_scale = 1.0
    depth_norm = np.clip(depth_m / (vis_scale + 1e-8), 0.0, 1.0)
    # turbo uses low values as blue and high values as red, matching the
    # near-blue / far-red convention used by the depth images in this project.
    depth_rgb = (plt.cm.turbo(depth_norm)[..., :3] * 255.0).astype(np.uint8)
    return depth_rgb


def _transform_lidar_points_to_camera(points_xyz: np.ndarray, T_lidar_to_cam: np.ndarray) -> np.ndarray:
    points_xyz = np.asarray(points_xyz, dtype=np.float32)
    if points_xyz.size == 0:
        return points_xyz.reshape(0, 3)
    if T_lidar_to_cam is None:
        return points_xyz.astype(np.float32)
    T = np.asarray(T_lidar_to_cam, dtype=np.float32)
    if T.shape != (4, 4):
        return points_xyz.astype(np.float32)
    pts_h = np.concatenate([points_xyz, np.ones((len(points_xyz), 1), dtype=np.float32)], axis=1)
    pts_cam = (T @ pts_h.T).T[:, :3]
    return pts_cam.astype(np.float32)


def _zero_translation(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float32)
    if T.shape != (4, 4):
        return T
    out = T.copy()
    out[:3, 3] = 0.0
    return out


def project_pcd_to_image_calibrated(
    pcd_path: str,
    K: np.ndarray,
    src_size: Tuple[int, int],
    target_size: Tuple[int, int] = (448, 448),
    lidar_to_cam_T: np.ndarray = None,
) -> Dict:
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points).astype(np.float32)
    result = {
        "pcd_path": pcd_path,
        "num_points": len(points),
        "projected_count": 0,
        "valid_count": 0,
        "u_coords": np.array([]),
        "v_coords": np.array([]),
        "z_depths": np.array([]),
        "reason": "ok",
    }
    if len(points) == 0:
        result["reason"] = "empty_point_cloud"
        return result

    H, W = target_size
    K_proj = _scale_intrinsics(K, src_size, (W, H))
    fx, fy = float(K_proj[0, 0]), float(K_proj[1, 1])
    cx, cy = float(K_proj[0, 2]), float(K_proj[1, 2])

    pts_cam = _transform_lidar_points_to_camera(points, lidar_to_cam_T)
    x, y, z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]

    valid_mask = z > 1e-4
    num_valid = int(np.count_nonzero(valid_mask))
    if num_valid == 0:
        result["reason"] = "no_valid_depth"
        return result

    x, y, z = x[valid_mask], y[valid_mask], z[valid_mask]
    u = (fx * x / z) + cx
    v = (fy * y / z) + cy
    in_image = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    num_projected = int(np.count_nonzero(in_image))
    if num_projected == 0:
        result["reason"] = "projection_out_of_frame"
        return result

    u = u[in_image]
    v = v[in_image]
    z_depth = z[in_image]

    result["projected_count"] = num_projected
    result["valid_count"] = num_projected
    result["u_coords"] = u.astype(np.float32)
    result["v_coords"] = v.astype(np.float32)
    result["z_depths"] = z_depth.astype(np.float32)
    return result


# ========================= Visualization =========================

def draw_projection_overlay(
    image_path: str,
    proj_result: Dict,
    out_path: str,
    target_size: Tuple[int, int] = (448, 448),
    title_prefix: str = "calibrated projection",
    mode: str = "rgb",
    depth_scale: float = 0.0,
):
    if mode == "depth":
        depth_raw = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            img_np = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        else:
            depth = np.asarray(depth_raw)
            if depth.ndim == 3 and depth.shape[2] >= 3:
                # Keep colorized depth images in their native blue-near / red-far coloring.
                img_np = cv2.cvtColor(depth[..., :3], cv2.COLOR_BGR2RGB)
            else:
                if depth.ndim == 3:
                    depth = depth[..., 0]
                img_np = _depth_to_display_rgb(depth, depth_scale=depth_scale, target_size=target_size)
    else:
        img = Image.open(image_path).convert("RGB")
        img_np = np.array(img)

    orig_h, orig_w = img_np.shape[:2]
    if (orig_w, orig_h) != target_size:
        img = Image.fromarray(img_np).resize(target_size, Image.BILINEAR)
        img_np = np.array(img)

    fig, ax = plt.subplots(1, 1, figsize=(7, 7), dpi=150)
    ax.imshow(img_np)

    u = proj_result.get("u_coords", np.array([]))
    v = proj_result.get("v_coords", np.array([]))
    z = proj_result.get("z_depths", np.array([]))

    if u.size > 0:
        z_max_vis = float(np.percentile(z, 95)) if len(z) > 10 else float(z.max())
        z_max_vis = max(z_max_vis, 1e-3)
        scatter = ax.scatter(
            u, v, c=z, cmap="turbo", s=3, alpha=0.8,
            vmin=0.0, vmax=z_max_vis,
        )
        cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Depth (m)", rotation=270, labelpad=18, fontsize=9)

    reason = proj_result.get("reason", "ok")
    count = proj_result.get("projected_count", 0)
    total = proj_result.get("num_points", 0)
    z_min = float(z.min()) if u.size > 0 else 0.0
    z_max = float(z.max()) if u.size > 0 else 0.0
    title = (
        f"{title_prefix} | {os.path.basename(image_path)}\n"
        f"proj={count}/{total} | {reason}\n"
        f"depth range: [{z_min:.2f}, {z_max:.2f}] m"
    )
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def draw_depth_heatmap(
    proj_result: Dict,
    out_path: str,
    target_size: Tuple[int, int] = (448, 448),
):
    H, W = target_size
    depth_map = np.full((H, W), np.nan, dtype=np.float32)
    u = proj_result.get("u_coords", np.array([])).astype(int)
    v = proj_result.get("v_coords", np.array([])).astype(int)
    z = proj_result.get("z_depths", np.array([]))

    for i in range(len(u)):
        ui, vi, zi = u[i], v[i], z[i]
        if 0 <= ui < W and 0 <= vi < H:
            if np.isnan(depth_map[vi, ui]) or zi < depth_map[vi, ui]:
                depth_map[vi, ui] = zi

    fig, ax = plt.subplots(1, 1, figsize=(7, 7), dpi=150)
    masked = np.ma.masked_where(np.isnan(depth_map), depth_map)
    z_max_vis = float(np.percentile(z, 95)) if len(z) > 10 else float(z.max())
    z_max_vis = max(z_max_vis, 1e-3)
    im = ax.imshow(masked, cmap="turbo", vmin=0.0, vmax=z_max_vis, interpolation="nearest")
    ax.set_title(f"calibrated depth | proj={proj_result.get('projected_count', 0)}", fontsize=9)
    ax.axis("off")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Depth (m)", rotation=270, labelpad=18, fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# ========================= Dataset pairing =========================

def collect_rgb_pcd_pairs(seq_root: str, tolerance_sec: float = 0.05):
    img_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp")
    part_folders = []
    for item in sorted(os.listdir(seq_root)):
        full_path = os.path.join(seq_root, item)
        if os.path.isdir(full_path) and re.match(r"shuangchuang_seq\d+_(night|daytime)\d+th", item):
            part_folders.append(item)

    root_intrinsics_file = os.path.join(seq_root, "color_camera_intrinsics.txt")
    root_K, root_size = _parse_camera_intrinsics_file(root_intrinsics_file)

    pairs = []
    for part in part_folders:
        part_path = os.path.join(seq_root, part)
        rgb_dir = os.path.join(part_path, "rgb")
        lidar_dir = os.path.join(part_path, "lidar")
        if not os.path.isdir(rgb_dir) or not os.path.isdir(lidar_dir):
            continue

        intrinsics_file = os.path.join(part_path, "color_camera_intrinsics.txt")
        if os.path.exists(intrinsics_file):
            K, src_size = _parse_camera_intrinsics_file(intrinsics_file)
        else:
            K, src_size = root_K, root_size

        rgb_files = []
        for fname in sorted(os.listdir(rgb_dir)):
            if fname.lower().endswith(img_exts):
                ipath = os.path.join(rgb_dir, fname)
                m = re.search(r"color_(\d+)", fname)
                if not m:
                    m = re.search(r"(\d+)", fname)
                if m:
                    ts_ns = int(m.group(1))
                    rgb_files.append((ts_ns, ipath))

        pcd_files = []
        for pf in sorted(glob.glob(os.path.join(lidar_dir, "*.pcd"))):
            m = re.search(r"(\d+)", os.path.basename(pf))
            ts_ns = int(m.group(1)) if m else int(os.path.getmtime(pf) * 1e9)
            pcd_files.append((ts_ns, pf))
        if not pcd_files:
            continue
        pcd_ts_arr = np.array([t for t, _ in pcd_files], dtype=np.int64)
        pcd_path_arr = np.array([p for _, p in pcd_files])

        tolerance_ns = int(tolerance_sec * 1e9)
        for img_ts_ns, ipath in rgb_files:
            idx = np.argmin(np.abs(pcd_ts_arr - img_ts_ns))
            matched_ts = int(pcd_ts_arr[idx])
            diff_ns = abs(matched_ts - img_ts_ns)
            if diff_ns <= tolerance_ns:
                pairs.append({
                    "image_path": ipath,
                    "pcd_path": pcd_path_arr[idx],
                    "K": K,
                    "src_size": src_size,
                    "img_ts_ns": img_ts_ns,
                    "diff_ns": diff_ns,
                    "part": part,
                })
    return pairs


def collect_depth_pcd_pairs(seq_root: str, tolerance_sec: float = 0.05, depth_intrinsics_path: str = ""):
    img_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp", ".npy")
    part_folders = []
    for item in sorted(os.listdir(seq_root)):
        full_path = os.path.join(seq_root, item)
        if os.path.isdir(full_path) and re.match(r"shuangchuang_seq\d+_(night|daytime)\d+th", item):
            part_folders.append(item)

    root_intrinsics_file = depth_intrinsics_path or os.path.join(seq_root, "depth_camera_intrinsics.txt")
    root_K, root_size = _parse_camera_intrinsics_file(root_intrinsics_file)

    pairs = []
    for part in part_folders:
        part_path = os.path.join(seq_root, part)
        depth_dir = os.path.join(part_path, "depth")
        lidar_dir = os.path.join(part_path, "lidar")
        if not os.path.isdir(depth_dir) or not os.path.isdir(lidar_dir):
            continue

        intrinsics_file = depth_intrinsics_path or os.path.join(part_path, "depth_camera_intrinsics.txt")
        if os.path.exists(intrinsics_file):
            K, src_size = _parse_camera_intrinsics_file(intrinsics_file)
        else:
            K, src_size = root_K, root_size

        depth_files = []
        for fname in sorted(os.listdir(depth_dir)):
            if fname.lower().endswith(img_exts):
                dpath = os.path.join(depth_dir, fname)
                m = re.search(r"(\d+)", fname)
                if m:
                    ts_ns = int(m.group(1))
                    depth_files.append((ts_ns, dpath))

        pcd_files = []
        for pf in sorted(glob.glob(os.path.join(lidar_dir, "*.pcd"))):
            m = re.search(r"(\d+)", os.path.basename(pf))
            ts_ns = int(m.group(1)) if m else int(os.path.getmtime(pf) * 1e9)
            pcd_files.append((ts_ns, pf))
        if not pcd_files:
            continue
        pcd_ts_arr = np.array([t for t, _ in pcd_files], dtype=np.int64)
        pcd_path_arr = np.array([p for _, p in pcd_files])

        tolerance_ns = int(tolerance_sec * 1e9)
        for img_ts_ns, dpath in depth_files:
            idx = np.argmin(np.abs(pcd_ts_arr - img_ts_ns))
            matched_ts = int(pcd_ts_arr[idx])
            diff_ns = abs(matched_ts - img_ts_ns)
            if diff_ns <= tolerance_ns:
                pairs.append({
                    "image_path": dpath,
                    "pcd_path": pcd_path_arr[idx],
                    "K": K,
                    "src_size": src_size,
                    "img_ts_ns": img_ts_ns,
                    "diff_ns": diff_ns,
                    "part": part,
                })
    return pairs


# ========================= Main =========================

def main():
    parser = argparse.ArgumentParser(description="LiDAR calibrated projection visualizer (random)")
    parser.add_argument("--seq_root", type=str, default="/add02/users/xuyh/seq1/")
    parser.add_argument("--output_dir", type=str, default="/home/xuyh/mapanything/output_proj/")
    parser.add_argument("--max_pairs", type=int, default=100)
    parser.add_argument("--img_size", type=int, default=448)
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument(
        "--mode",
        type=str,
        default="depth",
        choices=("auto", "rgb", "depth"),
        help="Projection target modality.",
    )
    parser.add_argument(
        "--lidar_extrinsics_path",
        type=str,
        default="/home/xuyh/mapanything/output/lidar_calibration/calibration_summary.json",
        help="4x4 LiDAR->camera extrinsics file (.npy/.txt/.json), defaulting to calibration_summary.json.",
    )
    parser.add_argument("--rgb_intrinsics_path", type=str, default="", help="Optional override for color intrinsics.")
    parser.add_argument("--depth_intrinsics_path", type=str, default="", help="Optional override for depth intrinsics.")
    parser.add_argument("--ignore_translation", action="store_true", default=False, help="Zero the extrinsics translation before projecting.")
    parser.add_argument(
        "--depth_scale",
        type=float,
        default=1000.0,
        help="Depth-to-meter scale factor. Use 1000 for mm->m; 0 means auto-detect.",
    )
    parser.add_argument("--save_overlay", action="store_true", default=True)
    parser.add_argument("--save_depth", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    target_size = (args.img_size, args.img_size)
    lidar_to_cam_T, extrinsics_source = _load_lidar_extrinsics_matrix(args.lidar_extrinsics_path)
    if args.ignore_translation:
        lidar_to_cam_T = _zero_translation(lidar_to_cam_T)
        print("[Input] extrinsics translation zeroed before projection")
    mode = args.mode
    if mode == "auto":
        if extrinsics_source.lower().endswith(".json"):
            try:
                import json

                with open(extrinsics_source, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "depth" in data and isinstance(data["depth"], dict) and "T_lidar_to_depth" in data["depth"]:
                    mode = "depth"
                else:
                    mode = "rgb"
            except Exception:
                mode = "rgb"
        else:
            mode = "rgb"
    if extrinsics_source:
        print(f"[Input] lidar_extrinsics_path={extrinsics_source}")
    else:
        print("[Input] lidar_extrinsics_path=<compatibility-fallback>")
    print(f"[Input] mode={mode}")

    print(f"[1/3] Scanning {args.seq_root} ...")
    if mode == "depth":
        pairs = collect_depth_pcd_pairs(
            args.seq_root,
            tolerance_sec=args.tolerance,
            depth_intrinsics_path=args.depth_intrinsics_path,
        )
    else:
        pairs = collect_rgb_pcd_pairs(
            args.seq_root,
            tolerance_sec=args.tolerance,
            rgb_intrinsics_path=args.rgb_intrinsics_path,
        )
    print(f"       Found {len(pairs)} matched pairs")
    if len(pairs) == 0:
        print("       No pairs found. Exiting.")
        return

    rng = random.Random(args.seed)
    n = min(args.max_pairs, len(pairs))
    pairs = rng.sample(pairs, n)
    print(f"[2/3] Randomly sampled {len(pairs)} pairs (seed={args.seed})")

    ok_count = 0
    fail_reasons = {}
    print(f"[3/3] Projecting with calibrated LiDAR extrinsics ...")

    for idx, pair in enumerate(pairs, start=1):
        if mode == "depth":
            proj = project_pcd_to_image_calibrated(
                pair["pcd_path"],
                pair["K"],
                pair["src_size"],
                target_size=target_size,
                lidar_to_cam_T=lidar_to_cam_T,
            )
        else:
            proj = project_pcd_to_image_calibrated(
                pair["pcd_path"],
                pair["K"],
                pair["src_size"],
                target_size=target_size,
                lidar_to_cam_T=lidar_to_cam_T,
            )
        reason = proj.get("reason", "ok")
        if reason != "ok":
            fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
        else:
            ok_count += 1

        base_name = f"{idx:04d}_{pair['part']}_{pair['img_ts_ns']}"
        if args.save_overlay:
            draw_projection_overlay(
                pair["image_path"],
                proj,
                os.path.join(args.output_dir, f"{base_name}_overlay.png"),
                target_size=target_size,
                title_prefix=f"{mode.upper()} calibrated projection",
                mode=mode,
                depth_scale=args.depth_scale,
            )
        if args.save_depth:
            draw_depth_heatmap(
                proj,
                os.path.join(args.output_dir, f"{base_name}_depth.png"),
                target_size=target_size,
            )
        if idx % 50 == 0 or idx == len(pairs):
            print(f"       Processed {idx}/{len(pairs)} ...")

    print(f"Done. Results saved to {args.output_dir}")
    print(f"       Successful projections: {ok_count}/{len(pairs)}")
    if fail_reasons:
        print(f"       Failures: {fail_reasons}")


if __name__ == "__main__":
    main()
