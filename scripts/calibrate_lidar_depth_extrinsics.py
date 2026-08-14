#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estimate LiDAR-to-depth-camera extrinsics from depth and LiDAR only.

The script follows the same sequence layout as the training pipeline:

* ``seq_root`` contains one or more ``shuangchuang_seq...th`` part folders.
* Each part may contain ``depth/`` and ``lidar/`` subfolders.
* Depth intrinsics are resolved from ``depth_camera_intrinsics.txt`` in the
  part folder, its parent, or the sequence root.

It estimates the LiDAR -> depth camera transform by:

1. Matching depth frames and LiDAR point clouds by timestamp.
2. Backprojecting depth images into 3D points using the depth intrinsics.
3. Refining a continuous rigid transform, optionally searching axis conventions
   only when the LiDAR frame is not already aligned.
4. Selecting the transform that best aligns LiDAR projections with the depth
   image structure.

The main output is the translation vector and the full 4x4
``T_lidar_to_depth`` matrix.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as SciRotation


def find_repo_root() -> Path:
    script_path = Path(__file__).resolve()
    for candidate in [script_path.parent, *script_path.parent.parents]:
        if (candidate / "mapanything").is_dir() and (candidate / "scripts").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate repository root from {script_path}")


ROOT = find_repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".tif")
PCD_EXTS = (".pcd",)
PART_NAME_RE = __import__("re").compile(r"^shuangchuang_seq\d+_(?:night|daytime)\d+th$")


def timestamp_ns_from_path(path: str) -> int:
    stem = Path(path).stem
    digits = __import__("re").findall(r"\d+", stem)
    if not digits:
        raise ValueError(f"Cannot parse timestamp from path: {path}")
    return int(max(digits, key=len))


def list_modal_files(modal_dir: str, exts: Sequence[str]) -> List[str]:
    if not os.path.isdir(modal_dir):
        return []
    return [
        os.path.join(modal_dir, name)
        for name in sorted(os.listdir(modal_dir))
        if name.lower().endswith(tuple(exts))
    ]


def parse_camera_intrinsics_file(file_path: str) -> Tuple[np.ndarray, Tuple[int, int]]:
    K = None
    width = None
    height = None
    if not file_path or not os.path.exists(file_path):
        return np.eye(3, dtype=np.float32), (0, 0)

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        width_match = __import__("re").search(r"width:\s*(\d+)", content)
        height_match = __import__("re").search(r"height:\s*(\d+)", content)
        if width_match:
            width = int(width_match.group(1))
        if height_match:
            height = int(height_match.group(1))

        k_match = __import__("re").search(r"K:\s*\[\[(.*?)\]\]", content, __import__("re").DOTALL)
        if k_match:
            nums = __import__("re").findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", k_match.group(1))
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


def scale_intrinsics(K: np.ndarray, src_size: Tuple[int, int], dst_size: Tuple[int, int]) -> np.ndarray:
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


def _infer_depth_scale_from_data(depth: np.ndarray) -> float:
    depth = np.asarray(depth)
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        return 1.0
    if not np.issubdtype(depth.dtype, np.integer):
        return 1.0

    max_v = float(np.max(depth[valid]))
    median_v = float(np.median(depth[valid]))

    # Common encodings:
    # - meters in float: scale 1
    # - centimeters in integer PNG: scale 100
    # - millimeters in integer PNG: scale 1000
    # Use broad thresholds so we do not collapse a 5-6m scene to 0.2m.
    if max_v >= 5000.0 or median_v >= 500.0:
        return 1000.0
    if max_v >= 500.0 or median_v >= 50.0:
        return 100.0
    if max_v >= 50.0 or median_v >= 5.0:
        return 10.0
    return 1.0


def _decode_colorized_depth(depth_bgr: np.ndarray) -> Tuple[np.ndarray, str]:
    depth_bgr = np.asarray(depth_bgr)
    if depth_bgr.ndim != 3 or depth_bgr.shape[2] < 3:
        raise ValueError("Expected a 3-channel colorized depth image.")

    # OpenCV loads BGR, matplotlib colormaps are RGB.
    rgb = cv2.cvtColor(depth_bgr, cv2.COLOR_BGR2RGB).reshape(-1, 3).astype(np.float32)
    if rgb.size == 0:
        return np.zeros(depth_bgr.shape[:2], dtype=np.float32), "empty"

    candidates = ("turbo", "turbo_r", "jet", "jet_r", "viridis", "viridis_r", "plasma", "plasma_r")
    best_name = candidates[0]
    best_norm = None
    best_err = float("inf")

    for cmap_name in candidates:
        cmap = plt.get_cmap(cmap_name)
        lut = (cmap(np.linspace(0.0, 1.0, 256))[:, :3] * 255.0).astype(np.float32)
        tree = cKDTree(lut)
        dist, idx = tree.query(rgb, k=1)
        err = float(np.mean(dist))
        if err < best_err:
            best_err = err
            best_name = cmap_name
            best_norm = idx.astype(np.float32) / 255.0

    if best_norm is None:
        best_norm = np.zeros((rgb.shape[0],), dtype=np.float32)

    return best_norm.reshape(depth_bgr.shape[:2]).astype(np.float32), best_name


def load_depth_image(path: str, depth_scale: float = 0.0) -> np.ndarray:
    depth_raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise ValueError(f"Failed to read depth image: {path}")
    depth = np.asarray(depth_raw)
    if depth.ndim == 3:
        if depth.shape[2] >= 3:
            channel_delta = np.max(np.abs(depth[..., 0].astype(np.int32) - depth[..., 1].astype(np.int32)))
            channel_delta = max(channel_delta, np.max(np.abs(depth[..., 1].astype(np.int32) - depth[..., 2].astype(np.int32))))
            if channel_delta > 0:
                norm_depth, cmap_name = _decode_colorized_depth(depth)
                scale = float(depth_scale) if depth_scale and depth_scale > 0 else 40.0
                warnings.warn(
                    f"Depth image {path} appears colorized; decoded with {cmap_name} and scaled to {scale:.1f} m."
                )
                return (norm_depth * scale).astype(np.float32)
            depth = depth[..., 0]
        else:
            depth = depth[..., 0]
    depth = depth.astype(np.float32)
    scale = float(depth_scale) if depth_scale and depth_scale > 0 else _infer_depth_scale_from_data(depth)
    if scale > 0:
        depth = depth / scale
    return depth


def load_pcd_points(path: str, max_points: int, rng: np.random.Generator) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(path)
    points = np.asarray(pcd.points, dtype=np.float32)
    if points.size == 0:
        return points.reshape(0, 3)
    if max_points > 0 and len(points) > max_points:
        idx = rng.choice(len(points), size=max_points, replace=False)
        points = points[idx]
    return points.astype(np.float32)


def backproject_depth_to_points(depth: np.ndarray, K: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32)
    ys, xs = np.where(valid)
    if max_points > 0 and len(xs) > max_points:
        idx = rng.choice(len(xs), size=max_points, replace=False)
        xs = xs[idx]
        ys = ys[idx]
    z = depth[ys, xs].astype(np.float32)
    x = (xs.astype(np.float32) - float(K[0, 2])) * z / float(K[0, 0])
    y = (ys.astype(np.float32) - float(K[1, 2])) * z / float(K[1, 1])
    return np.stack([x, y, z], axis=1).astype(np.float32)


def compute_edge_map_depth(depth: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        return np.zeros_like(depth, dtype=np.float32)
    depth_f = depth.copy().astype(np.float32)
    depth_f[~valid] = 0.0
    if np.count_nonzero(valid) > 0:
        scale = float(np.percentile(depth_f[valid], 95.0))
        if scale <= 1e-8:
            scale = float(np.max(depth_f[valid]) + 1e-8)
    else:
        scale = 1.0
    depth_n = np.clip(depth_f / (scale + 1e-8), 0.0, 1.0)
    gx = cv2.Sobel(depth_n, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth_n, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    denom = float(np.percentile(mag[valid], 99.0)) if np.any(valid) else float(mag.max() + 1e-8)
    if denom <= 1e-8:
        denom = float(mag.max() + 1e-8)
    return np.clip(mag / (denom + 1e-8), 0.0, 1.0).astype(np.float32)


def signed_permutation_candidates(search_axis_mapping: bool) -> List[str]:
    if not search_axis_mapping:
        return ["x_y_z"]
    results: List[str] = []
    order = {"x": 0, "y": 1, "z": 2}
    for perm in __import__("itertools").permutations(("x", "y", "z"), 3):
        idxs = [order[a] for a in perm]
        inv_count = sum(1 for i in range(3) for j in range(i + 1, 3) if idxs[i] > idxs[j])
        perm_sign = -1 if inv_count % 2 else 1
        for sign_bits in __import__("itertools").product((1, -1), repeat=3):
            if perm_sign * (sign_bits[0] * sign_bits[1] * sign_bits[2]) != 1:
                continue
            name = "_".join(f"{'-' if sign < 0 else ''}{axis}" for axis, sign in zip(perm, sign_bits))
            results.append(name)
    results.sort()
    return results


def axis_matrix_from_name(mapping_name: str) -> np.ndarray:
    components = mapping_name.split("_")
    axis_vectors = {
        "x": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "y": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "z": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        "-x": np.array([-1.0, 0.0, 0.0], dtype=np.float32),
        "-y": np.array([0.0, -1.0, 0.0], dtype=np.float32),
        "-z": np.array([0.0, 0.0, -1.0], dtype=np.float32),
    }
    if len(components) != 3:
        raise ValueError(f"Invalid mapping name: {mapping_name}")
    mat = np.stack([axis_vectors[name] for name in components], axis=1)
    return mat.astype(np.float32)


def apply_axis_mapping(points_xyz: np.ndarray, mapping_name: str) -> np.ndarray:
    components = mapping_name.split("_")
    axis_vectors = {
        "x": points_xyz[:, 0],
        "y": points_xyz[:, 1],
        "z": points_xyz[:, 2],
        "-x": -points_xyz[:, 0],
        "-y": -points_xyz[:, 1],
        "-z": -points_xyz[:, 2],
    }
    mapped = np.stack([axis_vectors[name] for name in components], axis=1)
    return mapped.astype(np.float32)


def transform_points(points_xyz: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (points_xyz @ R.T) + t.reshape(1, 3)


def rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    rotvec = np.asarray(rotvec, dtype=np.float32).reshape(3)
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-8:
        return np.eye(3, dtype=np.float32)
    return SciRotation.from_rotvec(rotvec.astype(np.float64)).as_matrix().astype(np.float32)


def compose_transform(mapping_name: str, rotvec: np.ndarray, translation: np.ndarray) -> np.ndarray:
    R_axis = axis_matrix_from_name(mapping_name)
    R_delta = rotvec_to_matrix(rotvec)
    R = R_delta @ R_axis
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(translation, dtype=np.float32).reshape(3)
    return T


def project_points(
    points_cam: np.ndarray,
    K: np.ndarray,
    image_size: Tuple[int, int],
) -> dict:
    width, height = image_size
    if points_cam.size == 0:
        return {
            "u": np.zeros((0,), dtype=np.float32),
            "v": np.zeros((0,), dtype=np.float32),
            "z": np.zeros((0,), dtype=np.float32),
            "valid": np.zeros((0,), dtype=bool),
            "projected_count": 0,
            "projected_count_raw": 0,
            "positive_depth_count": 0,
        }

    x = points_cam[:, 0]
    y = points_cam[:, 1]
    z = points_cam[:, 2]
    positive = z > 1e-4
    positive_count = int(np.count_nonzero(positive))
    if positive_count == 0:
        return {
            "u": np.zeros((0,), dtype=np.float32),
            "v": np.zeros((0,), dtype=np.float32),
            "z": np.zeros((0,), dtype=np.float32),
            "valid": np.zeros((0,), dtype=bool),
            "projected_count": 0,
            "projected_count_raw": 0,
            "positive_depth_count": 0,
        }

    x = x[positive]
    y = y[positive]
    z = z[positive]
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    u = (fx * x / z) + cx
    v = (fy * y / z) + cy
    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u = u[valid].astype(np.float32)
    v = v[valid].astype(np.float32)
    z = z[valid].astype(np.float32)
    projected_count_raw = int(len(z))
    return {
        "u": u,
        "v": v,
        "z": z,
        "valid": valid.astype(bool),
        "projected_count": int(len(z)),
        "projected_count_raw": projected_count_raw,
        "positive_depth_count": positive_count,
    }


def make_overlay(
    image_rgb: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    z: np.ndarray,
    point_radius: int,
    title: str,
    stats_text: Sequence[str],
) -> np.ndarray:
    canvas = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)
    if len(z) > 0:
        order = np.argsort(z)
        u = u[order]
        v = v[order]
        z = z[order]
        z_min = float(np.percentile(z, 5.0))
        z_max = float(np.percentile(z, 90.0))
        z_min = max(z_min, 1e-6)
        z_max = max(z_max, z_min + 1e-6)
        norm = np.clip((z - z_min) / (z_max - z_min), 0.0, 1.0)
        colors = cv2.applyColorMap((norm * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
        for idx in range(len(u)):
            center = (int(round(float(u[idx]))), int(round(float(v[idx]))))
            color = tuple(int(c) for c in colors[idx, 0].tolist())
            radius = point_radius + (1 if idx < max(1, len(u) // 4) else 0)
            cv2.circle(canvas, center, radius, color, thickness=-1, lineType=cv2.LINE_AA)

    panel_h = min(96, max(72, canvas.shape[0] // 6))
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], panel_h), (0, 0, 0), thickness=-1)
    cv2.putText(canvas, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    y = 54
    for line in stats_text:
        cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        y += 20
    return canvas


def summarize_translation(t: np.ndarray) -> dict:
    return {"x": float(t[0]), "y": float(t[1]), "z": float(t[2]), "norm": float(np.linalg.norm(t))}


def score_rgb_frame(points_cam: np.ndarray, K: np.ndarray, rgb_edge: np.ndarray):
    h, w = rgb_edge.shape[:2]
    proj = project_points(points_cam, K, (w, h))
    if proj["projected_count"] == 0:
        return 0.0, {"coverage": 0.0, "edge": 0.0, "near_coverage": 0.0}, proj
    u = proj["u"].astype(np.int32)
    v = proj["v"].astype(np.int32)
    edge_vals = rgb_edge[v, u]
    coverage = float(proj["projected_count"]) / float(max(len(points_cam), 1))
    near_coverage = float(proj["projected_count"]) / float(max(proj.get("projected_count_raw", proj["projected_count"]), 1))
    edge_score = float(np.mean(edge_vals)) if edge_vals.size > 0 else 0.0
    score = 0.72 * edge_score + 0.18 * near_coverage + 0.10 * coverage
    return score, {"coverage": coverage, "edge": edge_score, "near_coverage": near_coverage}, proj


def score_depth_frame(
    points_cam: np.ndarray,
    K: np.ndarray,
    depth: np.ndarray,
    depth_edge: np.ndarray,
    depth_error_scale: float,
):
    h, w = depth.shape[:2]
    proj = project_points(points_cam, K, (w, h))
    if proj["projected_count"] == 0:
        return 0.0, {"coverage": 0.0, "edge": 0.0, "depth": 0.0, "near_coverage": 0.0}, proj
    u = proj["u"].astype(np.int32)
    v = proj["v"].astype(np.int32)
    z_lidar = proj["z"]
    depth_vals = depth[v, u]
    valid = np.isfinite(depth_vals) & (depth_vals > 0)
    coverage = float(proj["projected_count"]) / float(max(len(points_cam), 1))
    near_coverage = float(proj["projected_count"]) / float(max(proj.get("projected_count_raw", proj["projected_count"]), 1))
    if np.any(valid):
        errors = np.abs(z_lidar[valid] - depth_vals[valid]).astype(np.float32)
        if errors.size > 1:
            keep = max(1, int(errors.size * 0.8))
            errors = np.partition(errors, keep - 1)[:keep]
        depth_err = float(np.mean(errors))
        depth_score = float(math.exp(-depth_err / max(depth_error_scale, 1e-6)))
    else:
        depth_err = float("inf")
        depth_score = 0.0
    edge_score = float(np.mean(depth_edge[v, u])) if u.size > 0 else 0.0
    score = 0.55 * depth_score + 0.25 * edge_score + 0.10 * near_coverage + 0.10 * coverage
    return score, {"coverage": coverage, "edge": edge_score, "depth": depth_score, "depth_err": depth_err, "near_coverage": near_coverage}, proj


def apply_candidate_transform(points_lidar: np.ndarray, mapping_name: str, rotvec: np.ndarray, translation: np.ndarray) -> np.ndarray:
    mapped = apply_axis_mapping(points_lidar, mapping_name)
    R_delta = rotvec_to_matrix(rotvec)
    points_cam = transform_points(mapped, R_delta, np.asarray(translation, dtype=np.float32))
    return points_cam


def robust_centroid(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return np.zeros(3, dtype=np.float32)
    return np.median(points.astype(np.float32), axis=0).astype(np.float32)


def estimate_initial_translation(samples, mapping_name: str, modality: str) -> np.ndarray:
    lidar_centroids = []
    target_centroids = []
    for sample in samples:
        if sample.lidar_points.size == 0:
            continue
        mapped = apply_axis_mapping(sample.lidar_points, mapping_name)
        lidar_centroids.append(robust_centroid(mapped))
        target = sample.depth_points
        if target.size > 0:
            target_centroids.append(robust_centroid(target))

    if not lidar_centroids or not target_centroids:
        return np.zeros(3, dtype=np.float32)

    lidar_c = np.mean(np.stack(lidar_centroids, axis=0), axis=0)
    target_c = np.mean(np.stack(target_centroids, axis=0), axis=0)
    return (target_c - lidar_c).astype(np.float32)


def aggregate_score(samples, mapping_name: str, rotvec: np.ndarray, translation: np.ndarray, modality: str, verbose: bool = False):
    rotvec = np.asarray(rotvec, dtype=np.float32).reshape(3)
    t = np.asarray(translation, dtype=np.float32).reshape(3)
    rot_penalty = 0.02 * float(np.dot(rotvec, rotvec))
    frame_scores = []
    frame_details: List[dict] = []
    metrics_acc = {"coverage": 0.0, "edge": 0.0, "depth": 0.0, "depth_err": 0.0}
    depth_count = 0
    for sample in samples:
        points = sample.lidar_points
        if points.size == 0:
            continue
        points_cam = apply_candidate_transform(points, mapping_name, rotvec, t)
        if modality == "rgb":
            score, metrics, proj = score_rgb_frame(points_cam, sample.rgb_K, getattr(sample, "rgb_edge", np.zeros_like(sample.depth_edge)))
        elif modality == "depth":
            score, metrics, proj = score_depth_frame(
                points_cam,
                sample.depth_K,
                sample.depth,
                sample.depth_edge,
                depth_error_scale=0.25,
            )
        else:
            raise ValueError(f"Unknown modality: {modality}")
        frame_scores.append(score)
        metrics_acc["coverage"] += metrics.get("coverage", 0.0)
        metrics_acc["edge"] += metrics.get("edge", 0.0)
        if "depth" in metrics:
            metrics_acc["depth"] += metrics.get("depth", 0.0)
            depth_count += 1
        if "depth_err" in metrics and np.isfinite(metrics["depth_err"]):
            metrics_acc["depth_err"] += metrics.get("depth_err", 0.0)
        frame_details.append(
            {
                "depth_path": sample.depth_path,
                "pcd_path": sample.pcd_path,
                "score": float(score),
                "metrics": {
                    k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) and v is not None else v)
                    for k, v in metrics.items()
                },
                "projected_count": int(proj["projected_count"]),
                "projected_count_raw": int(proj.get("projected_count_raw", proj["projected_count"])),
                "positive_depth_count": int(proj["positive_depth_count"]),
            }
        )

    if frame_scores:
        score = float(np.mean(frame_scores)) - rot_penalty
        metrics = {
            "coverage": metrics_acc["coverage"] / len(frame_scores),
            "edge": metrics_acc["edge"] / len(frame_scores),
            "rot_penalty": rot_penalty,
        }
        if modality == "depth" and depth_count > 0:
            metrics["depth"] = metrics_acc["depth"] / depth_count
            metrics["depth_err"] = metrics_acc["depth_err"] / depth_count
    else:
        score = -rot_penalty
        metrics = {"coverage": 0.0, "edge": 0.0, "rot_penalty": rot_penalty}

    if verbose:
        print(
            f"[{modality}] mapping={mapping_name} rotvec={rotvec.tolist()} t={t.tolist()} "
            f"score={score:.4f} cov={metrics.get('coverage', 0.0):.4f} edge={metrics.get('edge', 0.0):.4f}"
        )
    return score, metrics, frame_details


def optimize_transform(samples, mapping_name: str, modality: str, init_rotvec: np.ndarray, init_t: np.ndarray, search_radius: float, maxiter: int):
    lower = init_t - float(search_radius)
    upper = init_t + float(search_radius)
    cache: dict = {}

    def objective(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float32).reshape(6)
        rotvec = x[:3]
        trans = np.clip(x[3:], lower, upper)
        key = tuple(np.round(np.concatenate([rotvec, trans]), 4).tolist())
        if key in cache:
            return cache[key]
        score, _, _ = aggregate_score(samples, mapping_name, rotvec, trans, modality)
        value = -score
        cache[key] = value
        return value

    x0 = np.concatenate([np.asarray(init_rotvec, dtype=np.float32).reshape(3), np.asarray(init_t, dtype=np.float32).reshape(3)])
    result = minimize(
        objective,
        x0=x0,
        method="Powell",
        options={"maxiter": int(maxiter), "xtol": 1e-3, "ftol": 1e-4, "disp": False},
    )
    best_x = np.asarray(result.x, dtype=np.float32).reshape(6)
    best_rotvec = best_x[:3]
    best_t = np.clip(best_x[3:], lower, upper)
    best_score, best_metrics, frame_details = aggregate_score(samples, mapping_name, best_rotvec, best_t, modality)
    return best_rotvec, best_t, float(best_score), best_metrics, frame_details


def calibrate_modality(samples, modality: str, max_axis_candidates: int, search_radius: float, maxiter: int, use_depth_init_for_rgb: bool, search_axis_mapping: bool):
    mappings = signed_permutation_candidates(search_axis_mapping)
    quick_rank = []
    for mapping_name in mappings:
        init_t = estimate_initial_translation(samples, mapping_name, modality)
        init_rotvec = np.zeros(3, dtype=np.float32)
        if modality == "rgb" and use_depth_init_for_rgb:
            depth_init = estimate_initial_translation(samples, mapping_name, "depth")
            quick_inits = [np.zeros(3, dtype=np.float32), init_t, depth_init]
        else:
            quick_inits = [init_t]
        best_quick = -1e9
        best_quick_t = quick_inits[0]
        for cand_t in quick_inits:
            score, _, _ = aggregate_score(samples, mapping_name, init_rotvec, cand_t, modality)
            if score > best_quick:
                best_quick = score
                best_quick_t = cand_t
        quick_rank.append(
            {
                "mapping": mapping_name,
                "init_rotvec": init_rotvec.tolist(),
                "init_t": best_quick_t.tolist(),
                "quick_score": float(best_quick),
            }
        )

    quick_rank.sort(key=lambda x: (-x["quick_score"], x["mapping"]))
    keep = quick_rank[: max(1, min(max_axis_candidates, len(quick_rank)))]

    best = None
    for cand in keep:
        mapping_name = cand["mapping"]
        init_t = np.array(cand["init_t"], dtype=np.float32)
        init_rotvec = np.array(cand.get("init_rotvec", [0.0, 0.0, 0.0]), dtype=np.float32)
        if modality == "rgb" and use_depth_init_for_rgb:
            starts = [np.zeros(3, dtype=np.float32), init_t]
            local_best = None
            for start in starts:
                best_rotvec, best_t, score, metrics, frame_details = optimize_transform(
                    samples, mapping_name, modality, init_rotvec, start, search_radius, maxiter
                )
                record = {
                    "mapping": mapping_name,
                    "rotvec": best_rotvec.tolist(),
                    "translation": best_t.tolist(),
                    "score": float(score),
                    "metrics": metrics,
                    "frames": frame_details,
                }
                if local_best is None or record["score"] > local_best["score"]:
                    local_best = record
            record = local_best
        else:
            best_rotvec, best_t, score, metrics, frame_details = optimize_transform(
                samples, mapping_name, modality, init_rotvec, init_t, search_radius, maxiter
            )
            record = {
                "mapping": mapping_name,
                "rotvec": best_rotvec.tolist(),
                "translation": best_t.tolist(),
                "score": float(score),
                "metrics": metrics,
                "frames": frame_details,
            }

        record["init_t"] = init_t.tolist()
        record["init_rotvec"] = init_rotvec.tolist()
        record["quick_score"] = float(cand["quick_score"])
        if best is None or record["score"] > best["score"]:
            best = record

    assert best is not None
    best["quick_rank"] = keep
    return best


def build_overlay_for_best(samples, mapping_name: str, rotvec: np.ndarray, translation: np.ndarray, modality: str):
    best_sample = None
    best_score = -1e9
    best_proj = None
    best_metrics = None
    for sample in samples:
        points = sample.lidar_points
        if points.size == 0:
            continue
        points_cam = apply_candidate_transform(points, mapping_name, rotvec, translation)
        if modality == "rgb":
            score, metrics, proj = score_rgb_frame(points_cam, sample.rgb_K, sample.rgb_edge)
        else:
            score, metrics, proj = score_depth_frame(
                points_cam,
                sample.depth_K,
                sample.depth,
                sample.depth_edge,
                depth_error_scale=0.25,
            )
        if score > best_score:
            best_score = score
            best_sample = sample
            best_proj = proj
            best_metrics = metrics

    if best_sample is None or best_proj is None:
        raise ValueError(f"No valid sample found for modality={modality}")

    valid_depth = best_sample.depth[np.isfinite(best_sample.depth) & (best_sample.depth > 0)]
    depth_scale = float(np.percentile(valid_depth, 95.0)) if valid_depth.size > 0 else 1.0
    depth_scale = max(depth_scale, 1e-6)
    depth_vis = cv2.applyColorMap(
        (np.clip(best_sample.depth / depth_scale, 0.0, 1.0) * 255.0).astype(np.uint8),
        cv2.COLORMAP_TURBO,
    )
    depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_BGR2RGB)
    overlay = make_overlay(
        depth_vis,
        best_proj["u"],
        best_proj["v"],
        best_proj["z"],
        point_radius=2,
        title=f"Depth best | mapping={mapping_name}",
        stats_text=[
            f"score={best_score:.4f}",
            f"rotvec={np.asarray(rotvec).tolist()}",
            f"translation={translation.tolist()}",
            f"coverage={best_metrics.get('coverage', 0.0):.3f} edge={best_metrics.get('edge', 0.0):.3f}",
        ],
    )
    safe_metrics = {}
    for k, v in best_metrics.items():
        if v is None:
            safe_metrics[k] = None
        elif isinstance(v, (int, float, np.floating, np.integer)):
            safe_metrics[k] = float(v)
        else:
            safe_metrics[k] = v
    return overlay, safe_metrics


@dataclass
class DepthSample:
    part_root: str
    depth_path: str
    pcd_path: str
    depth_ts_ns: int
    pcd_ts_ns: int
    depth: np.ndarray
    depth_edge: np.ndarray
    depth_K: np.ndarray
    depth_size: Tuple[int, int]
    lidar_points: np.ndarray
    depth_points: np.ndarray
    rgb_path: str = ""


def discover_parts(seq_root: str) -> List[str]:
    seq_root = os.path.abspath(seq_root)
    if PART_NAME_RE.match(os.path.basename(seq_root)):
        return [seq_root]

    parts = [
        os.path.join(seq_root, name)
        for name in sorted(os.listdir(seq_root))
        if PART_NAME_RE.match(name) and os.path.isdir(os.path.join(seq_root, name))
    ]
    return parts if parts else [seq_root]


def select_parts(seq_root: str, part_index: int) -> List[str]:
    parts = discover_parts(seq_root)
    if part_index is None or int(part_index) < 0:
        return parts
    return [parts[int(part_index) % len(parts)]]


def resolve_intrinsics_path(part_root: str, seq_root: str) -> str:
    candidates = [
        os.path.join(part_root, "depth_camera_intrinsics.txt"),
        os.path.join(os.path.dirname(part_root), "depth_camera_intrinsics.txt"),
        os.path.join(seq_root, "depth_camera_intrinsics.txt"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise ValueError(
        "Could not locate depth intrinsics file. Expected one of: "
        + ", ".join(candidates)
    )


def _find_existing_modal_dir(part_root: str, name: str) -> str:
    candidates = [
        os.path.join(part_root, name),
        os.path.join(os.path.dirname(part_root), name),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    return os.path.join(part_root, name)


def _match_by_timestamp(
    reference_ts: np.ndarray,
    target_ts: np.ndarray,
    target_paths: Sequence[str],
    tolerance_ns: int,
) -> List[Tuple[int, str, int]]:
    rows: List[Tuple[int, str, int]] = []
    if reference_ts.size == 0 or target_ts.size == 0:
        return rows
    for idx, ref_ts in enumerate(reference_ts):
        nearest = int(np.argmin(np.abs(target_ts - ref_ts)))
        diff_ns = int(abs(int(target_ts[nearest]) - int(ref_ts)))
        if diff_ns <= tolerance_ns:
            rows.append((int(ref_ts), target_paths[nearest], int(target_ts[nearest])))
    return rows


def build_depth_samples(
    seq_root: str,
    part_index: int,
    tolerance_sec: float,
    max_pairs: int,
    max_lidar_points: int,
    max_depth_points: int,
    seed: int,
    depth_scale: float = 0.0,
) -> List[DepthSample]:
    parts = select_parts(seq_root, part_index)
    tolerance_ns = int(max(tolerance_sec, 0.0) * 1e9)
    rng = np.random.default_rng(seed)

    all_rows: List[Tuple[str, str, str, int, int]] = []
    for part_root in parts:
        depth_dir = _find_existing_modal_dir(part_root, "depth")
        lidar_dir = _find_existing_modal_dir(part_root, "lidar")

        depth_files = list_modal_files(depth_dir, IMAGE_EXTS)
        pcd_files = list_modal_files(lidar_dir, PCD_EXTS)
        if not depth_files or not pcd_files:
            continue

        depth_ts = np.array([timestamp_ns_from_path(path) for path in depth_files], dtype=np.int64)
        pcd_ts = np.array([timestamp_ns_from_path(path) for path in pcd_files], dtype=np.int64)
        matches = _match_by_timestamp(depth_ts, pcd_ts, pcd_files, tolerance_ns)
        if not matches:
            continue

        for depth_ts_ns, pcd_path, pcd_ts_ns in matches:
            depth_idx = int(np.argmin(np.abs(depth_ts - depth_ts_ns)))
            all_rows.append((part_root, depth_files[depth_idx], pcd_path, int(depth_ts_ns), int(pcd_ts_ns)))

    if not all_rows:
        raise ValueError(
            "No matched depth/LiDAR pairs were found under seq_root. "
            "Check the folder layout, timestamp format, and tolerance."
        )

    if max_pairs > 0 and len(all_rows) > max_pairs:
        indices = np.linspace(0, len(all_rows) - 1, num=max_pairs, dtype=int)
        all_rows = [all_rows[i] for i in indices]

    samples: List[DepthSample] = []
    for part_root, depth_path, pcd_path, depth_ts_ns, pcd_ts_ns in all_rows:
        depth_intrinsics_path = resolve_intrinsics_path(part_root, seq_root)
        depth_K_raw, depth_src_size = parse_camera_intrinsics_file(depth_intrinsics_path)

        depth = load_depth_image(depth_path, depth_scale=depth_scale)
        depth_edge = compute_edge_map_depth(depth)
        depth_K = scale_intrinsics(depth_K_raw, depth_src_size, (depth.shape[1], depth.shape[0]))
        lidar_points = load_pcd_points(pcd_path, max_lidar_points, rng)
        depth_points = backproject_depth_to_points(depth, depth_K, max_depth_points, rng)

        samples.append(
            DepthSample(
                part_root=part_root,
                depth_path=depth_path,
                pcd_path=pcd_path,
                depth_ts_ns=depth_ts_ns,
                pcd_ts_ns=pcd_ts_ns,
                depth=depth,
                depth_edge=depth_edge,
                depth_K=depth_K,
                depth_size=(depth.shape[1], depth.shape[0]),
                lidar_points=lidar_points,
                depth_points=depth_points,
            )
        )

    return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate LiDAR extrinsics against depth only.")
    parser.add_argument(
        "--seq_root",
        type=str,
        default="/add02/users/xuyh/seq1",
        help="Sequence root containing one or more part folders.",
    )
    parser.add_argument(
        "--part_index",
        type=int,
        default=-1,
        help="If >= 0, only calibrate the selected part. Default -1 uses all parts.",
    )
    parser.add_argument("--output_dir", type=str, default="output/calibration")
    parser.add_argument("--tolerance", type=float, default=0.03, help="Max timestamp difference in seconds.")
    parser.add_argument("--max_pairs", type=int, default=400, help="Maximum matched frames used for calibration.")
    parser.add_argument("--max_lidar_points", type=int, default=12000, help="Max LiDAR points per frame.")
    parser.add_argument("--max_depth_points", type=int, default=12000, help="Max depth points per frame for initialization.")
    parser.add_argument("--max_axis_candidates", type=int, default=6, help="How many axis candidates to refine.")
    parser.add_argument("--search_radius", type=float, default=1.0, help="Translation search radius in meters.")
    parser.add_argument("--maxiter", type=int, default=40, help="Powell iterations per candidate.")
    parser.add_argument(
        "--search_axis_mapping",
        action="store_true",
        default=False,
        help="Search signed LiDAR axis permutations. Disable this when LiDAR is already aligned.",
    )
    parser.add_argument(
        "--depth_scale",
        type=float,
        default=0.0,
        help="Depth-to-meter scale factor. Use 100 for cm->m, 1000 for mm->m, 0 to auto-infer.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_top_k", type=int, default=3, help="Print the top-k quick candidates.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    parts = select_parts(args.seq_root, args.part_index)
    print(f"[Input] seq_root={os.path.abspath(args.seq_root)}")
    print(f"[Input] parts={parts}")

    samples = build_depth_samples(
        seq_root=args.seq_root,
        part_index=args.part_index,
        tolerance_sec=args.tolerance,
        max_pairs=args.max_pairs,
        max_lidar_points=args.max_lidar_points,
        max_depth_points=args.max_depth_points,
        seed=args.seed,
        depth_scale=args.depth_scale,
    )
    print(f"[Data] matched depth/LiDAR pairs: {len(samples)}")
    if not samples:
        raise RuntimeError("No matched samples available.")

    depth_result = calibrate_modality(
        samples=samples,
        modality="depth",
        max_axis_candidates=args.max_axis_candidates,
        search_radius=args.search_radius,
        maxiter=args.maxiter,
        use_depth_init_for_rgb=False,
        search_axis_mapping=args.search_axis_mapping,
    )

    depth_t = np.array(depth_result["translation"], dtype=np.float32)
    depth_rotvec = np.array(depth_result["rotvec"], dtype=np.float32)
    depth_overlay, depth_overlay_metrics = build_overlay_for_best(
        samples, depth_result["mapping"], depth_rotvec, depth_t, "depth"
    )

    overlay_path = os.path.join(args.output_dir, "best_depth_overlay.png")
    cv2.imwrite(overlay_path, cv2.cvtColor(depth_overlay, cv2.COLOR_RGB2BGR))

    lidar_to_depth_T = compose_transform(depth_result["mapping"], depth_rotvec, depth_t)
    summary = {
        "seq_root": os.path.abspath(args.seq_root),
        "parts": parts,
        "num_samples": len(samples),
        "args": vars(args),
        "depth": {
            "mapping": depth_result["mapping"],
            "search_axis_mapping": bool(args.search_axis_mapping),
            "rotvec": depth_rotvec.tolist(),
            "translation": summarize_translation(depth_t),
            "score": float(depth_result["score"]),
            "metrics": {k: float(v) for k, v in depth_result["metrics"].items()},
            "T_lidar_to_depth": lidar_to_depth_T.tolist(),
            "overlay": overlay_path,
        },
        "depth_overlay_metrics": depth_overlay_metrics,
        "quick_rank_depth": depth_result["quick_rank"],
        "samples": [
            {
                "part_root": s.part_root,
                "depth_path": s.depth_path,
                "pcd_path": s.pcd_path,
                "depth_ts_ns": s.depth_ts_ns,
                "pcd_ts_ns": s.pcd_ts_ns,
            }
            for s in samples
        ],
    }

    summary_path = os.path.join(args.output_dir, "calibration_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[Output] overlay:     {overlay_path}")
    print(f"[Output] summary json: {summary_path}")
    print(
        f"[Result][DEPTH] mapping={depth_result['mapping']} score={depth_result['score']:.4f} "
        f"rotvec={depth_rotvec.tolist()} t={depth_t.tolist()}"
    )
    print(
        f"[Translation] LiDAR -> depth camera = {depth_t.tolist()} "
        f"(norm={float(np.linalg.norm(depth_t)):.4f} m)"
    )

    if args.save_top_k > 0:
        for idx, cand in enumerate(depth_result["quick_rank"][: args.save_top_k]):
            print(f"[TopDepth] {idx}: {cand['mapping']} quick_score={cand['quick_score']:.4f}")


if __name__ == "__main__":
    main()
