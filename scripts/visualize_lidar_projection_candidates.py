#!/usr/bin/env python3
# coding: utf-8
"""
Project one LiDAR PCD onto one RGB image under multiple coordinate-frame
hypotheses, then save every overlay for manual inspection.

Typical usage:

  python scripts/visualize_lidar_projection_candidates.py ^
      --seq_root /add02/users/xuyh/seq1/shuangchuang_seq1_night1th ^
      --image_index 120 ^
      --output_dir output/projection_candidates

  python scripts/visualize_lidar_projection_candidates.py ^
      --img_path /path/to/color_xxx.png ^
      --pcd_path /path/to/livox_xxx.pcd ^
      --intrinsics_path /path/to/color_camera_intrinsics.txt ^
      --output_dir output/projection_candidates
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import open3d as o3d
from PIL import Image


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".tif")
PCD_EXTS = (".pcd",)
PART_NAME_RE = re.compile(r"^shuangchuang_seq\d+_(?:night|daytime)\d+th$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize LiDAR-to-RGB projections under many coordinate-axis hypotheses."
    )
    parser.add_argument(
        "--seq_root",
        type=str,
        default="/add02/users/xuyh/seq1/",
        help="Sequence root or one part directory.",
    )
    parser.add_argument(
        "--img_path",
        type=str,
        default="/add02/users/xuyh/seq1/shuangchuang_seq1_night1th/rgb/",
        help="Explicit RGB image path, or an RGB directory.",
    )
    parser.add_argument(
        "--pcd_path",
        type=str,
        default="/add02/users/xuyh/seq1/shuangchuang_seq1_night1th/lidar/",
        help="Explicit PCD path, or a LiDAR directory.",
    )
    parser.add_argument(
        "--intrinsics_path",
        type=str,
        default="",
        help="Explicit color_camera_intrinsics.txt path. Optional when --seq_root is provided.",
    )
    parser.add_argument(
        "--part_index",
        type=int,
        default=0,
        help="When seq_root is a parent directory with multiple parts, choose one part by index.",
    )
    parser.add_argument(
        "--image_index",
        type=int,
        default=-1,
        help="Choose one RGB image by index within the selected part. -1 picks the middle image.",
    )
    parser.add_argument(
        "--max_time_diff_sec",
        type=float,
        default=0.05,
        help="Only used for warning when automatically pairing image and PCD by nearest timestamp.",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=120000,
        help="Randomly subsample the point cloud before projection for speed and cleaner overlays.",
    )
    parser.add_argument(
        "--top_k_grid",
        type=int,
        default=12,
        help="How many best candidates to include in the contact-sheet grid.",
    )
    parser.add_argument("--point_radius", type=int, default=1, help="Projected point radius in pixels.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/xuyh/mapanything/output/",
        help="Directory to store all overlay images and summary metadata.",
    )
    parser.add_argument(
        "--depth_percentile_max",
        type=float,
        default=98.0,
        help="Upper percentile for depth color normalization.",
    )
    parser.add_argument(
        "--specific_img",
        type=str,
        default="1765545966177996871",
        help="Specific RGB image key: exact filename, stem fragment, or timestamp digits.",
    )
    return parser.parse_args()


def timestamp_ns_from_path(path: str) -> int:
    basename = os.path.basename(path)
    match = re.search(r"color_(\d+)", basename)
    if match is None:
        match = re.search(r"livox_(\d+)", basename)
    if match is None:
        match = re.search(r"(\d+)", basename)
    if match is not None:
        return int(match.group(1))
    return int(os.path.getmtime(path) * 1e9)


def list_part_roots(seq_root: str) -> List[str]:
    seq_root = os.path.abspath(seq_root)
    if PART_NAME_RE.match(os.path.basename(seq_root)):
        return [seq_root]

    parts = [
        os.path.join(seq_root, name)
        for name in sorted(os.listdir(seq_root))
        if PART_NAME_RE.match(name) and os.path.isdir(os.path.join(seq_root, name))
    ]
    if parts:
        return parts
    return [seq_root]


def _parse_camera_intrinsics_file(file_path: str) -> Tuple[np.ndarray, Tuple[int, int]]:
    K = None
    width = None
    height = None
    if not file_path or not os.path.exists(file_path):
        return np.eye(3, dtype=np.float32), (0, 0)

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        width_match = re.search(r"width:\s*(\d+)", content)
        height_match = re.search(r"height:\s*(\d+)", content)
        if width_match:
            width = int(width_match.group(1))
        if height_match:
            height = int(height_match.group(1))

        k_match = re.search(r"K:\s*\[\[(.*?)\]\]", content, re.DOTALL)
        if k_match:
            nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", k_match.group(1))
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


def discover_rgb_and_pcd(part_root: str) -> Tuple[List[str], List[str]]:
    rgb_dir = os.path.join(part_root, "rgb")
    lidar_dir = os.path.join(part_root, "lidar")
    if not os.path.isdir(rgb_dir):
        raise ValueError(f"RGB directory not found: {rgb_dir}")
    if not os.path.isdir(lidar_dir):
        raise ValueError(f"LiDAR directory not found: {lidar_dir}")

    img_paths = [
        os.path.join(rgb_dir, name)
        for name in sorted(os.listdir(rgb_dir))
        if name.lower().endswith(IMAGE_EXTS)
    ]
    pcd_paths = [
        os.path.join(lidar_dir, name)
        for name in sorted(os.listdir(lidar_dir))
        if name.lower().endswith(PCD_EXTS)
    ]
    if not img_paths:
        raise ValueError(f"No RGB images found in {rgb_dir}")
    if not pcd_paths:
        raise ValueError(f"No PCD files found in {lidar_dir}")
    return img_paths, pcd_paths


def list_images_in_dir(rgb_dir: str) -> List[str]:
    if not os.path.isdir(rgb_dir):
        raise ValueError(f"RGB directory not found: {rgb_dir}")
    img_paths = [
        os.path.join(rgb_dir, name)
        for name in sorted(os.listdir(rgb_dir))
        if name.lower().endswith(IMAGE_EXTS)
    ]
    if not img_paths:
        raise ValueError(f"No RGB images found in {rgb_dir}")
    return img_paths


def list_pcds_in_dir(lidar_dir: str) -> List[str]:
    if not os.path.isdir(lidar_dir):
        raise ValueError(f"LiDAR directory not found: {lidar_dir}")
    pcd_paths = [
        os.path.join(lidar_dir, name)
        for name in sorted(os.listdir(lidar_dir))
        if name.lower().endswith(PCD_EXTS)
    ]
    if not pcd_paths:
        raise ValueError(f"No PCD files found in {lidar_dir}")
    return pcd_paths


def find_specific_image(img_paths: Sequence[str], specific_img: str) -> str:
    if not specific_img:
        raise ValueError("specific_img is empty")

    specific_img = str(specific_img).strip()
    if os.path.isfile(specific_img):
        return os.path.abspath(specific_img)

    target = os.path.basename(specific_img)
    stem = Path(target).stem
    digits = re.sub(r"\D", "", specific_img)

    exact_matches = []
    stem_matches = []
    digit_matches = []
    for path in img_paths:
        base = os.path.basename(path)
        base_stem = Path(base).stem
        if base == target or base_stem == stem:
            exact_matches.append(path)
            continue
        if target and target in base:
            stem_matches.append(path)
            continue
        if digits:
            path_digits = re.sub(r"\D", "", base)
            if digits == path_digits or digits in path_digits:
                digit_matches.append(path)

    if exact_matches:
        return exact_matches[0]
    if stem_matches:
        return stem_matches[0]
    if digit_matches:
        return digit_matches[0]

    raise ValueError(
        f"Could not resolve specific_img={specific_img!r} from {len(img_paths)} images."
    )


def find_best_timestamp_pair(
    img_paths: Sequence[str],
    pcd_paths: Sequence[str],
    max_time_diff_sec: float,
    image_index: int,
) -> Tuple[str, str, float]:
    pcd_timestamps = np.array([timestamp_ns_from_path(path) for path in pcd_paths], dtype=np.int64)
    tolerance_ns = int(max_time_diff_sec * 1e9)

    if image_index >= 0:
        if image_index >= len(img_paths):
            raise IndexError(f"--image_index {image_index} out of range for {len(img_paths)} RGB images.")
        candidate_images = [img_paths[image_index]]
    else:
        candidate_images = list(img_paths)

    best_pair = None
    best_diff_ns = None
    for img_path in candidate_images:
        img_ts = timestamp_ns_from_path(img_path)
        nearest_idx = int(np.argmin(np.abs(pcd_timestamps - img_ts)))
        diff_ns = int(abs(int(pcd_timestamps[nearest_idx]) - img_ts))
        if best_diff_ns is None or diff_ns < best_diff_ns:
            best_diff_ns = diff_ns
            best_pair = (img_path, pcd_paths[nearest_idx], diff_ns)

    if best_pair is None or best_diff_ns is None:
        raise ValueError("Could not find any RGB-PCD pair.")
    if best_diff_ns > tolerance_ns:
        raise ValueError(
            f"No RGB-PCD pair satisfies --max_time_diff_sec={max_time_diff_sec:.6f}s. "
            f"Best diff was {best_diff_ns / 1e9:.6f}s."
        )
    return best_pair[0], best_pair[1], best_pair[2] / 1e9


def choose_image_and_pcd(args: argparse.Namespace) -> Tuple[str, str, str]:
    img_path = args.img_path
    pcd_path = args.pcd_path
    intrinsics_path = args.intrinsics_path

    if img_path and pcd_path:
        img_path = os.path.abspath(img_path)
        pcd_path = os.path.abspath(pcd_path)
        if os.path.isdir(img_path) and os.path.isdir(pcd_path):
            img_paths = list_images_in_dir(img_path)
            pcd_paths = list_pcds_in_dir(pcd_path)
            if args.specific_img:
                img_path = find_specific_image(img_paths, args.specific_img)
                img_ts = timestamp_ns_from_path(img_path)
                pcd_timestamps = np.array([timestamp_ns_from_path(path) for path in pcd_paths], dtype=np.int64)
                nearest_idx = int(np.argmin(np.abs(pcd_timestamps - img_ts)))
                pcd_path = pcd_paths[nearest_idx]
                diff_sec = abs(timestamp_ns_from_path(pcd_path) - img_ts) / 1e9
                if diff_sec > args.max_time_diff_sec:
                    raise ValueError(
                        f"Selected specific_img={args.specific_img!r} but nearest PCD diff "
                        f"{diff_sec:.6f}s exceeds --max_time_diff_sec={args.max_time_diff_sec:.6f}s."
                    )
                print(f"[Pair] selected specific_img={os.path.basename(img_path)} diff={diff_sec:.6f}s")
            else:
                img_path, pcd_path, diff_sec = find_best_timestamp_pair(
                    img_paths,
                    pcd_paths,
                    max_time_diff_sec=args.max_time_diff_sec,
                    image_index=args.image_index,
                )
                print(f"[Pair] selected from directories with diff: {diff_sec:.6f}s")
            if not intrinsics_path:
                part_root = os.path.dirname(img_path)
                candidates = [
                    os.path.join(os.path.dirname(part_root), "color_camera_intrinsics.txt"),
                    os.path.join(os.path.dirname(os.path.dirname(part_root)), "color_camera_intrinsics.txt"),
                ]
                intrinsics_path = next((path for path in candidates if os.path.exists(path)), "")
            if not intrinsics_path:
                raise ValueError("Could not locate color_camera_intrinsics.txt for the provided RGB/LiDAR directories.")
            return img_path, pcd_path, os.path.abspath(intrinsics_path)

        if not intrinsics_path:
            raise ValueError("--intrinsics_path is required when --img_path and --pcd_path are given directly.")
        return img_path, pcd_path, os.path.abspath(intrinsics_path)

    if not args.seq_root:
        raise ValueError("Provide either (--img_path, --pcd_path, --intrinsics_path) or --seq_root.")

    part_roots = list_part_roots(args.seq_root)
    if args.part_index < 0 or args.part_index >= len(part_roots):
        raise IndexError(f"--part_index {args.part_index} out of range for {len(part_roots)} parts.")
    part_root = part_roots[args.part_index]
    img_paths, pcd_paths = discover_rgb_and_pcd(part_root)
    if args.specific_img:
        img_path = find_specific_image(img_paths, args.specific_img)
        img_ts = timestamp_ns_from_path(img_path)
        pcd_timestamps = np.array([timestamp_ns_from_path(path) for path in pcd_paths], dtype=np.int64)
        nearest_idx = int(np.argmin(np.abs(pcd_timestamps - img_ts)))
        pcd_path = pcd_paths[nearest_idx]
        diff_sec = abs(timestamp_ns_from_path(pcd_path) - img_ts) / 1e9
        if diff_sec > args.max_time_diff_sec:
            raise ValueError(
                f"Selected specific_img={args.specific_img!r} but nearest PCD diff "
                f"{diff_sec:.6f}s exceeds --max_time_diff_sec={args.max_time_diff_sec:.6f}s."
            )
        print(f"[Pair] selected specific_img={os.path.basename(img_path)} diff={diff_sec:.6f}s")
    else:
        img_path, pcd_path, diff_sec = find_best_timestamp_pair(
            img_paths,
            pcd_paths,
            max_time_diff_sec=args.max_time_diff_sec,
            image_index=args.image_index,
        )
        print(f"[Pair] selected from seq_root with diff: {diff_sec:.6f}s")

    if not intrinsics_path:
        candidates = [
            os.path.join(part_root, "color_camera_intrinsics.txt"),
            os.path.join(os.path.abspath(args.seq_root), "color_camera_intrinsics.txt"),
        ]
        intrinsics_path = next((path for path in candidates if os.path.exists(path)), "")
    if not intrinsics_path:
        raise ValueError("Could not locate color_camera_intrinsics.txt automatically.")

    return os.path.abspath(img_path), os.path.abspath(pcd_path), os.path.abspath(intrinsics_path)


def load_rgb(image_path: str) -> np.ndarray:
    rgb = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    return rgb


def load_pcd_points(pcd_path: str, max_points: int) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points, dtype=np.float32)
    if points.size == 0:
        raise ValueError(f"PCD contains no points: {pcd_path}")
    if max_points > 0 and len(points) > max_points:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(points), size=max_points, replace=False)
        points = points[indices]
    return points


def signed_permutation_candidates() -> List[str]:
    results: List[str] = []
    order = {"x": 0, "y": 1, "z": 2}
    for perm in itertools.permutations(("x", "y", "z"), 3):
        idxs = [order[a] for a in perm]
        inv_count = sum(1 for i in range(3) for j in range(i + 1, 3) if idxs[i] > idxs[j])
        perm_sign = -1 if inv_count % 2 else 1
        for sign_bits in itertools.product((1, -1), repeat=3):
            if perm_sign * (sign_bits[0] * sign_bits[1] * sign_bits[2]) != 1:
                continue
            name = "_".join(f"{'-' if sign < 0 else ''}{axis}" for axis, sign in zip(perm, sign_bits))
            results.append(name)
    results.sort()
    return results


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
    if len(components) != 3:
        raise ValueError(f"Invalid mapping name: {mapping_name}")
    mapped = np.stack([axis_vectors[name] for name in components], axis=1)
    return mapped.astype(np.float32)


def project_points(points_cam: np.ndarray, K: np.ndarray, image_size: Tuple[int, int]) -> Dict[str, object]:
    width, height = image_size
    x = points_cam[:, 0]
    y = points_cam[:, 1]
    z = points_cam[:, 2]
    positive = z > 1e-4
    positive_count = int(np.count_nonzero(positive))

    if positive_count == 0:
        return {
            "positive_depth_count": 0,
            "projected_count": 0,
            "u": np.empty((0,), dtype=np.float32),
            "v": np.empty((0,), dtype=np.float32),
            "depth": np.empty((0,), dtype=np.float32),
            "u_range": None,
            "v_range": None,
        }

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    x = x[positive]
    y = y[positive]
    z = z[positive]
    u = (fx * x / z) + cx
    v = (fy * y / z) + cy
    in_image = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u_in = u[in_image]
    v_in = v[in_image]
    z_in = z[in_image]

    return {
        "positive_depth_count": positive_count,
        "projected_count": int(np.count_nonzero(in_image)),
        "u": u_in.astype(np.float32),
        "v": v_in.astype(np.float32),
        "depth": z_in.astype(np.float32),
        "u_range": [float(np.min(u)), float(np.max(u))],
        "v_range": [float(np.min(v)), float(np.max(v))],
    }


def depth_to_bgr(depth: np.ndarray, percentile_max: float) -> np.ndarray:
    if depth.size == 0:
        return np.empty((0, 3), dtype=np.uint8)
    z_min = float(np.min(depth))
    z_max = float(np.percentile(depth, percentile_max))
    z_max = max(z_max, z_min + 1e-6)
    norm = np.clip((depth - z_min) / (z_max - z_min), 0.0, 1.0)
    colors = cv2.applyColorMap((norm * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    return colors.reshape(-1, 3)


def overlay_projection(
    rgb: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    depth: np.ndarray,
    point_radius: int,
    percentile_max: float,
    title: str,
    stats_text: Sequence[str],
) -> np.ndarray:
    canvas = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
    colors = depth_to_bgr(depth, percentile_max)
    for idx in range(len(u)):
        center = (int(round(float(u[idx]))), int(round(float(v[idx]))))
        color = tuple(int(c) for c in colors[idx].tolist())
        cv2.circle(canvas, center, point_radius, color, thickness=-1, lineType=cv2.LINE_AA)

    panel_h = min(96, max(72, canvas.shape[0] // 6))
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], panel_h), (0, 0, 0), thickness=-1)
    cv2.putText(
        canvas,
        title,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    y = 54
    for line in stats_text:
        cv2.putText(
            canvas,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        y += 20
    return canvas


def make_contact_sheet(images: Sequence[np.ndarray], labels: Sequence[str], cols: int = 3) -> np.ndarray:
    if not images:
        raise ValueError("No images to compose.")
    h, w = images[0].shape[:2]
    rows = int(math.ceil(len(images) / float(cols)))
    sheet = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for idx, (img, label) in enumerate(zip(images, labels)):
        r = idx // cols
        c = idx % cols
        y0 = r * h
        x0 = c * w
        tile = img.copy()
        cv2.rectangle(tile, (0, h - 28), (w, h), (0, 0, 0), thickness=-1)
        cv2.putText(tile, label, (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        sheet[y0:y0 + h, x0:x0 + w] = tile
    return sheet


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    img_path, pcd_path, intrinsics_path = choose_image_and_pcd(args)
    print(f"[Input] image: {img_path}")
    print(f"[Input] pcd: {pcd_path}")
    print(f"[Input] intrinsics: {intrinsics_path}")

    rgb = load_rgb(img_path)
    height, width = rgb.shape[:2]
    K_raw, src_size = _parse_camera_intrinsics_file(intrinsics_path)
    K = scale_intrinsics(K_raw, src_size, (width, height))
    points = load_pcd_points(pcd_path, max_points=args.max_points)

    image_ts = timestamp_ns_from_path(img_path)
    pcd_ts = timestamp_ns_from_path(pcd_path)
    time_diff_sec = abs(pcd_ts - image_ts) / 1e9

    candidate_names = signed_permutation_candidates()
    results: List[Dict[str, object]] = []
    overview_images: List[np.ndarray] = []
    overview_labels: List[str] = []

    for candidate_name in candidate_names:
        points_cam = apply_axis_mapping(points, candidate_name)
        proj = project_points(points_cam, K, (width, height))
        stats_text = [
            f"projected={proj['projected_count']} / positive_depth={proj['positive_depth_count']}",
            f"time_diff={time_diff_sec:.6f}s  image={os.path.basename(img_path)}",
            f"pcd={os.path.basename(pcd_path)}",
        ]
        overlay = overlay_projection(
            rgb=rgb,
            u=proj["u"],
            v=proj["v"],
            depth=proj["depth"],
            point_radius=args.point_radius,
            percentile_max=args.depth_percentile_max,
            title=f"Axis mapping: {candidate_name}",
            stats_text=stats_text,
        )
        out_name = f"{len(results):02d}_{candidate_name.replace('-', 'neg')}.png"
        out_path = os.path.join(args.output_dir, out_name)
        cv2.imwrite(out_path, overlay)

        result = {
            "name": candidate_name,
            "projected_count": int(proj["projected_count"]),
            "positive_depth_count": int(proj["positive_depth_count"]),
            "u_range": proj["u_range"],
            "v_range": proj["v_range"],
            "output_image": out_path,
        }
        results.append(result)

    results.sort(key=lambda item: (-item["projected_count"], item["name"]))

    top_k = max(1, min(args.top_k_grid, len(results)))
    for item in results[:top_k]:
        overlay = cv2.imread(item["output_image"], cv2.IMREAD_COLOR)
        overview_images.append(overlay)
        overview_labels.append(f"{item['name']} | proj={item['projected_count']}")

    contact_sheet = make_contact_sheet(overview_images, overview_labels, cols=3)
    contact_sheet_path = os.path.join(args.output_dir, "projection_contact_sheet.png")
    cv2.imwrite(contact_sheet_path, contact_sheet)

    summary = {
        "img_path": img_path,
        "pcd_path": pcd_path,
        "intrinsics_path": intrinsics_path,
        "image_timestamp_ns": image_ts,
        "pcd_timestamp_ns": pcd_ts,
        "time_diff_sec": time_diff_sec,
        "image_size": [width, height],
        "intrinsics_src_size": list(src_size),
        "intrinsics_scaled": K.tolist(),
        "num_points_used": int(len(points)),
        "results_sorted": results,
        "contact_sheet": contact_sheet_path,
    }
    summary_path = os.path.join(args.output_dir, "projection_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[Output] contact sheet: {contact_sheet_path}")
    print(f"[Output] summary json: {summary_path}")
    if results:
        best = results[0]
        print(
            f"[Best] {best['name']} projected={best['projected_count']} / "
            f"{best['positive_depth_count']}"
        )


if __name__ == "__main__":
    main()
