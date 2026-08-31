#!/usr/bin/env python3
"""Visualize whether step-0 OSVI waypoint predictions vary across rollouts.

This script is intentionally post-hoc: it reads outputs already written by
eval_scripted_rollout_inference.py and does not run model inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


TASK_RE = re.compile(r"task_?(\d+)")
TRAJ_RE = re.compile(r"traj_?(\d+)")
MODES = {"train", "test"}


def natural_key(value: Any) -> List[Any]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(value))]


def normalize_task(value: str) -> str:
    match = TASK_RE.fullmatch(value)
    if match:
        return f"task_{int(match.group(1)):02d}"
    return value


def parse_identity(json_path: Path, eval_root: Path) -> Dict[str, str]:
    try:
        parts = json_path.relative_to(eval_root).parts
    except ValueError:
        parts = json_path.parts

    mode = next((part for part in parts if part in MODES), "")
    task_matches = [normalize_task(part) for part in parts if TASK_RE.fullmatch(part)]
    traj_matches = [part for part in parts if TRAJ_RE.fullmatch(part)]

    return {
        "mode": mode,
        "task": task_matches[0] if task_matches else "unknown_task",
        "rollout": traj_matches[0] if traj_matches else json_path.parent.parent.name,
    }


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r") as handle:
        return json.load(handle)


def find_step_jsons(eval_root: Path, step: int) -> List[Path]:
    pattern = f"osvi_waypoints_t{step:03d}.json"
    return sorted(eval_root.rglob(pattern), key=natural_key)


def candidate_image_paths(step_dir: Path, step: int) -> List[Path]:
    return [
        step_dir / f"osvi_waypoints_gt_overlay_t{step:03d}.png",
        step_dir / f"osvi_waypoints_overlay_t{step:03d}.png",
        step_dir / f"osvi_input_front_t{step:03d}.png",
    ]


def choose_image_path(step_dir: Path, step: int) -> Optional[Path]:
    for path in candidate_image_paths(step_dir, step):
        if path.is_file():
            return path
    return None


def image_waypoints_to_pixels(image_waypoints: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
    height, width = image_shape
    points = np.asarray(image_waypoints, dtype=np.float64)
    pixels = np.full((len(points), 2), np.nan, dtype=np.float64)
    if points.size == 0:
        return pixels
    pixels[:, 0] = (points[:, 0] + 1.0) * 0.5 * (width - 1)
    pixels[:, 1] = (1.0 - points[:, 1]) * 0.5 * (height - 1)
    return pixels


def draw_pred_overlay_on_input(input_path: Path, image_waypoints: np.ndarray, threshold: float, out_path: Path) -> Optional[Path]:
    image_bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        return None
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]
    pixels = image_waypoints_to_pixels(image_waypoints, (height, width))
    overlay = image_rgb.copy()
    for index, (point, waypoint) in enumerate(zip(pixels, image_waypoints), start=1):
        if not np.isfinite(point).all():
            continue
        x, y = int(round(point[0])), int(round(point[1]))
        x = int(np.clip(x, 0, width - 1))
        y = int(np.clip(y, 0, height - 1))
        grip = float(waypoint[3])
        close = grip >= threshold
        color = (255, 40, 40) if close else (40, 220, 255)
        radius = 7 if close else 5
        cv2.circle(overlay, (x, y), radius, color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(overlay, (x, y), radius + 2, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
        cv2.putText(
            overlay,
            f"{index} g={grip:.2f}",
            (min(x + 9, width - 1), max(y - 9, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)):
        return out_path
    return None


def ensure_visual_path(step_dir: Path, step: int, image_waypoints: np.ndarray, threshold: float) -> Optional[Path]:
    existing = choose_image_path(step_dir, step)
    if existing is None:
        return None
    if existing.name.startswith("osvi_input_front"):
        generated = step_dir / f"osvi_step0_pred_overlay_t{step:03d}.png"
        return draw_pred_overlay_on_input(existing, image_waypoints, threshold, generated)
    return existing


def choose_pick_waypoint(
    image_waypoints: np.ndarray,
    threshold: float,
    policy: str,
    waypoint_index: int,
) -> Tuple[int, str]:
    if len(image_waypoints) == 0:
        return -1, "none"

    if policy == "waypoint-index":
        index = int(np.clip(waypoint_index, 0, len(image_waypoints) - 1))
        return index, "waypoint-index"

    if policy == "max-gripper":
        return int(np.nanargmax(image_waypoints[:, 3])), "max-gripper"

    close_indices = np.flatnonzero(image_waypoints[:, 3] >= threshold)
    if len(close_indices):
        return int(close_indices[0]), "first-close"
    return int(np.nanargmax(image_waypoints[:, 3])), "max-gripper-fallback"


def pairwise_max(points: np.ndarray) -> Optional[float]:
    points = np.asarray(points, dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 2:
        return None
    deltas = points[:, None, :] - points[None, :, :]
    return float(np.linalg.norm(deltas, axis=-1).max())


def pairwise_mean(points: np.ndarray) -> Optional[float]:
    points = np.asarray(points, dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 2:
        return None
    deltas = points[:, None, :] - points[None, :, :]
    dists = np.linalg.norm(deltas, axis=-1)
    upper = dists[np.triu_indices(len(points), k=1)]
    return float(upper.mean()) if len(upper) else None


def numeric_std(values: np.ndarray) -> Optional[float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None
    return float(values.std())


def rounded_unique_count(points: np.ndarray, decimals: int) -> int:
    points = np.asarray(points, dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) == 0:
        return 0
    rounded = np.round(points, decimals=decimals)
    return len({tuple(row.tolist()) for row in rounded})


def record_from_json(
    json_path: Path,
    eval_root: Path,
    args,
) -> Optional[Dict[str, Any]]:
    payload = load_json(json_path)
    image_waypoints = np.asarray(payload.get("image_waypoints", []), dtype=np.float64)
    base_waypoints = np.asarray(payload.get("base_waypoints", []), dtype=np.float64)
    if image_waypoints.ndim != 2 or image_waypoints.shape[1] < 4 or len(image_waypoints) == 0:
        return None

    pick_index, pick_policy = choose_pick_waypoint(
        image_waypoints,
        args.pred_gripper_threshold,
        args.pick_policy,
        args.pick_waypoint_index,
    )
    if pick_index < 0:
        return None

    identity = parse_identity(json_path, eval_root)
    step_dir = json_path.parent
    visual_path = ensure_visual_path(step_dir, args.step, image_waypoints, args.pred_gripper_threshold)

    base_pick = np.full(4, np.nan, dtype=np.float64)
    if base_waypoints.ndim == 2 and base_waypoints.shape[0] > pick_index:
        base_pick[: min(4, base_waypoints.shape[1])] = base_waypoints[pick_index, : min(4, base_waypoints.shape[1])]

    image_pick = image_waypoints[pick_index]
    return {
        **identity,
        "step": args.step,
        "json_path": str(json_path),
        "visual_path": "" if visual_path is None else str(visual_path),
        "context_source": payload.get("context_source", ""),
        "pick_waypoint_index": pick_index + 1,
        "pick_policy": pick_policy,
        "pick_x_norm": float(image_pick[0]),
        "pick_y_norm": float(image_pick[1]),
        "pick_z": float(image_pick[2]),
        "pick_gripper": float(image_pick[3]),
        "pick_base_x": float(base_pick[0]),
        "pick_base_y": float(base_pick[1]),
        "pick_base_z": float(base_pick[2]),
        "pick_base_gripper": float(base_pick[3]),
        "image_waypoints_json": json.dumps(image_waypoints.tolist()),
        "base_waypoints_json": json.dumps(base_waypoints.tolist()),
    }


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "mode",
        "task",
        "rollout",
        "step",
        "pick_waypoint_index",
        "pick_policy",
        "pick_x_norm",
        "pick_y_norm",
        "pick_z",
        "pick_gripper",
        "pick_base_x",
        "pick_base_y",
        "pick_base_z",
        "pick_base_gripper",
        "json_path",
        "visual_path",
        "context_source",
        "image_waypoints_json",
        "base_waypoints_json",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def safe_group_name(mode: str, task: str) -> str:
    prefix = f"{mode}_" if mode else ""
    return f"{prefix}{task}".replace("/", "_").replace("\\", "_")


def summarize_group(name: str, rows: Sequence[Dict[str, Any]], decimals: int, tolerance: float) -> Dict[str, Any]:
    image_points = np.asarray([[row["pick_x_norm"], row["pick_y_norm"]] for row in rows], dtype=np.float64)
    base_points = np.asarray([[row["pick_base_x"], row["pick_base_y"], row["pick_base_z"]] for row in rows], dtype=np.float64)
    max_image = pairwise_max(image_points)
    max_base = pairwise_max(base_points)
    return {
        "group": name,
        "num_samples": len(rows),
        "num_tasks": len({row["task"] for row in rows}),
        "num_rollouts": len({(row["task"], row["rollout"]) for row in rows}),
        "unique_pick_xy_norm_rounded": rounded_unique_count(image_points, decimals),
        "std_pick_x_norm": numeric_std(image_points[:, 0]),
        "std_pick_y_norm": numeric_std(image_points[:, 1]),
        "mean_pairwise_pick_xy_norm": pairwise_mean(image_points),
        "max_pairwise_pick_xy_norm": max_image,
        "std_pick_base_x_m": numeric_std(base_points[:, 0]),
        "std_pick_base_y_m": numeric_std(base_points[:, 1]),
        "std_pick_base_z_m": numeric_std(base_points[:, 2]),
        "mean_pairwise_pick_base_xyz_m": pairwise_mean(base_points),
        "max_pairwise_pick_base_xyz_m": max_base,
        "looks_identical_by_xy_tolerance": bool(max_image is not None and max_image <= tolerance),
    }


def write_summary(output_dir: Path, rows: Sequence[Dict[str, Any]], decimals: int, tolerance: float) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    groups["overall"] = list(rows)
    for row in rows:
        groups[safe_group_name(row["mode"], row["task"])].append(row)

    summary_rows = [
        summarize_group(name, group_rows, decimals, tolerance)
        for name, group_rows in sorted(groups.items(), key=lambda item: natural_key(item[0]))
        if group_rows
    ]

    csv_path = output_dir / "step0_diversity_summary.csv"
    json_path = output_dir / "step0_diversity_summary.json"
    fieldnames = list(summary_rows[0].keys()) if summary_rows else ["group", "num_samples"]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    with json_path.open("w") as handle:
        json.dump(summary_rows, handle, indent=2)
    return summary_rows


def read_visual(row: Dict[str, Any], thumb_size: Tuple[int, int]) -> Optional[np.ndarray]:
    path_text = row.get("visual_path", "")
    if not path_text:
        return None
    image = cv2.imread(path_text, cv2.IMREAD_COLOR)
    if image is None:
        return None

    thumb_w, thumb_h = thumb_size
    scale = min(thumb_w / image.shape[1], thumb_h / image.shape[0])
    new_w = max(1, int(round(image.shape[1] * scale)))
    new_h = max(1, int(round(image.shape[0] * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((thumb_h + 42, thumb_w, 3), 245, dtype=np.uint8)
    x0 = (thumb_w - new_w) // 2
    canvas[42 : 42 + new_h, x0 : x0 + new_w] = resized

    label = (
        f"{row.get('mode') or 'eval'} {row['task']} {row['rollout']} "
        f"pick={row['pick_waypoint_index']} "
        f"x={row['pick_x_norm']:.2f} y={row['pick_y_norm']:.2f}"
    )
    cv2.putText(canvas, label[:95], (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"base=({row['pick_base_x']:.3f},{row['pick_base_y']:.3f},{row['pick_base_z']:.3f})",
        (8, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (60, 60, 60),
        1,
        cv2.LINE_AA,
    )
    return canvas


def make_montage(
    rows: Sequence[Dict[str, Any]],
    out_path: Path,
    cols: int,
    thumb_size: Tuple[int, int],
    max_images: int,
) -> Optional[Path]:
    selected = list(rows[:max_images])
    tiles = [read_visual(row, thumb_size) for row in selected]
    tiles = [tile for tile in tiles if tile is not None]
    if not tiles:
        return None

    cols = max(1, min(cols, len(tiles)))
    rows_n = int(math.ceil(len(tiles) / cols))
    tile_h, tile_w = tiles[0].shape[:2]
    canvas = np.full((rows_n * tile_h, cols * tile_w, 3), 235, dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row = index // cols
        col = index % cols
        canvas[row * tile_h : (row + 1) * tile_h, col * tile_w : (col + 1) * tile_w] = tile

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if cv2.imwrite(str(out_path), canvas):
        return out_path
    return None


def task_color(index: int) -> Tuple[int, int, int]:
    palette = [
        (230, 25, 75),
        (60, 180, 75),
        (0, 130, 200),
        (245, 130, 48),
        (145, 30, 180),
        (70, 240, 240),
        (240, 50, 230),
        (210, 245, 60),
        (250, 190, 190),
        (0, 128, 128),
        (230, 190, 255),
        (170, 110, 40),
        (255, 250, 200),
        (128, 0, 0),
        (170, 255, 195),
        (0, 0, 128),
    ]
    return palette[index % len(palette)]


def make_scatter(rows: Sequence[Dict[str, Any]], out_path: Path, title: str) -> Optional[Path]:
    valid = [
        row
        for row in rows
        if np.isfinite([row["pick_x_norm"], row["pick_y_norm"]]).all()
    ]
    if not valid:
        return None

    width, height = 900, 700
    left, right, top, bottom = 80, 30, 50, 70
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    plot_w = width - left - right
    plot_h = height - top - bottom

    cv2.rectangle(canvas, (left, top), (left + plot_w, top + plot_h), (30, 30, 30), 1)
    cv2.line(canvas, (left, top + plot_h // 2), (left + plot_w, top + plot_h // 2), (210, 210, 210), 1)
    cv2.line(canvas, (left + plot_w // 2, top), (left + plot_w // 2, top + plot_h), (210, 210, 210), 1)
    cv2.putText(canvas, title, (left, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(canvas, "x_norm [-1, 1]", (left + plot_w // 2 - 60, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(canvas, "y_norm", (10, top + plot_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)

    tasks = sorted({row["task"] for row in valid}, key=natural_key)
    task_to_color = {task: task_color(i) for i, task in enumerate(tasks)}
    for row in valid:
        x_norm = float(row["pick_x_norm"])
        y_norm = float(row["pick_y_norm"])
        x = left + int(round((x_norm + 1.0) * 0.5 * plot_w))
        y = top + int(round((1.0 - (y_norm + 1.0) * 0.5) * plot_h))
        color = task_to_color[row["task"]]
        cv2.circle(canvas, (x, y), 5, color, thickness=-1, lineType=cv2.LINE_AA)

    legend_x = left + 8
    legend_y = top + 18
    for i, task in enumerate(tasks[:18]):
        y = legend_y + i * 20
        color = task_to_color[task]
        cv2.circle(canvas, (legend_x, y - 4), 5, color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.putText(canvas, task, (legend_x + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if cv2.imwrite(str(out_path), canvas):
        return out_path
    return None


def grouped_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    groups["overall"] = list(rows)
    for row in rows:
        groups[safe_group_name(row["mode"], row["task"])].append(row)
    return dict(groups)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", required=True, help="Root output directory from eval_scripted_rollout_inference.py.")
    parser.add_argument("--output-dir", default=None, help="Defaults to <eval-root>/step0_diversity.")
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--pred-gripper-threshold", type=float, default=0.18)
    parser.add_argument(
        "--pick-policy",
        choices=["first-close", "max-gripper", "waypoint-index"],
        default="first-close",
        help="How to choose the predicted pick waypoint for diversity stats.",
    )
    parser.add_argument(
        "--pick-waypoint-index",
        type=int,
        default=0,
        help="Zero-based index used only with --pick-policy waypoint-index.",
    )
    parser.add_argument("--round-decimals", type=int, default=4)
    parser.add_argument("--identical-tolerance", type=float, default=1e-4)
    parser.add_argument("--montage-cols", type=int, default=4)
    parser.add_argument("--thumb-width", type=int, default=420)
    parser.add_argument("--thumb-height", type=int, default=300)
    parser.add_argument("--max-images-per-montage", type=int, default=200)
    parser.add_argument("--no-task-montages", action="store_true")
    args = parser.parse_args()

    eval_root = Path(args.eval_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else eval_root / "step0_diversity"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_paths = find_step_jsons(eval_root, args.step)
    rows = []
    skipped = []
    for json_path in json_paths:
        try:
            row = record_from_json(json_path, eval_root, args)
        except Exception as exc:
            skipped.append({"json_path": str(json_path), "reason": f"{type(exc).__name__}: {exc}"})
            continue
        if row is None:
            skipped.append({"json_path": str(json_path), "reason": "missing or invalid image_waypoints"})
            continue
        rows.append(row)

    rows.sort(key=lambda row: natural_key((row["mode"], row["task"], row["rollout"], row["json_path"])))
    write_csv(output_dir / f"step{args.step:03d}_waypoint_diversity.csv", rows)
    with (output_dir / f"step{args.step:03d}_skipped.jsonl").open("w") as handle:
        for item in skipped:
            handle.write(json.dumps(item) + "\n")

    if not rows:
        raise SystemExit(f"No valid osvi_waypoints_t{args.step:03d}.json files found under {eval_root}")

    summary_rows = write_summary(output_dir, rows, args.round_decimals, args.identical_tolerance)
    groups = grouped_rows(rows)
    thumb_size = (args.thumb_width, args.thumb_height)

    montage_paths = []
    overall_montage = make_montage(
        groups["overall"],
        output_dir / f"montage_step{args.step:03d}_overall.png",
        args.montage_cols,
        thumb_size,
        args.max_images_per_montage,
    )
    if overall_montage is not None:
        montage_paths.append(str(overall_montage))

    if not args.no_task_montages:
        for group_name, group_rows in sorted(groups.items(), key=lambda item: natural_key(item[0])):
            if group_name == "overall":
                continue
            montage = make_montage(
                group_rows,
                output_dir / f"montage_step{args.step:03d}_{group_name}.png",
                args.montage_cols,
                thumb_size,
                args.max_images_per_montage,
            )
            if montage is not None:
                montage_paths.append(str(montage))

    scatter_path = make_scatter(rows, output_dir / f"scatter_step{args.step:03d}_pick_xy_norm.png", f"Step {args.step} predicted pick xy")

    report = {
        "eval_root": str(eval_root),
        "output_dir": str(output_dir),
        "step": args.step,
        "num_json_files": len(json_paths),
        "num_valid_records": len(rows),
        "num_skipped": len(skipped),
        "csv": str(output_dir / f"step{args.step:03d}_waypoint_diversity.csv"),
        "summary_csv": str(output_dir / "step0_diversity_summary.csv"),
        "summary_json": str(output_dir / "step0_diversity_summary.json"),
        "scatter_png": None if scatter_path is None else str(scatter_path),
        "montages": montage_paths,
        "overall": next((row for row in summary_rows if row["group"] == "overall"), None),
    }
    with (output_dir / f"step{args.step:03d}_diversity_report.json").open("w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
