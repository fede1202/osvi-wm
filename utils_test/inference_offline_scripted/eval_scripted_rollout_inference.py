#!/usr/bin/env python3
"""Run the existing OSVIController inference code on scripted rollouts.

The only adaptation is the input source: instead of waiting for ROS camera
messages, this script reads saved agent frames from script_controller_node
Trajectory .pkl files and passes them unchanged to OSVIController.inference().

Everything downstream of that call is the existing OSVI runtime path:
load_command() builds the context, pre_process() prepares the agent image,
inference() runs the model, projects waypoints, post_process() builds robot
actions, and OSVI's own JSON/overlay files are written in each step directory.

Image color/order handling is intentionally controlled only by the model
configuration consumed by OSVIController, especially image.live_images_are_rgb.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import pickle
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml


SCRIPTED_FRONT_IMAGE_KEYS = ("camera_front_image", "front_camera_image")
ROBOT_STATE_KEYS = (
    "eef_pos",
    "eef_quat",
    "joint_pos",
    "joint_vel",
    "gripper_qpos",
    "gripper_qvel",
)
SKIPPED_ROLLOUT_FIELDNAMES = (
    "rollout_path",
    "task_id",
    "num_rollout_steps",
    "reason",
)
TEST_CONFIG_MANIFEST = "test_config_manifest.json"


class RolloutSkipped(ValueError):
    def __init__(self, reason: str, num_steps: Optional[int] = None):
        super().__init__(reason)
        self.reason = reason
        self.num_steps = num_steps


def natural_key(path: Path) -> List[Any]:
    return [int(chunk) if chunk.isdigit() else chunk for chunk in re.split(r"(\d+)", str(path))]


def resolve_path(value: str) -> Path:
    return Path(os.path.expanduser(value)).resolve()


def load_yaml_mapping(path: Path) -> Dict[str, Any]:
    with path.open("r") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return data


def resolve_config_path(config_dir: Path, value: str) -> Path:
    path = Path(os.path.expanduser(str(value)))
    if not path.is_absolute():
        path = config_dir / path
    return path.resolve()


def as_output_relative(path: Path, output_dir: Path) -> str:
    try:
        return path.relative_to(output_dir).as_posix()
    except ValueError:
        return str(path)


def unique_copy_path(directory: Path, filename: str, used_names: set) -> Path:
    candidate = directory / filename
    stem = candidate.stem
    suffix = candidate.suffix
    counter = 2
    while candidate.name in used_names:
        candidate = directory / f"{stem}_{counter}{suffix}"
        counter += 1
    used_names.add(candidate.name)
    return candidate


def copy_osvi_test_configs(
    model_config_path: str,
    output_dir: Path,
    checkpoint_override: Optional[str],
) -> Tuple[Path, Dict[str, Any]]:
    """Stage the OSVI runtime YAML and its YAML dependencies inside output_dir."""
    source_runtime_config = resolve_path(model_config_path)
    if not source_runtime_config.is_file():
        raise FileNotFoundError(f"OSVI runtime config not found: {source_runtime_config}")

    source_config_dir = source_runtime_config.parent
    runtime_config = load_yaml_mapping(source_runtime_config)
    local_config_dir = output_dir / "configs"
    local_config_dir.mkdir(parents=True, exist_ok=True)

    copied_configs: Dict[str, Dict[str, str]] = {}
    used_names = set()
    for key, default_value in (
        ("training_config_path", "configs/training_config.yaml"),
        ("projection_matrix_path", "configs/ur5e_zed_front_projection.yaml"),
    ):
        source_path = resolve_config_path(source_config_dir, runtime_config.get(key, default_value))
        if not source_path.is_file():
            raise FileNotFoundError(f"{key} referenced by {source_runtime_config} was not found: {source_path}")
        dest_path = unique_copy_path(local_config_dir, source_path.name, used_names)
        if dest_path.resolve() != source_path.resolve():
            shutil.copy2(source_path, dest_path)
        runtime_config[key] = as_output_relative(dest_path, output_dir)
        copied_configs[key] = {
            "source": str(source_path),
            "copy": str(dest_path),
        }

    if checkpoint_override:
        runtime_config["checkpoint_path"] = str(resolve_path(checkpoint_override))
    elif runtime_config.get("checkpoint_path"):
        runtime_config["checkpoint_path"] = str(resolve_config_path(source_config_dir, runtime_config["checkpoint_path"]))

    local_runtime_config = output_dir / source_runtime_config.name
    if local_runtime_config.resolve() == source_runtime_config.resolve():
        local_runtime_config = output_dir / f"{source_runtime_config.stem}_local{source_runtime_config.suffix}"
    with local_runtime_config.open("w") as handle:
        yaml.safe_dump(runtime_config, handle, sort_keys=False)

    manifest = {
        "runtime_config": {
            "source": str(source_runtime_config),
            "copy": str(local_runtime_config),
        },
        "referenced_configs": copied_configs,
        "checkpoint_path": runtime_config.get("checkpoint_path"),
    }
    with (output_dir / TEST_CONFIG_MANIFEST).open("w") as handle:
        json.dump(manifest, handle, indent=2)
    return local_runtime_config, manifest


def add_ai_controller_to_path(root_arg: str) -> Path:
    """Add the source dir that exposes the ai_controller Python package."""
    root = resolve_path(root_arg)
    candidates = [
        root,
        root / "ai_controller",
        root / "UR5e-2f-85" / "ai_controller",
    ]
    for candidate in candidates:
        controller_file = candidate / "ai_controller" / "models" / "osvi_controller" / "osvi_controller.py"
        if controller_file.is_file():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return candidate
    raise FileNotFoundError(
        "Could not find ai_controller.models.osvi_controller.osvi_controller.py. "
        f"Checked under {root}."
    )


def add_dataset_collector_savers_to_path(root_arg: Optional[str]) -> Optional[Path]:
    """Make savers.Trajectory importable for pickle loading."""
    candidates: List[Path] = []
    if root_arg:
        root = resolve_path(root_arg)
        candidates.extend(
            [
                root,
                root / "scripts",
                root / "dataset_collector_pkg" / "scripts",
                root / "dataset_collector" / "dataset_collector_pkg" / "scripts",
            ]
        )

    env_root = os.environ.get("DATASET_COLLECTOR_ROOT")
    if env_root:
        root = resolve_path(env_root)
        candidates.extend([root, root / "scripts", root / "dataset_collector_pkg" / "scripts"])

    for candidate in candidates:
        if (candidate / "savers.py").is_file():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return candidate
    return None


def import_existing_osvi_controller(ai_controller_root: str, dataset_collector_root: Optional[str]):
    add_ai_controller_to_path(ai_controller_root)
    add_dataset_collector_savers_to_path(dataset_collector_root)

    from ai_controller.models.osvi_controller.osvi_controller import (  # noqa: PLC0415
        OSVIController,
        TrajectoryUnpickler,
    )

    return OSVIController, TrajectoryUnpickler


def expand_rollouts(specs: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for spec in specs:
        expanded = os.path.expanduser(spec)
        if any(char in expanded for char in "*?["):
            paths.extend(resolve_path(path) for path in glob.glob(expanded, recursive=True))
            continue

        path = resolve_path(expanded)
        if path.is_dir():
            paths.extend(sorted(path.rglob("*.pkl"), key=natural_key))
        else:
            paths.append(path)

    unique: List[Path] = []
    seen = set()
    for path in paths:
        if path.suffix == ".pkl" and path not in seen:
            unique.append(path)
            seen.add(path)
    return sorted(unique, key=natural_key)


def load_rollout_steps(path: Path, trajectory_unpickler_cls) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Rollout file not found: {path}")
    with path.open("rb") as handle:
        payload = trajectory_unpickler_cls(handle).load()

    trajectory = payload.get("traj", payload) if isinstance(payload, dict) else payload
    if hasattr(trajectory, "get"):
        return [trajectory.get(index) for index in range(len(trajectory))]
    return [trajectory[index] for index in range(len(trajectory))]


def infer_task_id(path: Path, explicit_task_id: Optional[str]) -> str:
    if explicit_task_id:
        return normalize_task_id(explicit_task_id)

    for part in reversed(path.parts):
        match = re.fullmatch(r"task_?(\d+)", part)
        if match:
            return f"task_{int(match.group(1)):02d}"
    raise ValueError(f"Could not infer task id from {path}. Pass --task-id.")


def normalize_task_id(task_id: str) -> str:
    task_id = str(task_id)
    if task_id.startswith("task_"):
        return task_id
    return f"task_{int(task_id):02d}"


def get_step_obs(step: Dict[str, Any]) -> Dict[str, Any]:
    obs = step.get("obs") if isinstance(step, dict) else None
    if not isinstance(obs, dict):
        raise ValueError("Rollout step has no obs dict.")
    return obs


def get_obs_image(obs: Dict[str, Any], image_key: Optional[str]) -> Any:
    keys = (image_key,) if image_key else SCRIPTED_FRONT_IMAGE_KEYS
    for key in keys:
        if key and key in obs:
            return obs[key]
    raise KeyError(f"None of image keys {keys} found. Available keys: {sorted(obs.keys())}")


def extract_action(step: Dict[str, Any]) -> Optional[np.ndarray]:
    action = step.get("action") if isinstance(step, dict) else None
    if action is None:
        return None
    action = np.asarray(action, dtype=np.float64).reshape(-1)
    if action.size < 8:
        return None
    return action[:8]


def future_scripted_actions(
    steps: Sequence[Dict[str, Any]],
    start_step: int,
    count: int,
) -> Optional[np.ndarray]:
    actions = [extract_action(step) for step in steps[start_step:]]
    actions = [action for action in actions if action is not None]
    if not actions:
        return None
    indices = np.linspace(0, len(actions) - 1, num=count, endpoint=True, dtype=int)
    return np.stack([actions[int(index)] for index in indices], axis=0)


def future_scripted_positions(steps: Sequence[Dict[str, Any]], start_step: int) -> Optional[np.ndarray]:
    actions = [extract_action(step) for step in steps[start_step:]]
    positions = [action[:3] for action in actions if action is not None]
    if not positions:
        return None
    return np.stack(positions, axis=0)


def nearest_distances(pred_xyz: np.ndarray, future_xyz: np.ndarray) -> np.ndarray:
    deltas = pred_xyz[:, None, :] - future_xyz[None, :, :]
    return np.linalg.norm(deltas, axis=-1).min(axis=1)


def compare_predictions(
    osvi_payload: Dict[str, Any],
    steps: Sequence[Dict[str, Any]],
    step_index: int,
    pred_gripper_threshold: float,
    scripted_gripper_threshold: float,
) -> Dict[str, Optional[float]]:
    base_waypoints = np.asarray(osvi_payload.get("base_waypoints", []), dtype=np.float64)
    image_waypoints = np.asarray(osvi_payload.get("image_waypoints", []), dtype=np.float64)
    if base_waypoints.size == 0:
        return empty_metrics()

    pred_xyz = base_waypoints[:, :3]
    future_xyz = future_scripted_positions(steps, step_index)
    sampled_future = future_scripted_actions(steps, step_index, len(base_waypoints))

    metrics = empty_metrics()
    if future_xyz is not None:
        nearest = nearest_distances(pred_xyz, future_xyz)
        metrics["mean_nearest_l2_m"] = float(np.mean(nearest))
        metrics["max_nearest_l2_m"] = float(np.max(nearest))
        metrics["final_to_final_l2_m"] = float(np.linalg.norm(pred_xyz[-1] - future_xyz[-1]))

    if sampled_future is not None:
        count = min(len(pred_xyz), len(sampled_future))
        sampled_l2 = np.linalg.norm(pred_xyz[:count] - sampled_future[:count, :3], axis=1)
        metrics["mean_sampled_l2_m"] = float(np.mean(sampled_l2))
        metrics["first_sampled_l2_m"] = float(sampled_l2[0])
        metrics["last_sampled_l2_m"] = float(sampled_l2[-1])

        if image_waypoints.size:
            pred_closed = image_waypoints[:count, 3] >= pred_gripper_threshold
            scripted_closed = sampled_future[:count, 7] >= scripted_gripper_threshold
            metrics["sampled_gripper_accuracy"] = float(np.mean(pred_closed == scripted_closed))

    return metrics


def empty_metrics() -> Dict[str, Optional[float]]:
    return {
        "mean_nearest_l2_m": None,
        "max_nearest_l2_m": None,
        "final_to_final_l2_m": None,
        "mean_sampled_l2_m": None,
        "first_sampled_l2_m": None,
        "last_sampled_l2_m": None,
        "sampled_gripper_accuracy": None,
    }


def load_osvi_step_payload(step_dir: Path, t: int) -> Dict[str, Any]:
    path = step_dir / f"osvi_waypoints_t{t:03d}.json"
    if not path.is_file():
        raise FileNotFoundError(
            "OSVIController.inference() did not write the expected waypoint JSON: "
            f"{path}"
        )
    with path.open("r") as handle:
        return json.load(handle)


def effective_min_traj_len(args, controller) -> int:
    if args.min_traj_len is not None:
        return int(args.min_traj_len)
    training_cfg = getattr(controller, "training_config", {}) or {}
    data_cfg = training_cfg.get("data", {}) if isinstance(training_cfg, dict) else {}
    return int(data_cfg.get("min_traj_len", 1))


def rollout_step_indices(steps: Sequence[Dict[str, Any]], args) -> List[int]:
    step_indices = list(range(0, len(steps), max(1, int(args.stride))))
    if args.max_steps is not None:
        step_indices = step_indices[: int(args.max_steps)]
    return step_indices


def validate_rollout_steps(
    steps: Sequence[Dict[str, Any]],
    args,
    min_traj_len: int,
) -> None:
    if len(steps) < min_traj_len:
        raise RolloutSkipped(
            f"num_steps={len(steps)} is below min_traj_len={min_traj_len}",
            num_steps=len(steps),
        )

    step_indices = rollout_step_indices(steps, args)
    if not step_indices:
        raise RolloutSkipped("no step selected for evaluation", num_steps=len(steps))

    for step_index in step_indices:
        try:
            obs = get_step_obs(steps[step_index])
            get_obs_image(obs, args.rollout_image_key)
        except Exception as exc:
            raise RolloutSkipped(
                f"invalid step {step_index}: {type(exc).__name__}: {exc}",
                num_steps=len(steps),
            ) from exc


def controller_projection_matrix_np(controller) -> np.ndarray:
    projection = controller.projection_matrix
    if hasattr(projection, "detach"):
        projection = projection.detach().cpu().numpy()
    return np.asarray(projection, dtype=np.float64)


def image_waypoints_to_pixels(image_waypoints: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
    height, width = image_shape
    points = np.asarray(image_waypoints, dtype=np.float64)
    pixels = np.full((len(points), 2), np.nan, dtype=np.float64)
    if points.size == 0:
        return pixels
    pixels[:, 0] = (points[:, 0] + 1.0) * 0.5 * (width - 1)
    pixels[:, 1] = (1.0 - points[:, 1]) * 0.5 * (height - 1)
    return pixels


def base_actions_to_image_waypoints(
    actions: np.ndarray,
    projection_matrix: np.ndarray,
) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float64)
    base_h = np.concatenate(
        [actions[:, :3], np.ones((len(actions), 1), dtype=np.float64)],
        axis=1,
    )
    image_h = np.einsum("dh,nh->nd", np.linalg.inv(projection_matrix), base_h)
    z = image_h[:, 2]
    valid_z = np.abs(z) > 1e-8

    image_waypoints = np.full((len(actions), 4), np.nan, dtype=np.float64)
    image_waypoints[valid_z, 0] = image_h[valid_z, 0] / z[valid_z]
    image_waypoints[valid_z, 1] = image_h[valid_z, 1] / z[valid_z]
    image_waypoints[valid_z, 2] = z[valid_z]
    image_waypoints[:, 3] = actions[:, 7]
    return image_waypoints


def draw_ground_truth_overlay(
    controller,
    saved_frame: Any,
    osvi_payload: Dict[str, Any],
    steps: Sequence[Dict[str, Any]],
    step_index: int,
    step_dir: Path,
    pred_gripper_threshold: float,
    scripted_gripper_threshold: float,
) -> Optional[Path]:
    image_waypoints = np.asarray(osvi_payload.get("image_waypoints", []), dtype=np.float64)
    if image_waypoints.size == 0:
        return None

    gt_actions = future_scripted_actions(steps, step_index, len(image_waypoints))
    if gt_actions is None:
        return None

    rgb_frame = controller._prepare_live_frame(saved_frame)
    overlay = controller._draw_waypoint_overlay(rgb_frame, image_waypoints)
    height, width = overlay.shape[:2]

    gt_image_waypoints = base_actions_to_image_waypoints(
        gt_actions,
        controller_projection_matrix_np(controller),
    )
    gt_pixels = image_waypoints_to_pixels(gt_image_waypoints, (height, width))
    finite = np.isfinite(gt_pixels).all(axis=1)
    in_bounds = (
        finite
        & (gt_pixels[:, 0] >= 0)
        & (gt_pixels[:, 0] < width)
        & (gt_pixels[:, 1] >= 0)
        & (gt_pixels[:, 1] < height)
    )

    if in_bounds.sum() >= 2:
        pts = np.round(gt_pixels[in_bounds]).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(overlay, [pts], isClosed=False, color=(80, 255, 80), thickness=2, lineType=cv2.LINE_AA)

    pred_closed = image_waypoints[:, 3] >= pred_gripper_threshold
    gt_closed = gt_actions[:, 7] >= scripted_gripper_threshold
    for idx, (x, y) in enumerate(gt_pixels):
        if not in_bounds[idx]:
            continue
        center = (int(round(x)), int(round(y)))
        color = (80, 255, 80) if gt_closed[idx] else (255, 220, 40)
        radius = 8 if gt_closed[idx] else 6
        thickness = 3 if bool(pred_closed[idx]) != bool(gt_closed[idx]) else 2
        cv2.circle(overlay, center, radius, color, thickness=thickness, lineType=cv2.LINE_AA)
        cv2.circle(overlay, center, radius + 3, (20, 20, 20), thickness=1, lineType=cv2.LINE_AA)
        cv2.putText(
            overlay,
            f"GT{idx + 1}",
            (min(center[0] + 10, width - 1), max(center[1] + 14, 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        overlay,
        "pred: filled OSVI markers | GT: hollow markers/green path",
        (8, height - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    out_path = step_dir / f"osvi_waypoints_gt_overlay_t{step_index:03d}.png"
    step_dir.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)):
        raise IOError(f"Could not write ground-truth overlay: {out_path}")
    return out_path


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "rollout_path",
        "task_id",
        "step",
        "num_rollout_steps",
        "num_pred_actions",
        "mean_nearest_l2_m",
        "max_nearest_l2_m",
        "final_to_final_l2_m",
        "mean_sampled_l2_m",
        "first_sampled_l2_m",
        "last_sampled_l2_m",
        "sampled_gripper_accuracy",
        "osvi_waypoint_json",
        "gt_overlay_png",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_skipped_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SKIPPED_ROLLOUT_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in SKIPPED_ROLLOUT_FIELDNAMES})


def numeric_mean(rows: Sequence[Dict[str, Any]], key: str) -> Optional[float]:
    values = [row.get(key) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def disable_debug_window(controller) -> None:
    if getattr(controller, "cfg", None) is not None:
        controller.cfg.debug["show_waypoint_overlay"] = False


def evaluate_rollout(
    controller,
    trajectory_unpickler_cls,
    rollout_path: Path,
    rollout_index: int,
    args,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    steps = load_rollout_steps(rollout_path, trajectory_unpickler_cls)
    task_id = infer_task_id(rollout_path, args.task_id)
    task_number_for_osvi = task_id.replace("task_", "")
    min_traj_len = effective_min_traj_len(args, controller)
    validate_rollout_steps(steps, args, min_traj_len)

    controller.reset()
    controller.load_command(
        args.demo_dir,
        task_number_for_osvi,
        save_demo_frames=args.save_context_frames,
        traj_cnt=rollout_index,
        save_path=args.output_dir,
    )

    step_indices = rollout_step_indices(steps, args)

    rollout_out_dir = Path(args.output_dir) / task_id / rollout_path.stem
    records: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []

    pred_gripper_threshold = float(
        args.pred_gripper_threshold
        if args.pred_gripper_threshold is not None
        else controller.cfg.control.get("gripper_close_threshold", 0.18)
    )

    for step_index in step_indices:
        step = steps[step_index]
        obs = get_step_obs(step)
        missing_state = [key for key in ROBOT_STATE_KEYS if key not in obs]

        saved_frame = get_obs_image(obs, args.rollout_image_key)
        step_dir = rollout_out_dir / f"step_{step_index:03d}"

        actions = controller.inference(
            input_data=[[saved_frame], obs],
            t=step_index,
            save_path=str(step_dir),
        )
        osvi_payload = load_osvi_step_payload(step_dir, step_index)
        metrics = compare_predictions(
            osvi_payload,
            steps,
            step_index,
            pred_gripper_threshold=pred_gripper_threshold,
            scripted_gripper_threshold=float(args.scripted_gripper_threshold),
        )

        osvi_json_path = step_dir / f"osvi_waypoints_t{step_index:03d}.json"
        gt_overlay_path = None
        if args.save_gt_overlay:
            gt_overlay_path = draw_ground_truth_overlay(
                controller,
                saved_frame,
                osvi_payload,
                steps,
                step_index,
                step_dir,
                pred_gripper_threshold=pred_gripper_threshold,
                scripted_gripper_threshold=float(args.scripted_gripper_threshold),
            )
        record = {
            "rollout_path": str(rollout_path),
            "task_id": task_id,
            "step": step_index,
            "num_rollout_steps": len(steps),
            "missing_robot_state_keys": missing_state,
            "config_live_images_are_rgb": bool(controller.cfg.image.get("live_images_are_rgb", True)),
            "returned_actions": [np.asarray(action, dtype=np.float64).tolist() for action in actions],
            "osvi_waypoint_json": str(osvi_json_path),
            "gt_overlay_png": str(gt_overlay_path) if gt_overlay_path is not None else None,
            "metrics": metrics,
        }
        records.append(record)

        row = {
            "rollout_path": str(rollout_path),
            "task_id": task_id,
            "step": step_index,
            "num_rollout_steps": len(steps),
            "num_pred_actions": len(actions),
            "osvi_waypoint_json": str(osvi_json_path),
            "gt_overlay_png": str(gt_overlay_path) if gt_overlay_path is not None else None,
        }
        row.update(metrics)
        rows.append(row)

        print(
            "[eval] "
            f"{rollout_path.name} step={step_index} "
            f"nearest={metrics['mean_nearest_l2_m']} "
            f"sampled={metrics['mean_sampled_l2_m']}"
        )

    return records, rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ai-controller-root",
        default="../UR5e-2f-85/ai_controller",
        help="Path that contains the ai_controller Python package.",
    )
    parser.add_argument(
        "--dataset-collector-root",
        help="Path to dataset_collector_pkg or its scripts dir, so savers.py is importable.",
    )
    parser.add_argument(
        "--config",
        "--model-config-path",
        dest="model_config_path",
        required=True,
        help=(
            "Existing OSVI config passed to OSVIController, e.g. osvi_config.yaml. "
            "Image color/order is controlled there via image.live_images_are_rgb."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        help=(
            "Optional checkpoint override written into the local copied OSVI config. "
            "Useful for keeping one result folder per test, like overlay_ur5e_waypoints.py."
        ),
    )
    parser.add_argument(
        "--demo-dir",
        required=True,
        help="Existing expert demo base dir passed unchanged to OSVIController.load_command().",
    )
    parser.add_argument(
        "--rollout",
        action="append",
        default=[],
        help="Scripted rollout .pkl, directory, or glob. Can be repeated.",
    )
    parser.add_argument("--task-id", help="Override task id, e.g. 1 or task_01.")
    parser.add_argument("--output-dir", default="scripted_osvi_existing_eval")
    parser.add_argument("--max-rollouts", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--rollout-image-key", help="Override front-image key in scripted rollout obs.")
    parser.add_argument(
        "--min-traj-len",
        type=int,
        default=None,
        help="Minimum rollout length to evaluate. Defaults to data.min_traj_len from the training config.",
    )
    parser.add_argument(
        "--skip-invalid-rollouts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip empty/invalid rollout PKLs and write skipped_rollouts reports instead of aborting.",
    )
    parser.add_argument(
        "--save-gt-overlay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save an extra overlay with OSVI prediction plus sampled scripted ground truth.",
    )
    parser.add_argument(
        "--disable-debug-window",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable only cv2.imshow debug display; OSVI preprocessing/inference/postprocess are unchanged.",
    )
    parser.add_argument(
        "--copy-configs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Copy the OSVI runtime config plus its training/projection YAML dependencies into "
            "--output-dir and run from the copied runtime config."
        ),
    )
    parser.add_argument("--save-context-frames", action="store_true")
    parser.add_argument("--scripted-gripper-threshold", type=float, default=0.5)
    parser.add_argument("--pred-gripper-threshold", type=float)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.rollout:
        raise ValueError("Pass at least one --rollout file, directory, or glob.")

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(output_dir)
    config_manifest = None

    if args.checkpoint and not args.copy_configs:
        raise ValueError("--checkpoint can only be used when --copy-configs is enabled.")
    if args.copy_configs:
        local_config_path, config_manifest = copy_osvi_test_configs(
            args.model_config_path,
            output_dir,
            args.checkpoint,
        )
        args.model_config_path = str(local_config_path)
        print(f"[eval] using local copied OSVI config: {local_config_path}")

    OSVIController, TrajectoryUnpickler = import_existing_osvi_controller(
        args.ai_controller_root,
        args.dataset_collector_root,
    )
    controller = OSVIController(args.model_config_path, task_name="pick_place")
    if args.disable_debug_window:
        disable_debug_window(controller)

    rollouts = expand_rollouts(args.rollout)
    if args.max_rollouts is not None:
        rollouts = rollouts[: int(args.max_rollouts)]
    if not rollouts:
        raise FileNotFoundError("No scripted rollout .pkl files matched --rollout.")

    all_records: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []
    skipped_rollouts: List[Dict[str, Any]] = []
    for rollout_index, rollout_path in enumerate(rollouts):
        print(f"[eval] running exact OSVIController inference on {rollout_path}")
        try:
            records, rows = evaluate_rollout(
                controller,
                TrajectoryUnpickler,
                rollout_path,
                rollout_index,
                args,
            )
        except RolloutSkipped as exc:
            task_id = infer_task_id(rollout_path, args.task_id)
            skipped_rollouts.append(
                {
                    "rollout_path": str(rollout_path),
                    "task_id": task_id,
                    "num_rollout_steps": exc.num_steps,
                    "reason": exc.reason,
                }
            )
            print(f"[eval] skipped {rollout_path.name}: {exc.reason}")
            continue
        except Exception as exc:
            if not args.skip_invalid_rollouts:
                raise
            task_id = infer_task_id(rollout_path, args.task_id)
            skipped_rollouts.append(
                {
                    "rollout_path": str(rollout_path),
                    "task_id": task_id,
                    "num_rollout_steps": None,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"[eval] skipped {rollout_path.name}: {type(exc).__name__}: {exc}")
            continue
        all_records.extend(records)
        all_rows.extend(rows)

    write_jsonl(output_dir / "records.jsonl", all_records)
    write_csv(output_dir / "summary.csv", all_rows)
    write_jsonl(output_dir / "skipped_rollouts.jsonl", skipped_rollouts)
    write_skipped_csv(output_dir / "skipped_rollouts.csv", skipped_rollouts)

    summary = {
        "num_rollouts": len(rollouts),
        "num_rollouts_evaluated": len({row["rollout_path"] for row in all_rows}),
        "num_rollouts_skipped": len(skipped_rollouts),
        "num_inferences": len(all_rows),
        "mean_nearest_l2_m": numeric_mean(all_rows, "mean_nearest_l2_m"),
        "mean_sampled_l2_m": numeric_mean(all_rows, "mean_sampled_l2_m"),
        "sampled_gripper_accuracy": numeric_mean(all_rows, "sampled_gripper_accuracy"),
        "records_jsonl": str(output_dir / "records.jsonl"),
        "summary_csv": str(output_dir / "summary.csv"),
        "skipped_rollouts_jsonl": str(output_dir / "skipped_rollouts.jsonl"),
        "skipped_rollouts_csv": str(output_dir / "skipped_rollouts.csv"),
        "model_config_path": args.model_config_path,
        "config_manifest": str(output_dir / TEST_CONFIG_MANIFEST) if config_manifest is not None else None,
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
