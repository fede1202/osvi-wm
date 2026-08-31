import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data._utils.collate import default_collate

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.agent_teacher_dataset import AgentTeacherDataset
from overlay_ur5e_waypoints import (
    gripper_states,
    gripper_transitions,
    load_model,
    select_policy_waypoints,
    unpack_batch,
    unwrap_dataset,
)


def image_waypoints_to_base_batch(waypoints, projection_matrix):
    hom_im_coords = np.concatenate(
        [waypoints[..., :2] * waypoints[..., 2:3], waypoints[..., 2:3]],
        axis=-1,
    )
    hom_im_4d = np.concatenate(
        [hom_im_coords, np.ones((*hom_im_coords.shape[:-1], 1), dtype=np.float32)],
        axis=-1,
    )
    trans_waypoints = np.einsum("bij,bwj->bwi", projection_matrix, hom_im_4d)
    return np.concatenate([trans_waypoints[..., :3], waypoints[..., 3:4]], axis=-1)


def resample_sequence(seq, target_len):
    seq = np.asarray(seq, dtype=np.float32)
    if len(seq) == target_len:
        return seq
    src_t = np.linspace(0.0, 1.0, len(seq), dtype=np.float32)
    dst_t = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
    out = np.empty((target_len, seq.shape[-1]), dtype=np.float32)
    for dim in range(seq.shape[-1]):
        out[:, dim] = np.interp(dst_t, src_t, seq[:, dim])
    return out


def interpolate_waypoint_path(start_point, waypoints, num_points):
    waypoints = np.asarray(waypoints, dtype=np.float32)
    prev = np.asarray(start_point, dtype=np.float32)
    segments = []
    per_segment = max(1, num_points // len(waypoints))
    for i, waypoint in enumerate(waypoints):
        endpoint = i == len(waypoints) - 1
        alphas = np.linspace(0.0, 1.0, per_segment, endpoint=endpoint, dtype=np.float32)
        segment = prev[None, :] + alphas[:, None] * (waypoint[None, :] - prev[None, :])
        segment[:, -1] = waypoint[-1]
        segments.append(segment)
        prev = waypoint
    path = np.concatenate(segments, axis=0)
    if len(path) != num_points:
        path = resample_sequence(path, num_points)
    return path


def mean_min_distance(points, reference, dims):
    points = points[:, dims]
    reference = reference[:, dims]
    dists = np.linalg.norm(points[:, None, :] - reference[None, :, :], axis=-1)
    mins = dists.min(axis=1)
    return float(mins.mean()), float(mins.max())


def dtw_distance(a, b, dims):
    a = a[:, dims]
    b = b[:, dims]
    costs = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    dp = np.full((len(a) + 1, len(b) + 1), np.inf, dtype=np.float32)
    dp[0, 0] = 0.0
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dp[i, j] = costs[i - 1, j - 1] + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[len(a), len(b)] / max(len(a), len(b)))


def first_transition(transitions, state, start_after=-1):
    for transition in transitions:
        if transition["state"] == state and transition["index"] > start_after:
            return transition
    return None


def parse_xyz_series(value):
    if not value:
        return []
    return [
        [float(component) for component in point.split(",")]
        for point in value.split(";")
        if point
    ]


def metric_row(
    *,
    meta,
    pred_waypoints_base,
    gt_points,
    gripper_threshold,
    interp_points,
    ade_xy_threshold,
    waypoint_xy_threshold,
):
    pred_path = interpolate_waypoint_path(gt_points[0], pred_waypoints_base, interp_points)
    gt_resampled = resample_sequence(gt_points, interp_points)

    diff_xy = np.linalg.norm(pred_path[:, :2] - gt_resampled[:, :2], axis=-1)
    diff_xyz = np.linalg.norm(pred_path[:, :3] - gt_resampled[:, :3], axis=-1)
    wp_min_xy_mean, wp_min_xy_max = mean_min_distance(pred_waypoints_base, gt_points, [0, 1])
    wp_min_xyz_mean, wp_min_xyz_max = mean_min_distance(pred_waypoints_base, gt_points, [0, 1, 2])

    pred_values = pred_waypoints_base[:, 3]
    pred_states = gripper_states(pred_values, gripper_threshold)
    pred_transitions = gripper_transitions(pred_values, gripper_threshold)
    gt_transitions = gripper_transitions(gt_points[:, 3], gripper_threshold)

    pred_close_wp = next((i for i, state in enumerate(pred_states) if state == "close"), None)
    pred_release_wp = None
    if pred_close_wp is not None:
        pred_release_wp = next(
            (i for i, state in enumerate(pred_states[pred_close_wp + 1 :], start=pred_close_wp + 1) if state == "open"),
            None,
        )

    gt_close = first_transition(gt_transitions, "close")
    gt_release = first_transition(gt_transitions, "open", start_after=gt_close["index"] if gt_close else -1)

    pick_xy_error = ""
    if pred_close_wp is not None and gt_close is not None:
        pick_xy_error = float(np.linalg.norm(pred_waypoints_base[pred_close_wp, :2] - gt_points[gt_close["index"], :2]))

    place_xy_error = ""
    if pred_release_wp is not None and gt_release is not None:
        place_xy_error = float(np.linalg.norm(pred_waypoints_base[pred_release_wp, :2] - gt_points[gt_release["index"], :2]))

    row = {
        **meta,
        "ade_xy_m": float(diff_xy.mean()),
        "ade_xyz_m": float(diff_xyz.mean()),
        "fde_xy_m": float(diff_xy[-1]),
        "fde_xyz_m": float(diff_xyz[-1]),
        "max_xy_m": float(diff_xy.max()),
        "max_xyz_m": float(diff_xyz.max()),
        "dtw_xy_m": dtw_distance(pred_path, gt_resampled, [0, 1]),
        "dtw_xyz_m": dtw_distance(pred_path, gt_resampled, [0, 1, 2]),
        "waypoint_min_xy_mean_m": wp_min_xy_mean,
        "waypoint_min_xy_max_m": wp_min_xy_max,
        "waypoint_min_xyz_mean_m": wp_min_xyz_mean,
        "waypoint_min_xyz_max_m": wp_min_xyz_max,
        "path_success": int(float(diff_xy.mean()) <= ade_xy_threshold),
        "waypoints_on_gt_success": int(wp_min_xy_mean <= waypoint_xy_threshold),
        "pred_gripper_values": ";".join(f"{float(v):.6f}" for v in pred_values),
        "pred_gripper_states": ";".join(pred_states),
        "pred_gripper_transitions": ";".join(
            f"{t['index']}:{t['state']}:{t['value']:.6f}" for t in pred_transitions
        ),
        "gt_gripper_transitions": ";".join(
            f"{t['index']}:{t['state']}:{t['value']:.6f}" for t in gt_transitions
        ),
        "pred_has_close": int(pred_close_wp is not None),
        "pred_has_release_after_close": int(pred_release_wp is not None),
        "pred_close_waypoint": "" if pred_close_wp is None else pred_close_wp + 1,
        "pred_release_waypoint": "" if pred_release_wp is None else pred_release_wp + 1,
        "gt_close_index": "" if gt_close is None else gt_close["index"],
        "gt_release_index": "" if gt_release is None else gt_release["index"],
        "pick_xy_error_m": pick_xy_error,
        "place_xy_error_m": place_xy_error,
        "pred_base_xyz": ";".join(
            ",".join(f"{float(v):.6f}" for v in point[:3]) for point in pred_waypoints_base
        ),
    }
    return row


def choose_dataset_indices(dataset, tasks, samples_per_task, pair_scope, seed):
    rng = np.random.default_rng(seed)
    chosen = []
    for task in tasks:
        task_indices = list(dataset.task_to_indices[task])
        if pair_scope == "all-pairs":
            selected = task_indices if samples_per_task is None else task_indices[:samples_per_task]
            for pair_index in selected:
                agent_traj_index, teacher_traj_index = dataset._pairs[pair_index]
                chosen.append((task, pair_index, agent_traj_index, teacher_traj_index))
            continue

        by_agent = {}
        for pair_index in task_indices:
            agent_traj_index, teacher_traj_index = dataset._pairs[pair_index]
            by_agent.setdefault(agent_traj_index, []).append((pair_index, teacher_traj_index))
        agent_indices = list(by_agent)
        if pair_scope == "random-agent":
            rng.shuffle(agent_indices)
        if samples_per_task is not None:
            agent_indices = agent_indices[:samples_per_task]
        for agent_traj_index in agent_indices:
            pair_index, teacher_traj_index = by_agent[agent_traj_index][0]
            chosen.append((task, pair_index, agent_traj_index, teacher_traj_index))
    return chosen


def as_float_or_none(value):
    if value == "" or value is None:
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    return float(value)


def summarize(rows, metric_fields, rate_fields):
    groups = defaultdict(list)
    for row in rows:
        groups["overall"].append(row)
        groups[f"{row['mode']}:{row['task']}"].append(row)

    summary = {}
    for key, group_rows in groups.items():
        item = {"count": len(group_rows)}
        for field in metric_fields:
            values = [as_float_or_none(row.get(field)) for row in group_rows]
            values = [value for value in values if value is not None]
            if values:
                item[f"{field}_mean"] = float(np.mean(values))
                item[f"{field}_median"] = float(np.median(values))
        for field in rate_fields:
            values = [as_float_or_none(row.get(field)) for row in group_rows]
            values = [value for value in values if value is not None]
            if values:
                item[f"{field}_rate"] = float(np.mean(values))
                item[f"{field}_n"] = len(values)
        summary[key] = item
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Compute numeric OSVI-WM waypoint metrics on UR5e train/test splits."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--modes", nargs="+", default=["test"], choices=["train", "test"])
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--train-tasks", nargs="+", default=None)
    parser.add_argument("--test-tasks", nargs="+", default=None)
    parser.add_argument(
        "--pair-scope",
        choices=["unique-agent", "random-agent", "all-pairs"],
        default="unique-agent",
        help="unique-agent evaluates one teacher pairing per robot trajectory; all-pairs evaluates every agent/teacher pair.",
    )
    parser.add_argument(
        "--samples-per-task",
        type=int,
        default=None,
        help="Optional limit per task. Omit to evaluate the whole selected scope.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--interp-points", type=int, default=60)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--osvi-gripper-threshold", type=float, default=None)
    parser.add_argument("--ade-xy-threshold", type=float, default=0.06)
    parser.add_argument("--waypoint-xy-threshold", type=float, default=0.05)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    if args.osvi_gripper_threshold is None:
        grasp_scale = float(config.get("data", {}).get("grasp_scale", 0.2))
        args.osvi_gripper_threshold = grasp_scale / 2.0

    device = torch.device(args.device)
    model, checkpoint = load_model(config, args.checkpoint, device)
    checkpoint_epoch = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None

    rows = []
    for mode in args.modes:
        dataset = AgentTeacherDataset(**dict(config["data"]), mode=mode)
        delegate = unwrap_dataset(dataset)
        split_tasks = args.train_tasks if mode == "train" else args.test_tasks
        tasks = split_tasks or args.tasks or delegate.task_names
        selected = choose_dataset_indices(
            delegate,
            tasks,
            args.samples_per_task,
            args.pair_scope,
            args.seed,
        )
        print(f"{mode}: evaluating {len(selected)} samples from tasks {tasks}")

        for start in range(0, len(selected), args.batch_size):
            chunk = selected[start : start + args.batch_size]
            batch = default_collate([dataset[item[1]] for item in chunk])
            expert_context, agent_traj, _ = unpack_batch(batch)
            images = agent_traj["images"].to(device)
            context = expert_context["video"].to(device)

            with torch.inference_mode():
                if device.type == "cuda":
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        out = model(images, context)
                else:
                    out = model(images, context)

            pred = select_policy_waypoints(out["waypoints"], config["model"]).detach().float().cpu().numpy()
            gt_points = agent_traj["traj_points"].detach().float().cpu().numpy()
            projection_matrix = agent_traj["projection_matrix"].detach().float().cpu().numpy()
            pred_base = image_waypoints_to_base_batch(pred, projection_matrix)

            agent_fnames = agent_traj.get("fname", [""] * len(chunk))
            teacher_fnames = expert_context.get("fname", [""] * len(chunk))
            for offset, (task, dataset_idx, agent_traj_index, teacher_traj_index) in enumerate(chunk):
                meta = {
                    "mode": mode,
                    "task": task,
                    "dataset_index": dataset_idx,
                    "agent_traj_index": agent_traj_index,
                    "teacher_traj_index": teacher_traj_index,
                    "checkpoint_epoch": checkpoint_epoch,
                    "agent_fname": str(agent_fnames[offset]),
                    "teacher_fname": str(teacher_fnames[offset]),
                }
                rows.append(
                    metric_row(
                        meta=meta,
                        pred_waypoints_base=pred_base[offset],
                        gt_points=gt_points[offset],
                        gripper_threshold=args.osvi_gripper_threshold,
                        interp_points=args.interp_points,
                        ade_xy_threshold=args.ade_xy_threshold,
                        waypoint_xy_threshold=args.waypoint_xy_threshold,
                    )
                )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "mode",
        "task",
        "dataset_index",
        "agent_traj_index",
        "teacher_traj_index",
        "checkpoint_epoch",
        "agent_fname",
        "teacher_fname",
        "ade_xy_m",
        "ade_xyz_m",
        "fde_xy_m",
        "fde_xyz_m",
        "max_xy_m",
        "max_xyz_m",
        "dtw_xy_m",
        "dtw_xyz_m",
        "waypoint_min_xy_mean_m",
        "waypoint_min_xy_max_m",
        "waypoint_min_xyz_mean_m",
        "waypoint_min_xyz_max_m",
        "path_success",
        "waypoints_on_gt_success",
        "pred_gripper_values",
        "pred_gripper_states",
        "pred_gripper_transitions",
        "gt_gripper_transitions",
        "pred_has_close",
        "pred_has_release_after_close",
        "pred_close_waypoint",
        "pred_release_waypoint",
        "gt_close_index",
        "gt_release_index",
        "pick_xy_error_m",
        "place_xy_error_m",
        "pred_base_xyz",
    ]
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    metric_fields = [
        "ade_xy_m",
        "ade_xyz_m",
        "fde_xy_m",
        "fde_xyz_m",
        "dtw_xy_m",
        "dtw_xyz_m",
        "waypoint_min_xy_mean_m",
        "pick_xy_error_m",
        "place_xy_error_m",
    ]
    rate_fields = [
        "path_success",
        "waypoints_on_gt_success",
        "pred_has_close",
        "pred_has_release_after_close",
    ]
    summary = summarize(rows, metric_fields, rate_fields)
    output_json = Path(args.output_json) if args.output_json else output_csv.with_suffix(".summary.json")
    with output_json.open("w") as f:
        json.dump(
            {
                "checkpoint": str(args.checkpoint),
                "checkpoint_epoch": checkpoint_epoch,
                "pair_scope": args.pair_scope,
                "samples_per_task": args.samples_per_task,
                "osvi_gripper_threshold": args.osvi_gripper_threshold,
                "ade_xy_threshold": args.ade_xy_threshold,
                "waypoint_xy_threshold": args.waypoint_xy_threshold,
                "summary": summary,
            },
            f,
            indent=2,
        )

    print(f"Wrote metrics: {output_csv}")
    print(f"Wrote summary: {output_json}")


if __name__ == "__main__":
    main()
