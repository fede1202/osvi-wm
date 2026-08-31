import argparse
import csv
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data._utils.collate import default_collate

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.agent_teacher_dataset import AgentTeacherDataset
from models.model import StateSpaceModel
from utils.projection_utils import image_point_to_pixels


IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def chw_to_uint8_rgb(frame):
    arr = frame.detach().float().cpu().numpy()
    arr = np.transpose(arr, (1, 2, 0))
    if arr.min() < 0.0 or arr.max() > 1.5:
        arr = arr * IMAGENET_STD + IMAGENET_MEAN
    if arr.max() <= 1.5:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def unwrap_dataset(dataset):
    return getattr(dataset, "_delegate", None) or dataset


def unpack_batch(batch):
    if len(batch) != 3:
        raise ValueError(f"Expected a 3-item batch, got {len(batch)} items")
    expert_context, second, third = batch
    if isinstance(second, dict) and "traj_points" in second:
        return expert_context, second, third
    if isinstance(third, dict) and "traj_points" in third:
        return expert_context, third, None
    raise ValueError("Could not find agent_traj dict with 'traj_points' in the batch")


def load_model(config, checkpoint_path, device):
    model_cfg = config["model"]
    model = StateSpaceModel(
        latent_dim=model_cfg["latent_dim"],
        waypoints=model_cfg["waypoints"],
        sub_waypoints=model_cfg["sub_waypoints"],
        metaworld=False,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if state_dict and all(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model, checkpoint


def select_policy_waypoints(out_waypoints, model_cfg):
    num_waypoints = int(model_cfg["waypoints"])
    if model_cfg.get("sub_waypoints", False):
        start = num_waypoints * (num_waypoints - 1) // 2
    else:
        start = 0
    return out_waypoints[:, start : start + num_waypoints]


def normalized_xy_to_pixels(points, image_shape):
    row_col = image_point_to_pixels(points[..., :2], image_shape)
    xy = np.stack([row_col[..., 1], row_col[..., 0]], axis=-1)
    return xy


def homogeneous_image_to_pixels(points, image_shape):
    row_col = image_point_to_pixels(points[..., :3], image_shape)
    xy = np.stack([row_col[..., 1], row_col[..., 0]], axis=-1)
    return xy


def base_points_to_xy_pixels(points_base, projection_matrix, image_shape):
    points_h = np.concatenate(
        [points_base[..., :3], np.ones((*points_base.shape[:-1], 1), dtype=np.float32)],
        axis=-1,
    )
    image_h = np.einsum("ij,...j->...i", np.linalg.inv(projection_matrix), points_h)
    return homogeneous_image_to_pixels(image_h, image_shape)


def draw_overlay(
    out_path,
    image,
    pred_xy,
    pred_gripper_values,
    pred_gripper_states,
    gt_xy,
    title,
    agent_fname,
    teacher_fname,
    expert_images=None,
):
    h, w = image.shape[:2]
    if expert_images:
        fig = plt.figure(figsize=(10, 8))
        grid = fig.add_gridspec(2, len(expert_images), height_ratios=[0.55, 1.0])
        for col, (expert_img, expert_title) in enumerate(expert_images):
            expert_ax = fig.add_subplot(grid[0, col])
            expert_ax.imshow(expert_img)
            expert_ax.set_title(expert_title, fontsize=8)
            expert_ax.axis("off")
        ax = fig.add_subplot(grid[1, :])
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(image)

    in_bounds = (
        (pred_xy[:, 0] >= 0)
        & (pred_xy[:, 0] < w)
        & (pred_xy[:, 1] >= 0)
        & (pred_xy[:, 1] < h)
    )
    for i, (x, y) in enumerate(pred_xy):
        color = "tab:orange" if in_bounds[i] else "red"
        ax.scatter(x, y, c=color, marker="x", s=70, linewidths=2)
        label = f"{i + 1} {pred_gripper_states[i]} g={pred_gripper_values[i]:.2f}"
        ax.text(x + 4, y - 4, label, color=color, fontsize=8, weight="bold")

    gt_in_bounds = (
        (gt_xy[:, 0] >= 0)
        & (gt_xy[:, 0] < w)
        & (gt_xy[:, 1] >= 0)
        & (gt_xy[:, 1] < h)
    )
    if gt_in_bounds.any():
        valid_gt = gt_xy[gt_in_bounds]
        ax.plot(valid_gt[:, 0], valid_gt[:, 1], color="tab:blue", linewidth=1.5, alpha=0.85)
        ax.scatter(valid_gt[0, 0], valid_gt[0, 1], c="lime", s=35, label="GT start")
        ax.scatter(valid_gt[-1, 0], valid_gt[-1, 1], c="tab:red", s=35, label="GT end")

    ax.set_title(title, fontsize=10)
    ax.text(
        0.01,
        -0.08,
        f"agent: {Path(agent_fname).name} | teacher: {Path(teacher_fname).name}",
        transform=ax.transAxes,
        fontsize=7,
        va="top",
    )
    ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def gripper_states(values, threshold):
    return ["close" if float(value) >= threshold else "open" for value in values]


def gripper_transitions(values, threshold):
    states = gripper_states(values, threshold)
    transitions = []
    previous = states[0] if states else None
    for index, state in enumerate(states[1:], start=1):
        if state != previous:
            transitions.append(
                {
                    "index": index,
                    "state": state,
                    "value": float(values[index]),
                }
            )
            previous = state
    return transitions


def choose_indices(dataset, tasks, samples_per_task, seed, random_agent_order, random_teacher):
    rng = random.Random(seed)
    chosen = []
    for task in tasks:
        by_agent_traj = {}
        for pair_index in dataset.task_to_indices[task]:
            agent_traj_index, teacher_traj_index = dataset._pairs[pair_index]
            by_agent_traj.setdefault(agent_traj_index, []).append((pair_index, teacher_traj_index))

        agent_traj_indices = list(by_agent_traj)
        if random_agent_order:
            rng.shuffle(agent_traj_indices)
        selected_agent_trajs = agent_traj_indices[:samples_per_task]
        if len(selected_agent_trajs) < samples_per_task:
            print(
                f"WARNING: {task} has only {len(selected_agent_trajs)} unique agent trajectories, "
                f"requested {samples_per_task}."
            )

        for agent_traj_index in selected_agent_trajs:
            pair_candidates = by_agent_traj[agent_traj_index]
            if random_teacher:
                pair_index, teacher_traj_index = rng.choice(pair_candidates)
            else:
                pair_index, teacher_traj_index = pair_candidates[0]
            chosen.append((task, pair_index, agent_traj_index, teacher_traj_index))
    return chosen


def default_output_dir(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent / "image_waypoint_overlays"
    return Path("image_waypoint_overlays")


def run_mode(args, config, model, checkpoint_epoch, mode, output_dir, device):
    data_cfg = dict(config["data"])
    dataset = AgentTeacherDataset(**data_cfg, mode=mode)
    delegate = unwrap_dataset(dataset)
    if not hasattr(delegate, "task_to_indices"):
        raise TypeError("This script expects the UR5e task-folder dataset.")

    all_tasks = delegate.task_names
    split_tasks = args.train_tasks if mode == "train" else args.test_tasks
    tasks = split_tasks or args.tasks or all_tasks
    missing = sorted(set(tasks) - set(all_tasks))
    if missing:
        raise ValueError(f"Tasks not present in {mode} split: {missing}. Available: {all_tasks}")

    selected = choose_indices(
        delegate,
        tasks,
        args.samples_per_task,
        args.seed,
        args.random,
        args.random_teacher,
    )
    rows = []
    for sample_id, (task, dataset_idx, agent_traj_index, teacher_traj_index) in enumerate(selected):
        batch = default_collate([dataset[dataset_idx]])
        expert_context, agent_traj, _ = unpack_batch(batch)

        images = agent_traj["images"].to(device)
        context = expert_context["video"].to(device)
        with torch.inference_mode():
            if device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out = model(images, context)
            else:
                out = model(images, context)

        pred = select_policy_waypoints(out["waypoints"], config["model"])[0].detach().float().cpu().numpy()
        pred_gripper_values = pred[:, 3]
        pred_gripper_states = gripper_states(pred_gripper_values, args.osvi_gripper_threshold)
        image = chw_to_uint8_rgb(agent_traj["images"][0, 0])
        image_shape = image.shape[:2]
        pred_xy = normalized_xy_to_pixels(pred, image_shape)

        gt_points = agent_traj["traj_points"][0].detach().float().cpu().numpy()
        projection_matrix = agent_traj["projection_matrix"][0].detach().float().cpu().numpy()
        gt_xy = base_points_to_xy_pixels(gt_points, projection_matrix, image_shape)

        expert_video = expert_context["video"][0]
        expert_inds = sorted(set([0, expert_video.shape[0] // 2, expert_video.shape[0] - 1]))
        expert_images = [
            (chw_to_uint8_rgb(expert_video[i]), f"teacher t={i}")
            for i in expert_inds
        ]

        agent_fname = agent_traj.get("fname", ["unknown"])[0]
        teacher_fname = expert_context.get("fname", ["unknown"])[0]
        out_path = (
            output_dir
            / mode
            / task
            / f"{sample_id:04d}_agent_{agent_traj_index}_teacher_{teacher_traj_index}_idx_{dataset_idx}.png"
        )
        title = (
            f"{mode} {task} | idx={dataset_idx} | "
            f"agent_traj={agent_traj_index} | teacher_traj={teacher_traj_index} | "
            f"checkpoint_epoch={checkpoint_epoch if checkpoint_epoch is not None else 'unknown'}"
        )
        draw_overlay(
            out_path,
            image,
            pred_xy,
            pred_gripper_values,
            pred_gripper_states,
            gt_xy,
            title,
            agent_fname,
            teacher_fname,
            expert_images=expert_images,
        )

        rows.append(
            {
                "mode": mode,
                "task": task,
                "sample_id": sample_id,
                "dataset_index": dataset_idx,
                "agent_traj_index": agent_traj_index,
                "teacher_traj_index": teacher_traj_index,
                "checkpoint_epoch": checkpoint_epoch,
                "output_png": str(out_path),
                "agent_fname": str(agent_fname),
                "teacher_fname": str(teacher_fname),
                "pred_gripper_values": ";".join(f"{float(v):.6f}" for v in pred_gripper_values),
                "pred_gripper_states": ";".join(pred_gripper_states),
                "pred_in_bounds": int(
                    np.all(
                        (pred_xy[:, 0] >= 0)
                        & (pred_xy[:, 0] < image_shape[1])
                        & (pred_xy[:, 1] >= 0)
                        & (pred_xy[:, 1] < image_shape[0])
                    )
                ),
            }
        )

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Overlay OSVI-WM predicted UR5e image-space waypoints on agent images."
    )
    parser.add_argument("--config", required=True, help="Training config YAML or copied run config.yaml.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path, e.g. runs/<run>/checkpoints/last.pt.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--modes", nargs="+", default=["train", "test"], choices=["train", "test"])
    parser.add_argument("--tasks", nargs="+", default=None, help="Optional task list applied to each selected split.")
    parser.add_argument("--train-tasks", nargs="+", default=None, help="Optional train split task list.")
    parser.add_argument("--test-tasks", nargs="+", default=None, help="Optional test split task list.")
    parser.add_argument("--samples-per-task", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--random",
        action="store_true",
        help="Randomize the selected agent/robot trajectories within each task.",
    )
    parser.add_argument(
        "--random-teacher",
        action="store_true",
        help="Randomize the teacher trajectory too. By default the same first teacher is reused per task.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--expected-epoch", type=int, default=None)
    parser.add_argument(
        "--osvi-gripper-threshold",
        type=float,
        default=None,
        help="Threshold on OSVI waypoint gripper value. Defaults to grasp_scale/2, or 0.1.",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    if args.osvi_gripper_threshold is None:
        grasp_scale = float(config.get("data", {}).get("grasp_scale", 0.2))
        args.osvi_gripper_threshold = grasp_scale / 2.0
    print(f"Using OSVI gripper threshold: {args.osvi_gripper_threshold}")

    device = torch.device(args.device)
    model, checkpoint = load_model(config, args.checkpoint, device)
    checkpoint_epoch = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
    if args.expected_epoch is not None and checkpoint_epoch != args.expected_epoch:
        print(
            f"WARNING: checkpoint epoch is {checkpoint_epoch}, "
            f"not expected epoch {args.expected_epoch}."
        )

    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(args.checkpoint)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for mode in args.modes:
        all_rows.extend(run_mode(args, config, model, checkpoint_epoch, mode, output_dir, device))

    csv_path = output_dir / "index.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mode",
                "task",
                "sample_id",
                "dataset_index",
                "agent_traj_index",
                "teacher_traj_index",
                "checkpoint_epoch",
                "output_png",
                "agent_fname",
                "teacher_fname",
                "pred_gripper_values",
                "pred_gripper_states",
                "pred_in_bounds",
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Saved {len(all_rows)} overlays to {output_dir}")
    print(f"Wrote index: {csv_path}")


if __name__ == "__main__":
    main()
