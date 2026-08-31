import argparse
import os
import re
from collections import Counter

import torch
import yaml
from torch.utils.data import DataLoader

from dataset.agent_teacher_dataset import AgentTeacherDataset
from dataset.ur5e_task_dataset import make_balanced_task_batch_sampler


TASK_RE = re.compile(r"(task_\d+)")


def extract_task(path):
    match = TASK_RE.search(str(path))
    return match.group(1) if match else "unknown"


def as_list(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def describe_tensor(name, tensor):
    print(f"{name}:")
    print(f"  shape: {tuple(tensor.shape)}")
    print(f"  dtype: {tensor.dtype}")
    print(f"  device: {tensor.device}")
    print(f"  finite: {bool(torch.isfinite(tensor).all())}")
    print(f"  min/max: {float(tensor.min()):.6f} / {float(tensor.max()):.6f}")


def unpack_training_batch(batch):
    if len(batch) != 3:
        raise ValueError(f"Expected a 3-item batch, got {len(batch)} items")

    expert_context, second, third = batch
    if isinstance(second, dict) and "traj_points" in second:
        return expert_context, second, third

    if isinstance(third, dict) and "traj_points" in third:
        print("Detected agent_context batch; using third item as agent_traj.")
        return expert_context, third, None

    raise ValueError("Could not find agent_traj dict with 'traj_points' in the batch")


def validate_batch(expert_context, agent_traj, cfg):
    required_expert = ["video"]
    required_agent = ["images", "traj_points", "projection_matrix", "head_label"]

    for key in required_expert:
        if key not in expert_context:
            raise KeyError(f"expert_context missing key: {key}")
    for key in required_agent:
        if key not in agent_traj:
            raise KeyError(f"agent_traj missing key: {key}")

    context = expert_context["video"]
    images = agent_traj["images"]
    traj_points = agent_traj["traj_points"]
    projection = agent_traj["projection_matrix"]

    expected_h = cfg["height"]
    expected_w = cfg["width"]
    expected_context = cfg["T_context"]
    expected_agent_frames = cfg["T_pair"] + 1

    assert context.ndim == 5, f"expert video should be [B,T,C,H,W], got {context.shape}"
    assert images.ndim == 5, f"agent images should be [B,T,C,H,W], got {images.shape}"
    assert traj_points.ndim == 3, f"traj_points should be [B,N,4], got {traj_points.shape}"
    assert projection.ndim == 3, f"projection_matrix should be [B,4,4], got {projection.shape}"

    assert tuple(context.shape[1:]) == (expected_context, 3, expected_h, expected_w)
    assert tuple(images.shape[1:]) == (expected_agent_frames, 3, expected_h, expected_w)
    assert traj_points.shape[-1] == 4
    assert tuple(projection.shape[1:]) == (4, 4)

    for name, tensor in [
        ("expert_context['video']", context),
        ("agent_traj['images']", images),
        ("agent_traj['traj_points']", traj_points),
        ("agent_traj['projection_matrix']", projection),
    ]:
        assert torch.isfinite(tensor).all(), f"{name} contains NaN or inf"


def main():
    parser = argparse.ArgumentParser(description="Inspect UR5e batches exactly as train_pp.py receives them.")
    parser.add_argument("--config", default="configs/ur5e_pick_place_data.yaml")
    parser.add_argument("--mode", default="train", choices=["train", "test"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--num-batches", type=int, default=3)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    training_cfg = config.get("training", {})
    batch_size = args.batch_size or training_cfg.get("batch_size", 1)
    num_workers = args.num_workers if args.num_workers is not None else training_cfg.get("num_workers", 0)

    dataset = AgentTeacherDataset(**data_cfg, mode=args.mode)
    balanced_batches = bool(data_cfg.get("balanced_batches", False)) and args.mode == "train"
    if balanced_batches:
        batch_sampler = make_balanced_task_batch_sampler(
            dataset,
            batch_size=batch_size,
            samples_per_task_per_batch=data_cfg.get("samples_per_task_per_batch"),
            drop_last=data_cfg.get("balanced_drop_last", True),
            shuffle=args.shuffle and data_cfg.get("balanced_shuffle", True),
            seed=data_cfg.get("balanced_seed", 0),
            epoch_strategy=data_cfg.get("balanced_epoch_strategy", "max"),
        )
        loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=num_workers)
        batch_size = batch_sampler.batch_size
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=args.shuffle, num_workers=num_workers)

    print(f"Config: {args.config}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Mode: {args.mode}")
    print(f"Dataset length: {len(dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Balanced batches: {balanced_batches}")
    if balanced_batches:
        print(f"Balanced tasks: {batch_sampler.tasks}")
        print(f"Samples per task per batch: {batch_sampler.samples_per_task_per_batch}")
        print(f"Balanced batches per epoch: {len(batch_sampler)}")
    print(f"Shuffle: {args.shuffle}")
    print(f"Num workers: {num_workers}")

    total_task_counts = Counter()
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= args.num_batches:
            break

        expert_context, agent_traj, labels = unpack_training_batch(batch)
        validate_batch(expert_context, agent_traj, data_cfg)

        agent_fnames = as_list(agent_traj.get("fname", []))
        expert_fnames = as_list(expert_context.get("fname", []))
        agent_tasks = as_list(agent_traj.get("task_name", [])) or [extract_task(path) for path in agent_fnames]
        expert_tasks = as_list(expert_context.get("task_name", [])) or [extract_task(path) for path in expert_fnames]
        batch_task_counts = Counter(agent_tasks)
        total_task_counts.update(agent_tasks)

        print(f"\nBatch {batch_idx}")
        print(f"  labels: {labels}")
        print(f"  agent tasks: {dict(batch_task_counts)}")
        print(f"  expert tasks: {Counter(expert_tasks)}")
        print(f"  first agent fname: {agent_fnames[0] if agent_fnames else 'n/a'}")
        print(f"  first expert fname: {expert_fnames[0] if expert_fnames else 'n/a'}")
        describe_tensor("  expert_context['video']", expert_context["video"])
        describe_tensor("  agent_traj['images']", agent_traj["images"])
        describe_tensor("  agent_traj['traj_points']", agent_traj["traj_points"])
        describe_tensor("  agent_traj['projection_matrix']", agent_traj["projection_matrix"])

        # This mirrors the variables read in train_pp.py.
        images = agent_traj["images"]
        context = expert_context["video"]
        traj_points = agent_traj["traj_points"]
        projection_matrix = agent_traj["projection_matrix"]
        head_label = agent_traj["head_label"]
        print("  train_pp variables OK:")
        print(f"    images {tuple(images.shape)}")
        print(f"    context {tuple(context.shape)}")
        print(f"    traj_points {tuple(traj_points.shape)}")
        print(f"    projection_matrix {tuple(projection_matrix.shape)}")
        print(f"    head_label {tuple(head_label.shape)}")

    print(f"\nObserved task counts across inspected batches: {dict(total_task_counts)}")
    print("BATCH INSPECTION OK")


if __name__ == "__main__":
    main()
