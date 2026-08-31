import argparse
from pathlib import Path

import numpy as np
import yaml
import cv2
from torch.utils.data import DataLoader

from dataset.agent_teacher_dataset import AgentTeacherDataset


IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def describe_array(name, value):
    arr = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    print(f"{name}:")
    print(f"  shape: {arr.shape}")
    print(f"  dtype: {arr.dtype}")
    print(f"  min/max: {arr.min()} / {arr.max()}")


def chw_to_uint8_rgb(frame):
    arr = frame.detach().cpu().numpy() if hasattr(frame, "detach") else np.asarray(frame)
    arr = np.transpose(arr, (1, 2, 0)).astype(np.float32)
    # randomize_video normalizes like ImageNet in OSVI. Undo it for visualization.
    if arr.min() < 0.0 or arr.max() > 1.5:
        arr = arr * IMAGENET_STD + IMAGENET_MEAN
    if arr.max() <= 1.5:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def save_preview_frames(expert_context, agent_traj, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    expert_video = expert_context["video"]
    agent_images = agent_traj["images"]

    expert_indices = sorted(set([0, len(expert_video) // 2, len(expert_video) - 1]))
    for i in expert_indices:
        rgb = chw_to_uint8_rgb(expert_video[i])
        cv2.imwrite(str(out_dir / f"expert_context_{i:02d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    for i in range(len(agent_images)):
        rgb = chw_to_uint8_rgb(agent_images[i])
        cv2.imwrite(str(out_dir / f"agent_image_{i:02d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    print(f"\nSaved preview frames in: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Smoke test for the UR5e OSVI dataset loader.")
    parser.add_argument("--config", default="configs/ur5e_pick_place_data.yaml")
    parser.add_argument("--mode", default="train", choices=["train", "test"])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--preview-dir", default=None)
    parser.add_argument(
        "--dump-preprocessing-dir",
        default=None,
        help="Save raw/decode/crop/resize/model-input images from the dataset pipeline.",
    )
    parser.add_argument(
        "--dump-preprocessing-max-samples",
        type=int,
        default=5,
        help="Maximum number of agent and teacher samples to dump.",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)["data"]

    if args.dump_preprocessing_dir:
        cfg["debug_preprocess_dump_dir"] = args.dump_preprocessing_dir
        cfg["debug_preprocess_dump_max_samples"] = args.dump_preprocessing_max_samples

    dataset = AgentTeacherDataset(**cfg, mode=args.mode)
    print(f"Dataset mode: {args.mode}")
    print(f"Dataset length: {len(dataset)}")

    expert_context, agent_traj, label = dataset[0]
    print("\nSingle sample keys:")
    print("  expert_context:", sorted(expert_context.keys()))
    print("  agent_traj:", sorted(agent_traj.keys()))
    print("  label:", label)
    print("  expert fname:", expert_context.get("fname"))
    print("  agent fname:", agent_traj.get("fname"))

    print("\nSingle sample arrays:")
    describe_array("expert_context['video']", expert_context["video"])
    describe_array("agent_traj['images']", agent_traj["images"])
    describe_array("agent_traj['traj_points']", agent_traj["traj_points"])
    describe_array("agent_traj['projection_matrix']", agent_traj["projection_matrix"])
    if args.preview_dir:
        save_preview_frames(expert_context, agent_traj, args.preview_dir)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    expert_batch, agent_batch, labels = next(iter(loader))

    print("\nBatch arrays:")
    describe_array("expert_batch['video']", expert_batch["video"])
    describe_array("agent_batch['images']", agent_batch["images"])
    describe_array("agent_batch['traj_points']", agent_batch["traj_points"])
    describe_array("agent_batch['projection_matrix']", agent_batch["projection_matrix"])
    print("labels:", labels)

    expected_projection_shape = (4, 4)
    assert expert_context["video"].shape[1:] == (3, cfg["height"], cfg["width"])
    assert agent_traj["images"].shape[1:] == (3, cfg["height"], cfg["width"])
    assert agent_traj["traj_points"].shape[-1] == 4
    assert agent_traj["projection_matrix"].shape == expected_projection_shape

    print("\nSMOKE TEST OK")


if __name__ == "__main__":
    main()
