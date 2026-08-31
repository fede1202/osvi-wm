#!/usr/bin/env python3
"""Extract the front camera image from trajectory PKLs and report its format."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


DEFAULT_IMAGE_KEYS = ("camera_front_image", "image_camera_front", "front_camera_image")


class FallbackTrajectory:
    def __len__(self):
        return len(getattr(self, "_data", []))

    def get(self, index, decompress=True):
        obs, reward, done, info, action = self._data[index]
        return {
            key: value
            for key, value in {
                "obs": obs,
                "reward": reward,
                "done": done,
                "info": info,
                "action": action,
            }.items()
            if value is not None
        }


def natural_key(path: Path) -> List[Any]:
    return [int(chunk) if chunk.isdigit() else chunk for chunk in re.split(r"(\d+)", str(path))]


def resolve_path(value: str) -> Path:
    return Path(os.path.expanduser(value)).resolve()


def add_dataset_collector_savers_to_path(root_arg: Optional[str]) -> Optional[Path]:
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


def resolve_trajectory_class():
    for module_name in ("savers", "scripts.savers", "hem.datasets.savers.trajectory"):
        try:
            module = __import__(module_name, fromlist=["Trajectory"])
            return getattr(module, "Trajectory")
        except Exception:
            continue
    return FallbackTrajectory


class CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "Trajectory":
            return resolve_trajectory_class()
        return super().find_class(module, name)


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return CompatUnpickler(handle).load()


def unwrap_trajectory(payload):
    if isinstance(payload, dict) and "traj" in payload:
        return payload["traj"]
    return payload


def get_step(traj, index: int):
    if hasattr(traj, "get"):
        return traj.get(index)
    return traj[index]


def get_obs(step) -> Dict[str, Any]:
    if isinstance(step, dict) and isinstance(step.get("obs"), dict):
        return step["obs"]
    if isinstance(step, dict):
        return step
    raise ValueError(f"Step is not a dict: {type(step).__name__}")


def find_front_image(obs: Dict[str, Any], image_keys: Sequence[str]) -> Tuple[str, Any]:
    for key in image_keys:
        if key in obs:
            return key, obs[key]
    raise KeyError(f"None of image keys {tuple(image_keys)} found. Available keys: {sorted(obs.keys())}")


def array_stats(arr: np.ndarray, prefix: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        f"{prefix}_shape": list(arr.shape),
        f"{prefix}_dtype": str(arr.dtype),
        f"{prefix}_ndim": int(arr.ndim),
    }
    if arr.size:
        if np.issubdtype(arr.dtype, np.number):
            result[f"{prefix}_min"] = float(np.nanmin(arr))
            result[f"{prefix}_max"] = float(np.nanmax(arr))
        result[f"{prefix}_size"] = int(arr.size)
    return result


def to_uint8_image(value: Any, raw_color: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    info: Dict[str, Any] = {"python_type": type(value).__name__}

    if isinstance(value, (bytes, bytearray)):
        encoded = np.frombuffer(value, dtype=np.uint8)
        info["encoded_bytes"] = int(encoded.size)
        image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError("cv2.imdecode failed for encoded image bytes")
        info["decode_method"] = "cv2.imdecode"
        info["decoded_color_note"] = "cv2 returns BGR/BGRA for color images"
        info.update(array_stats(image, "decoded"))
        return normalize_decoded_image_for_cv2(image), info

    arr = np.asarray(value)
    info.update(array_stats(arr, "original"))

    if arr.ndim == 1 and arr.dtype == np.uint8:
        image = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError("cv2.imdecode failed for 1D uint8 encoded image")
        info["decode_method"] = "cv2.imdecode"
        info["decoded_color_note"] = "cv2 returns BGR/BGRA for color images"
        info.update(array_stats(image, "decoded"))
        return normalize_decoded_image_for_cv2(image), info

    info["decode_method"] = "raw_array"
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
        info["layout_conversion"] = "CHW_to_HWC"
    else:
        info["layout_conversion"] = "none"

    if arr.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D image array, got shape {arr.shape}")

    image = normalize_dtype_to_uint8(arr)
    info.update(array_stats(image, "saved_array"))

    if image.ndim == 3 and image.shape[-1] == 3 and raw_color == "rgb":
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        info["save_color_conversion"] = "RGB_to_BGR_for_cv2_imwrite"
    elif image.ndim == 3 and image.shape[-1] == 4 and raw_color == "rgb":
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
        info["save_color_conversion"] = "RGBA_to_BGRA_for_cv2_imwrite"
    else:
        info["save_color_conversion"] = "none"

    return image, info


def normalize_decoded_image_for_cv2(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        return normalize_dtype_to_uint8(image)
    return image


def normalize_dtype_to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    if arr.size and np.nanmin(arr) >= 0.0 and np.nanmax(arr) <= 1.0:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def infer_task_id(path: Path) -> str:
    for part in reversed(path.parts):
        match = re.fullmatch(r"task_?(\d+)", part)
        if match:
            return f"task_{int(match.group(1)):02d}"
    return "unknown_task"


def make_output_path(output_dir: Path, pkl_path: Path, step_index: int, image_key: str) -> Path:
    task_id = infer_task_id(pkl_path)
    safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", image_key)
    return output_dir / task_id / f"{pkl_path.stem}_step_{step_index:03d}_{safe_key}.png"


def expand_pickles(specs: Sequence[str]) -> List[Path]:
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


def extract_one(pkl_path: Path, args) -> Dict[str, Any]:
    payload = load_pickle(pkl_path)
    traj = unwrap_trajectory(payload)
    step = get_step(traj, int(args.step))
    obs = get_obs(step)
    image_key, image_value = find_front_image(obs, args.image_keys)
    image_for_write, info = to_uint8_image(image_value, args.raw_color)

    output_path = make_output_path(resolve_path(args.output_dir), pkl_path, int(args.step), image_key)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image_for_write):
        raise IOError(f"Could not write image: {output_path}")

    row = {
        "pkl_path": str(pkl_path),
        "step": int(args.step),
        "trajectory_len": int(len(traj)),
        "image_key": image_key,
        "output_png": str(output_path),
        "written_shape": list(image_for_write.shape),
        "written_dtype": str(image_for_write.dtype),
    }
    row.update(info)
    return row


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="+", help="Trajectory .pkl file, directory, or glob.")
    parser.add_argument("--dataset-collector-root", help="Optional dataset_collector path so savers.py is importable.")
    parser.add_argument("--output-dir", default="front_image_extract")
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument(
        "--image-key",
        dest="image_keys",
        action="append",
        default=None,
        help="Image key to try. Can be repeated. Defaults to camera_front_image, image_camera_front, front_camera_image.",
    )
    parser.add_argument(
        "--raw-color",
        choices=["rgb", "bgr"],
        default="rgb",
        help="Color order for raw HWC/CHW arrays. Encoded images decoded by cv2 are saved as decoded.",
    )
    parser.add_argument("--max-files", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.image_keys = tuple(args.image_keys or DEFAULT_IMAGE_KEYS)
    add_dataset_collector_savers_to_path(args.dataset_collector_root)

    pkl_paths = expand_pickles(args.input)
    if args.max_files is not None:
        pkl_paths = pkl_paths[: int(args.max_files)]
    if not pkl_paths:
        raise FileNotFoundError("No .pkl files matched the input.")

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    errors = []
    for pkl_path in pkl_paths:
        try:
            row = extract_one(pkl_path, args)
            rows.append(row)
            print(
                f"[ok] {pkl_path} step={row['step']} key={row['image_key']} "
                f"shape={row.get('original_shape', row.get('decoded_shape'))} "
                f"dtype={row.get('original_dtype', row.get('decoded_dtype'))} -> {row['output_png']}"
            )
        except Exception as exc:
            errors.append({"pkl_path": str(pkl_path), "error": f"{type(exc).__name__}: {exc}"})
            print(f"[error] {pkl_path}: {type(exc).__name__}: {exc}")

    if rows:
        write_jsonl(output_dir / "front_image_info.jsonl", rows)
        write_csv(output_dir / "front_image_info.csv", rows)
    if errors:
        write_jsonl(output_dir / "front_image_errors.jsonl", errors)

    print(
        json.dumps(
            {
                "num_pkls": len(pkl_paths),
                "num_saved": len(rows),
                "num_errors": len(errors),
                "output_dir": str(output_dir),
                "info_jsonl": str(output_dir / "front_image_info.jsonl") if rows else None,
                "info_csv": str(output_dir / "front_image_info.csv") if rows else None,
                "errors_jsonl": str(output_dir / "front_image_errors.jsonl") if errors else None,
            },
            indent=2,
        )
    )
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
