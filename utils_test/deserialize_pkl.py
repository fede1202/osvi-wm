#!/usr/bin/env python3
"""Deserialize a PKL trajectory and dump the full object to JSON.

This is intentionally generic: it does not extract images or compute metrics.
It loads the pickle, materializes every Trajectory step, converts NumPy arrays
and bytes to JSON-safe values, and writes one *_deserialized.json file.
"""

from __future__ import annotations

import argparse
import base64
import glob
import json
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


class FallbackTrajectory:
    """Small replacement for savers.Trajectory when the original class is unavailable."""

    @property
    def T(self):
        return len(getattr(self, "_data", []))

    def __len__(self):
        return self.T

    def __iter__(self):
        for index in range(self.T):
            yield self.get(index)

    def __getitem__(self, index):
        return self.get(index)

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

    def get_raw_state(self, index):
        raw_state = getattr(self, "_raw_state", None)
        if raw_state is None:
            raise AttributeError("FallbackTrajectory has no _raw_state")
        return raw_state[index]


def resolve_path(value: str) -> Path:
    return Path(os.path.expanduser(value)).resolve()


def add_dataset_collector_savers_to_path(root_arg: Optional[str]) -> None:
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
        if (candidate / "savers.py").is_file() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
            return


def resolve_trajectory_class():
    for module_name in ("savers", "scripts.savers", "hem.datasets.savers.trajectory"):
        try:
            module = __import__(module_name, fromlist=["Trajectory"])
            return getattr(module, "Trajectory")
        except Exception:
            continue
    return FallbackTrajectory


PLACEHOLDER_CLASSES: Dict[str, type] = {}


def placeholder_class(module: str, name: str) -> type:
    key = f"{module}.{name}"
    if key in PLACEHOLDER_CLASSES:
        return PLACEHOLDER_CLASSES[key]

    cls = type(name, (object,), {"__module__": module})
    PLACEHOLDER_CLASSES[key] = cls
    return cls


class CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "Trajectory":
            return resolve_trajectory_class()
        if name == "LazyDecompressionDict":
            return dict
        try:
            return super().find_class(module, name)
        except (AttributeError, ImportError, ModuleNotFoundError):
            return placeholder_class(module, name)


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return CompatUnpickler(handle).load()


def natural_key(path: Path) -> List[Any]:
    return [int(chunk) if chunk.isdigit() else chunk for chunk in re.split(r"(\d+)", str(path))]


def expand_pickles(specs: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for spec in specs:
        expanded = os.path.expanduser(spec)
        if any(char in expanded for char in "*?["):
            paths.extend(resolve_path(path) for path in glob.glob(expanded, recursive=True))
            continue

        path = resolve_path(expanded)
        if path.is_dir():
            paths.extend(path.rglob("*.pkl"))
        else:
            paths.append(path)

    unique: List[Path] = []
    seen = set()
    for path in sorted(paths, key=natural_key):
        if path.suffix.lower() == ".pkl" and path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def is_trajectory_like(value: Any) -> bool:
    return (
        hasattr(value, "get")
        and hasattr(value, "__len__")
        and (value.__class__.__name__ == "Trajectory" or hasattr(value, "_data"))
    )


def materialize(value: Any, seen: Optional[set] = None) -> Any:
    if seen is None:
        seen = set()

    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return {
            "__ndarray__": value.tolist(),
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }

    if isinstance(value, bytes):
        return {
            "__bytes_base64__": base64.b64encode(value).decode("ascii"),
            "num_bytes": len(value),
        }

    if isinstance(value, bytearray):
        raw = bytes(value)
        return {
            "__bytearray_base64__": base64.b64encode(raw).decode("ascii"),
            "num_bytes": len(raw),
        }

    object_id = id(value)
    if object_id in seen:
        return {"__recursive_ref__": type(value).__name__}
    seen.add(object_id)

    if is_trajectory_like(value):
        steps = []
        for index in range(len(value)):
            steps.append(materialize(value.get(index), seen))

        result = {
            "__type__": f"{value.__class__.__module__}.{value.__class__.__name__}",
            "length": len(value),
            "steps": steps,
        }
        if hasattr(value, "config_str"):
            result["config_str"] = materialize(getattr(value, "config_str"), seen)
        if hasattr(value, "get_raw_state"):
            raw_states = []
            for index in range(len(value)):
                try:
                    raw_states.append(materialize(value.get_raw_state(index), seen))
                except Exception:
                    raw_states.append(None)
            result["raw_states"] = raw_states
        seen.remove(object_id)
        return result

    if isinstance(value, dict):
        if all(isinstance(key, str) for key in value.keys()):
            result = {}
            for key in value.keys():
                result[key] = materialize(value[key], seen)
        else:
            result = {
                "__dict_items__": [
                    {"key": materialize(key, seen), "value": materialize(value[key], seen)}
                    for key in value.keys()
                ]
            }
        seen.remove(object_id)
        return result

    if isinstance(value, (list, tuple, set)):
        result = [materialize(item, seen) for item in value]
        seen.remove(object_id)
        return result

    if hasattr(value, "__dict__"):
        result = {
            "__type__": f"{value.__class__.__module__}.{value.__class__.__name__}",
            "__attrs__": materialize(vars(value), seen),
        }
        seen.remove(object_id)
        return result

    seen.remove(object_id)
    return repr(value)


def output_path_for(pkl_path: Path, output_dir: Optional[Path], output_path: Optional[Path]) -> Path:
    if output_path is not None:
        return output_path
    directory = output_dir if output_dir is not None else pkl_path.parent
    return directory / f"{pkl_path.stem}_deserialized.json"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="+", help="PKL file, directory, or glob.")
    parser.add_argument(
        "--dataset-collector-root",
        help="Optional path to dataset_collector_pkg or its scripts dir, so savers.py is importable.",
    )
    parser.add_argument("--output", help="Output JSON path. Only valid with one input PKL.")
    parser.add_argument("--output-dir", help="Directory for generated *_deserialized.json files.")
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation. Use 0 for compact JSON.")
    return parser.parse_args()


def main():
    args = parse_args()
    add_dataset_collector_savers_to_path(args.dataset_collector_root)

    pkl_paths = expand_pickles(args.input)
    if not pkl_paths:
        raise FileNotFoundError(f"No PKL files found from input: {args.input}")

    if args.output and len(pkl_paths) != 1:
        raise ValueError("--output can be used only with one PKL input")

    output_dir = resolve_path(args.output_dir) if args.output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    output_path = resolve_path(args.output) if args.output else None

    indent = None if args.indent == 0 else args.indent

    for pkl_path in pkl_paths:
        payload = load_pickle(pkl_path)
        deserialized = materialize(payload)
        out_path = output_path_for(pkl_path, output_dir, output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(deserialized, handle, indent=indent)

        print(f"[ok] {pkl_path}")
        print(f"     type: {payload.__class__.__module__}.{payload.__class__.__name__}")
        print(f"     json: {out_path}")


if __name__ == "__main__":
    main()
