#!/usr/bin/env python3

import json
import pickle
import re
from pathlib import Path

import numpy as np


# ==================================================================
# PATH
# ==================================================================

INPUT_ROOT = Path(
    "/home/asus-mivia/Desktop/dataset/pick_place/real_eye_in_hand_ur5e_pick_place"
)

OUTPUT_ROOT = Path(
    "/home/asus-mivia/Desktop/Multi-Task-LFD/repo/osvi-wm/utils_test/extraction_video_pkl_dataset_eyeinhand"
)


# ==================================================================
# Compatibilità con i pickle contenenti Trajectory
# ==================================================================

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


def resolve_trajectory_class():
    for module_name in (
        "savers",
        "scripts.savers",
        "hem.datasets.savers.trajectory",
    ):
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


def load_pickle(path):
    with open(path, "rb") as f:
        return CompatUnpickler(f).load()


# ==================================================================
# Conversione valori NumPy -> JSON
# ==================================================================

def to_json(value):

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, tuple):
        return list(value)

    return value


# ==================================================================
# Accesso agli step
# ==================================================================

def get_step(traj, index):

    if hasattr(traj, "get"):
        return traj.get(index)

    return traj[index]


# ==================================================================
# Recupero action
# ==================================================================

def get_action(step, obs, payload, step_index):
    """
    Cerca l'action in diversi possibili punti:

    1. step["action"]
    2. step["actions"]
    3. obs["action"]
    4. obs["actions"]
    5. payload["actions"][step_index]
    6. payload["action"][step_index]
    """

    action = None

    # --------------------------------------------------------------
    # 1-2. Action a livello dello step
    # --------------------------------------------------------------

    if isinstance(step, dict):

        if "action" in step:
            action = step["action"]

        elif "actions" in step:
            action = step["actions"]

    # --------------------------------------------------------------
    # 3-4. Eventualmente dentro obs
    # --------------------------------------------------------------

    if action is None and isinstance(obs, dict):

        if "action" in obs:
            action = obs["action"]

        elif "actions" in obs:
            action = obs["actions"]

    # --------------------------------------------------------------
    # 5-6. Eventuale array globale nel payload
    # --------------------------------------------------------------

    if action is None and isinstance(payload, dict):

        actions = None

        if "actions" in payload:
            actions = payload["actions"]

        elif "action" in payload:
            actions = payload["action"]

        if actions is not None:
            try:
                action = actions[step_index]
            except (IndexError, TypeError, KeyError):
                pass

    return action


def get_action_gripper(action):
    """
    Restituisce l'ultimo elemento del vettore action.

    Se:
        action = [dx, dy, dz, ..., gripper]

    allora:
        action_gripper = action[-1]
    """

    if action is None:
        return None

    try:
        array = np.asarray(action)

        if array.size == 0:
            return None

        value = array.reshape(-1)[-1]

        if isinstance(value, np.generic):
            value = value.item()

        return value

    except Exception:
        return None


# ==================================================================
# Estrazione pose + gripper + action
# ==================================================================

def extract_poses(traj, payload):

    trajectory_json = []

    for step_index in range(len(traj)):

        step = get_step(traj, step_index)

        # Le osservazioni sono contenute sotto "obs"
        obs = step.get("obs", step)

        eef_pos = obs.get("eef_pos")
        eef_quat = obs.get("eef_quat")
        gripper = obs.get("gripper_qpos")

        # ----------------------------------------------------------
        # Recupera l'action
        # ----------------------------------------------------------

        action = get_action(
            step=step,
            obs=obs,
            payload=payload,
            step_index=step_index,
        )

        # Ultima componente dell'action
        action_gripper = get_action_gripper(action)

        trajectory_json.append({
            "step": step_index,

            "eef_pos": to_json(eef_pos),
            "eef_quat": to_json(eef_quat),

            "gripper": to_json(gripper),

            "action": to_json(action),
            "action_gripper": to_json(action_gripper),
        })

    return trajectory_json


# ==================================================================
# Nome output
# traj000.pkl -> traj_000.json
# ==================================================================

def make_output_filename(pkl_path):

    stem = pkl_path.stem

    match = re.fullmatch(r"traj_?(\d+)", stem)

    if match:
        trajectory_number = int(match.group(1))

        return f"traj_{trajectory_number:03d}.json"

    return f"{stem}.json"


# ==================================================================
# Conversione di un singolo PKL
# ==================================================================

def convert_pkl(pkl_path, output_path):

    payload = load_pickle(pkl_path)

    # Il pickle può contenere direttamente Trajectory
    # oppure:
    #
    # {
    #     "traj": Trajectory(...),
    #     "actions": ...
    # }
    #
    if isinstance(payload, dict) and "traj" in payload:
        traj = payload["traj"]
    else:
        traj = payload

    trajectory_data = extract_poses(
        traj=traj,
        payload=payload,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            trajectory_data,
            f,
            indent=4,
            ensure_ascii=False
        )

    return len(traj)


# ==================================================================
# Main
# ==================================================================

def main():

    print(f"Input root : {INPUT_ROOT}")
    print(f"Output root: {OUTPUT_ROOT}")
    print()

    if not INPUT_ROOT.exists():
        raise FileNotFoundError(
            f"Directory input non trovata: {INPUT_ROOT}"
        )

    # Cerca ricorsivamente tutti i .pkl
    pkl_files = sorted(INPUT_ROOT.rglob("*.pkl"))

    print(f"Trovati {len(pkl_files)} file PKL")
    print()

    converted = 0
    errors = 0

    for pkl_path in pkl_files:

        relative_parent = pkl_path.parent.relative_to(INPUT_ROOT)

        output_directory = OUTPUT_ROOT / relative_parent

        output_filename = make_output_filename(pkl_path)

        output_path = output_directory / output_filename

        print(f"{pkl_path}")
        print(f"  -> {output_path}")

        try:

            num_steps = convert_pkl(
                pkl_path,
                output_path
            )

            print(f"  OK - {num_steps} step")

            converted += 1

        except Exception as exc:

            print(
                f"  ERRORE: "
                f"{type(exc).__name__}: {exc}"
            )

            errors += 1

        print()

    print("=" * 60)
    print("Conversione completata")
    print(f"PKL trovati : {len(pkl_files)}")
    print(f"Convertiti  : {converted}")
    print(f"Errori      : {errors}")
    print("=" * 60)


if __name__ == "__main__":
    main()