#!/usr/bin/env python3

import pickle
import re
from pathlib import Path
import sys

import numpy as np


# ==================================================================
# PATH
# ==================================================================

INPUT_ROOT = Path(
    "/home/asus-mivia/Desktop/dataset/pick_place/"
    "real_eye_in_hand_ur5e_pick_place"
)

OUTPUT_ROOT = Path(
    "/home/asus-mivia/Desktop/Multi-Task-LFD/repo/osvi-wm/"
    "dataset_clean"
)

OLD_GRIPPER_VALUE = 124
OLD_GRIPPER_VALUE_2 = 123
NEW_GRIPPER_VALUE = 3
NEW_ACTION_GRIPPER = 0

# ==================================================================
# PATH ALLA REPOSITORY UR5
# ==================================================================

UR5_REPO = Path(
    "/home/asus-mivia/Desktop/UR-Control/UR5e-2f-85"
)


def add_import_path(path):
    path = Path(path).expanduser()

    if path.is_dir() and str(path) not in sys.path:
        sys.path.insert(0, str(path))


def add_ur5_repo_paths():

    add_import_path(
        UR5_REPO
        / "dataset_collector"
        / "dataset_collector_pkg"
    )

    add_import_path(
        UR5_REPO
        / "dataset_collector"
        / "dataset_collector_pkg"
        / "scripts"
    )

    add_import_path(
        UR5_REPO
    )


add_ur5_repo_paths()


# ==================================================================
# Recupero della classe Trajectory originale
# ==================================================================

def resolve_trajectory_class():

    for module_name in (
        "savers",
        "scripts.savers",
        "hem.datasets.savers.trajectory",
    ):
        try:
            module = __import__(
                module_name,
                fromlist=["Trajectory"]
            )

            return getattr(module, "Trajectory")

        except Exception:
            continue

    raise ImportError(
        "Non riesco a importare la classe originale Trajectory. "
        "Esegui lo script nell'ambiente in cui è disponibile savers.py."
    )


TrajectoryClass = resolve_trajectory_class()


class CompatUnpickler(pickle.Unpickler):

    def find_class(self, module, name):

        if name == "Trajectory":
            return TrajectoryClass

        return super().find_class(module, name)


def load_pickle(path):

    with path.open("rb") as f:
        return CompatUnpickler(f).load()


# ==================================================================
# Utility
# ==================================================================

def get_scalar(value):
    """
    Converte:
        124
        np.int64(124)
        [124]
        np.array([124])

    nel valore scalare 124.
    """

    if isinstance(value, np.ndarray):

        if value.size == 1:
            return value.reshape(-1)[0].item()

    if isinstance(value, (list, tuple)):

        if len(value) == 1:
            return get_scalar(value[0])

    if isinstance(value, np.generic):
        return value.item()

    return value


# ==================================================================
# Modifica gripper_qpos mantenendo tipo e shape
# ==================================================================

def set_gripper_qpos(obs, new_value):

    value = obs["gripper_qpos"]

    # NumPy array
    if isinstance(value, np.ndarray):

        # modifica in-place
        value.reshape(-1)[0] = new_value
        return

    # Lista
    if isinstance(value, list):

        value[0] = new_value
        return

    # Tupla: essendo immutabile va sostituita
    if isinstance(value, tuple):

        tmp = list(value)
        tmp[0] = new_value

        obs["gripper_qpos"] = tuple(tmp)
        return

    # NumPy scalar
    if isinstance(value, np.generic):

        obs["gripper_qpos"] = type(value)(new_value)
        return

    # Scalar Python
    obs["gripper_qpos"] = new_value


# ==================================================================
# Modifica action[-1]
# ==================================================================

def get_action_last(action):

    if action is None:
        return None

    if isinstance(action, np.ndarray):

        if action.size == 0:
            return None

        return action.reshape(-1)[-1].item()

    if isinstance(action, (list, tuple)):

        if not action:
            return None

        value = action[-1]

        if isinstance(value, np.generic):
            return value.item()

        return value

    return None


def set_action_last(action, new_value):
    """
    Modifica SOLO action[-1].
    """

    if isinstance(action, np.ndarray):

        action.reshape(-1)[-1] = new_value
        return action

    if isinstance(action, list):

        action[-1] = new_value
        return action

    if isinstance(action, tuple):

        # tuple immutabile: bisogna crearne una nuova
        tmp = list(action)
        tmp[-1] = new_value

        return tuple(tmp)

    raise TypeError(
        f"Tipo action non gestito: {type(action)}"
    )


# ==================================================================
# Modifica esclusivamente l'ultimo step
# ==================================================================

def modify_last_step(traj):

    if not hasattr(traj, "_data"):
        raise AttributeError(
            "Trajectory non contiene l'attributo _data."
        )

    if len(traj._data) == 0:
        raise ValueError("Trajectory vuota.")

    last_index = len(traj._data) - 1

    step = traj._data[last_index]

    if not isinstance(step, (tuple, list)):
        raise TypeError(
            f"Formato step non riconosciuto: {type(step)}"
        )

    if len(step) < 5:
        raise ValueError(
            f"Lo step contiene solo {len(step)} elementi."
        )

    # Struttura dello step:
    # obs, reward, done, info, action
    obs = step[0]
    action = step[4]

    if not isinstance(obs, dict):
        raise TypeError(
            f"obs non è un dizionario: {type(obs)}"
        )

    if "gripper_qpos" not in obs:
        raise KeyError(
            "'gripper_qpos' non trovato nell'ultimo step."
        )

    # ==============================================================
    # Valori originali
    # ==============================================================

    old_gripper = get_scalar(
        obs["gripper_qpos"]
    )

    old_action_last = get_action_last(
        action
    )

    gripper_modified = False
    action_modified = False

    # ==============================================================
    # 1. Se gripper_qpos è 123 oppure 124 -> 3
    # ==============================================================

    if old_gripper in (
        OLD_GRIPPER_VALUE,
        OLD_GRIPPER_VALUE_2,
    ):

        set_gripper_qpos(
            obs,
            NEW_GRIPPER_VALUE
        )

        gripper_modified = True

    # ==============================================================
    # 2. Se action[-1] == 1 -> 0
    #
    # Questa modifica è INDIPENDENTE dal valore di gripper_qpos.
    # ==============================================================

    if old_action_last == 1:

        new_action = set_action_last(
            action,
            NEW_ACTION_GRIPPER
        )

        # Se action era una tuple, la funzione crea una nuova tuple.
        # Bisogna quindi sostituirla nello step.
        if new_action is not action:

            if isinstance(step, tuple):

                step_list = list(step)
                step_list[4] = new_action

                traj._data[last_index] = tuple(
                    step_list
                )

            else:

                step[4] = new_action

        action_modified = True

    # ==============================================================
    # Rileggi lo step finale dopo le eventuali modifiche
    # ==============================================================

    final_step = traj._data[last_index]

    new_gripper = get_scalar(
        final_step[0]["gripper_qpos"]
    )

    new_action_last = get_action_last(
        final_step[4]
    )

    # La traiettoria risulta modificata se è cambiato
    # almeno uno dei due valori.
    modified = (
        gripper_modified
        or action_modified
    )

    return {
        "modified": modified,

        "gripper_modified": gripper_modified,
        "action_modified": action_modified,

        "step": last_index,

        "old_gripper": old_gripper,
        "new_gripper": new_gripper,

        "old_action": old_action_last,
        "new_action": new_action_last,
    }


# ==================================================================
# Nome output
#
# traj000.pkl  -> traj_000.pkl
# traj_000.pkl -> traj_000.pkl
# ==================================================================

def make_output_filename(path):

    match = re.fullmatch(
        r"traj_?(\d+)",
        path.stem
    )

    if match:

        number = int(
            match.group(1)
        )

        return f"traj_{number:03d}.pkl"

    return path.name


# ==================================================================
# Processa un singolo PKL
# ==================================================================

def process_pkl(input_path, output_path):

    # --------------------------------------------------------------
    # Deserializza TUTTO il payload
    # --------------------------------------------------------------

    payload = load_pickle(
        input_path
    )

    # --------------------------------------------------------------
    # Recupera esclusivamente il riferimento a Trajectory
    # --------------------------------------------------------------

    if (
        isinstance(payload, dict)
        and "traj" in payload
    ):
        traj = payload["traj"]

    else:
        traj = payload

    # --------------------------------------------------------------
    # Modifica esclusivamente l'ultimo step
    # --------------------------------------------------------------

    result = modify_last_step(
        traj
    )

    # --------------------------------------------------------------
    # Crea directory
    # --------------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------------
    # Risalva TUTTO il payload
    #
    # Non costruiamo un nuovo payload.
    # È lo stesso oggetto caricato dal PKL.
    # --------------------------------------------------------------

    with output_path.open("wb") as f:

        pickle.dump(
            payload,
            f,
            protocol=pickle.HIGHEST_PROTOCOL
        )

    return result


# ==================================================================
# MAIN
# ==================================================================

def main():

    print(f"INPUT : {INPUT_ROOT}")
    print(f"OUTPUT: {OUTPUT_ROOT}")
    print()

    if not INPUT_ROOT.exists():

        raise FileNotFoundError(
            INPUT_ROOT
        )

    pkl_files = sorted(
        INPUT_ROOT.rglob("*.pkl")
    )

    print(
        f"Trovati {len(pkl_files)} PKL\n"
    )

    processed = 0
    modified = 0
    unchanged = 0
    errors = 0

    for input_path in pkl_files:

        # Mantiene:
        #
        # task_00/
        # task_01/
        # ...
        #
        relative_parent = (
            input_path
            .parent
            .relative_to(INPUT_ROOT)
        )

        output_directory = (
            OUTPUT_ROOT
            / relative_parent
        )

        output_filename = (
            make_output_filename(
                input_path
            )
        )

        output_path = (
            output_directory
            / output_filename
        )

        print(
            f"{relative_parent}/{input_path.name}"
        )

        try:

            result = process_pkl(
                input_path,
                output_path
            )

            processed += 1

            if result["modified"]:

                modified += 1

                print(
                    f"  ultimo step: "
                    f"{result['step']}"
                )

                print(
                    f"  gripper_qpos: "
                    f"{result['old_gripper']} "
                    f"-> {result['new_gripper']}"
                )

                print(
                    f"  action[-1]: "
                    f"{result['old_action']} "
                    f"-> {result['new_action']}"
                )

            else:

                unchanged += 1

                print(
                    f"  invariato: "
                    f"gripper finale = "
                    f"{result['old_gripper']}"
                )

            print(
                f"  salvato: {output_path}"
            )

        except Exception as exc:

            errors += 1

            print(
                f"  ERRORE: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

        print()

    # ==================================================================
    # Riepilogo
    # ==================================================================

    print("=" * 70)
    print("DATASET CLEAN CREATO")
    print("=" * 70)

    print(
        f"PKL trovati       : {len(pkl_files)}"
    )

    print(
        f"PKL processati    : {processed}"
    )

    print(
        f"Modificati        : {modified}"
    )

    print(
        f"Lasciati invariati: {unchanged}"
    )

    print(
        f"Errori             : {errors}"
    )

    print()

    print(
        f"Output:\n{OUTPUT_ROOT}"
    )


if __name__ == "__main__":
    main()