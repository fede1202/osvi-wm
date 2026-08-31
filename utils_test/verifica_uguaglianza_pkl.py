#!/usr/bin/env python3

import json
import math
import pickle
import re
import sys
from pathlib import Path

import numpy as np


# ==================================================================
# PATH
# ==================================================================

ORIGINAL_ROOT = Path(
    "/home/asus-mivia/Desktop/dataset/pick_place/"
    "real_eye_in_hand_ur5e_pick_place"
)

CLEAN_ROOT = Path(
    "/home/asus-mivia/Desktop/Multi-Task-LFD/repo/osvi-wm/"
    "dataset_clean"
)

REPORT_FILE = CLEAN_ROOT / "verification_report.json"


# ==================================================================
# Modifiche che CI ASPETTIAMO
# ==================================================================

OLD_GRIPPER_VALUES = {123, 124}
NEW_GRIPPER_VALUE = 3

OLD_ACTION_VALUE = 1
NEW_ACTION_VALUE = 0


# ==================================================================
# Repository contenente savers.py
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
# Classe Trajectory
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

            return getattr(
                module,
                "Trajectory"
            )

        except Exception:
            continue

    raise ImportError(
        "Non riesco a importare Trajectory."
    )


TrajectoryClass = resolve_trajectory_class()


class CompatUnpickler(pickle.Unpickler):

    def find_class(self, module, name):

        if name == "Trajectory":
            return TrajectoryClass

        return super().find_class(
            module,
            name
        )


def load_pickle(path):

    with path.open("rb") as f:

        return CompatUnpickler(
            f
        ).load()


# ==================================================================
# Utility differenze
# ==================================================================

MAX_REPORTED_DIFFERENCES = 100


def add_difference(result, path, message):

    result["difference_count"] += 1

    if len(result["differences"]) < MAX_REPORTED_DIFFERENCES:

        result["differences"].append({
            "path": path,
            "message": message,
        })


# ==================================================================
# Uguaglianza scalare
# ==================================================================

def scalar_equal(a, b):

    # Gestione NaN
    try:

        if (
            isinstance(a, (float, np.floating))
            and isinstance(b, (float, np.floating))
            and math.isnan(a)
            and math.isnan(b)
        ):
            return True

    except Exception:
        pass

    try:
        return bool(a == b)

    except Exception:
        return False


def scalar_value(value):

    if isinstance(value, np.ndarray):

        if value.size == 1:
            return value.reshape(-1)[0].item()

    if isinstance(value, (list, tuple)):

        if len(value) == 1:
            return scalar_value(value[0])

    if isinstance(value, np.generic):
        return value.item()

    return value


# ==================================================================
# CONFRONTO RICORSIVO GENERALE
# ==================================================================

def compare_exact(original, clean, path, result):
    """
    Confronto ESATTO.

    Nessuna tolleranza numerica.
    """

    # --------------------------------------------------------------
    # NumPy ndarray
    # --------------------------------------------------------------

    if isinstance(original, np.ndarray):

        result["arrays_compared"] += 1
        result["array_bytes_compared"] += original.nbytes

        if not isinstance(clean, np.ndarray):

            add_difference(
                result,
                path,
                f"tipo differente: ndarray vs {type(clean)}"
            )
            return

        if original.shape != clean.shape:

            add_difference(
                result,
                path,
                f"shape differente: "
                f"{original.shape} vs {clean.shape}"
            )
            return

        if original.dtype != clean.dtype:

            add_difference(
                result,
                path,
                f"dtype differente: "
                f"{original.dtype} vs {clean.dtype}"
            )
            return

        if not np.array_equal(
            original,
            clean,
            equal_nan=True
        ):

            add_difference(
                result,
                path,
                "contenuto ndarray differente"
            )

        return

    # --------------------------------------------------------------
    # bytes
    # --------------------------------------------------------------

    if isinstance(original, (bytes, bytearray)):

        result["bytes_objects_compared"] += 1
        result["raw_bytes_compared"] += len(original)

        if type(original) is not type(clean):

            add_difference(
                result,
                path,
                f"tipo bytes differente: "
                f"{type(original)} vs {type(clean)}"
            )
            return

        if original != clean:

            add_difference(
                result,
                path,
                f"contenuto binario differente "
                f"({len(original)} byte)"
            )

        return

    # --------------------------------------------------------------
    # NumPy scalar
    # --------------------------------------------------------------

    if isinstance(original, np.generic):

        if not isinstance(clean, np.generic):

            add_difference(
                result,
                path,
                f"tipo differente: "
                f"{type(original)} vs {type(clean)}"
            )
            return

        if original.dtype != clean.dtype:

            add_difference(
                result,
                path,
                f"dtype scalare differente"
            )
            return

        if not scalar_equal(
            original.item(),
            clean.item()
        ):

            add_difference(
                result,
                path,
                f"valore differente: "
                f"{original.item()} vs {clean.item()}"
            )

        return

    # --------------------------------------------------------------
    # Dizionari
    # --------------------------------------------------------------

    if isinstance(original, dict):

        if not isinstance(clean, dict):

            add_difference(
                result,
                path,
                f"tipo differente: dict vs {type(clean)}"
            )
            return

        original_keys = set(original.keys())
        clean_keys = set(clean.keys())

        if original_keys != clean_keys:

            missing = original_keys - clean_keys
            extra = clean_keys - original_keys

            add_difference(
                result,
                path,
                f"chiavi differenti. "
                f"mancanti={list(missing)}, "
                f"extra={list(extra)}"
            )

        for key in original_keys & clean_keys:

            compare_exact(
                original[key],
                clean[key],
                f"{path}[{key!r}]",
                result
            )

        return

    # --------------------------------------------------------------
    # Liste / tuple
    # --------------------------------------------------------------

    if isinstance(original, (list, tuple)):

        if type(original) is not type(clean):

            add_difference(
                result,
                path,
                f"tipo differente: "
                f"{type(original)} vs {type(clean)}"
            )
            return

        if len(original) != len(clean):

            add_difference(
                result,
                path,
                f"lunghezza differente: "
                f"{len(original)} vs {len(clean)}"
            )
            return

        for index, (a, b) in enumerate(
            zip(original, clean)
        ):

            compare_exact(
                a,
                b,
                f"{path}[{index}]",
                result
            )

        return

    # --------------------------------------------------------------
    # Oggetti custom
    # --------------------------------------------------------------

    if hasattr(original, "__dict__"):

        if type(original) is not type(clean):

            add_difference(
                result,
                path,
                f"classe differente: "
                f"{type(original)} vs {type(clean)}"
            )
            return

        compare_exact(
            vars(original),
            vars(clean),
            f"{path}.__dict__",
            result
        )

        return

    # --------------------------------------------------------------
    # Tipi standard
    # --------------------------------------------------------------

    if type(original) is not type(clean):

        add_difference(
            result,
            path,
            f"tipo differente: "
            f"{type(original).__name__} vs "
            f"{type(clean).__name__}"
        )
        return

    if not scalar_equal(
        original,
        clean
    ):

        add_difference(
            result,
            path,
            f"valore differente: "
            f"{original!r} vs {clean!r}"
        )


# ==================================================================
# Confronto SPECIALE gripper finale
# ==================================================================

def compare_final_gripper(
    original,
    clean,
    path,
    result
):

    original_scalar = scalar_value(
        original
    )

    expected = (
        NEW_GRIPPER_VALUE
        if original_scalar in OLD_GRIPPER_VALUES
        else original_scalar
    )

    # --------------------------------------------------------------
    # Mantieni stessa struttura
    # --------------------------------------------------------------

    if type(original) is not type(clean):

        add_difference(
            result,
            path,
            f"tipo gripper modificato: "
            f"{type(original)} vs {type(clean)}"
        )
        return

    # NumPy
    if isinstance(original, np.ndarray):

        if original.shape != clean.shape:

            add_difference(
                result,
                path,
                "shape gripper modificata"
            )
            return

        if original.dtype != clean.dtype:

            add_difference(
                result,
                path,
                "dtype gripper modificato"
            )
            return

        if clean.size != 1:

            add_difference(
                result,
                path,
                "gripper non è scalare"
            )
            return

        clean_scalar = clean.reshape(-1)[0].item()

    elif isinstance(original, (list, tuple)):

        if len(original) != len(clean):

            add_difference(
                result,
                path,
                "lunghezza gripper modificata"
            )
            return

        if len(clean) != 1:

            add_difference(
                result,
                path,
                "gripper non contiene un solo valore"
            )
            return

        clean_scalar = scalar_value(
            clean
        )

    else:

        clean_scalar = scalar_value(
            clean
        )

    # --------------------------------------------------------------
    # Controllo trasformazione
    # --------------------------------------------------------------

    if not scalar_equal(
        clean_scalar,
        expected
    ):

        add_difference(
            result,
            path,
            f"gripper finale errato: "
            f"originale={original_scalar}, "
            f"atteso={expected}, "
            f"clean={clean_scalar}"
        )

    if original_scalar in OLD_GRIPPER_VALUES:

        result["expected_gripper_changes"] += 1


# ==================================================================
# Confronto SPECIALE action finale
# ==================================================================

def compare_final_action(
    original,
    clean,
    path,
    result
):

    # --------------------------------------------------------------
    # NumPy action
    # --------------------------------------------------------------

    if isinstance(original, np.ndarray):

        if not isinstance(clean, np.ndarray):

            add_difference(
                result,
                path,
                "action non è più ndarray"
            )
            return

        if original.shape != clean.shape:

            add_difference(
                result,
                path,
                f"shape action differente: "
                f"{original.shape} vs {clean.shape}"
            )
            return

        if original.dtype != clean.dtype:

            add_difference(
                result,
                path,
                f"dtype action differente: "
                f"{original.dtype} vs {clean.dtype}"
            )
            return

        original_flat = original.reshape(-1)
        clean_flat = clean.reshape(-1)

        if original_flat.size == 0:
            return

        # Tutto tranne l'ultimo elemento deve essere IDENTICO
        if not np.array_equal(
            original_flat[:-1],
            clean_flat[:-1],
            equal_nan=True
        ):

            add_difference(
                result,
                path,
                "elementi di action diversi prima di action[-1]"
            )

        original_last = original_flat[-1].item()
        clean_last = clean_flat[-1].item()

    # --------------------------------------------------------------
    # Lista / tuple
    # --------------------------------------------------------------

    elif isinstance(original, (list, tuple)):

        if type(original) is not type(clean):

            add_difference(
                result,
                path,
                "tipo action modificato"
            )
            return

        if len(original) != len(clean):

            add_difference(
                result,
                path,
                "lunghezza action modificata"
            )
            return

        if len(original) == 0:
            return

        # Confronto ogni elemento tranne l'ultimo
        for i in range(len(original) - 1):

            compare_exact(
                original[i],
                clean[i],
                f"{path}[{i}]",
                result
            )

        original_last = scalar_value(
            original[-1]
        )

        clean_last = scalar_value(
            clean[-1]
        )

    else:

        compare_exact(
            original,
            clean,
            path,
            result
        )
        return

    # --------------------------------------------------------------
    # Cosa DEVE essere action[-1]
    # --------------------------------------------------------------

    expected = (
        NEW_ACTION_VALUE
        if scalar_equal(
            original_last,
            OLD_ACTION_VALUE
        )
        else original_last
    )

    if not scalar_equal(
        clean_last,
        expected
    ):

        add_difference(
            result,
            f"{path}[-1]",
            f"action[-1] errata: "
            f"originale={original_last}, "
            f"atteso={expected}, "
            f"clean={clean_last}"
        )

    if scalar_equal(
        original_last,
        OLD_ACTION_VALUE
    ):

        result["expected_action_changes"] += 1


# ==================================================================
# Confronto OBS dell'ultimo step
# ==================================================================

def compare_final_obs(
    original,
    clean,
    path,
    result
):

    if not isinstance(original, dict) or not isinstance(clean, dict):

        compare_exact(
            original,
            clean,
            path,
            result
        )
        return

    original_keys = set(
        original.keys()
    )

    clean_keys = set(
        clean.keys()
    )

    if original_keys != clean_keys:

        add_difference(
            result,
            path,
            f"chiavi obs differenti"
        )

    for key in original_keys & clean_keys:

        if key == "gripper_qpos":

            compare_final_gripper(
                original[key],
                clean[key],
                f"{path}['gripper_qpos']",
                result
            )

        else:

            compare_exact(
                original[key],
                clean[key],
                f"{path}[{key!r}]",
                result
            )


# ==================================================================
# Confronto ultimo step
# ==================================================================

def compare_final_step(
    original,
    clean,
    path,
    result
):

    if type(original) is not type(clean):

        add_difference(
            result,
            path,
            "tipo ultimo step differente"
        )
        return

    if not isinstance(
        original,
        (tuple, list)
    ):

        compare_exact(
            original,
            clean,
            path,
            result
        )
        return

    if len(original) != len(clean):

        add_difference(
            result,
            path,
            "numero campi ultimo step differente"
        )
        return

    # --------------------------------------------------------------
    # obs = indice 0
    # --------------------------------------------------------------

    compare_final_obs(
        original[0],
        clean[0],
        f"{path}[0]",
        result
    )

    # --------------------------------------------------------------
    # reward, done, info ecc.
    # Devono essere IDENTICI
    # --------------------------------------------------------------

    for index in range(
        1,
        len(original)
    ):

        # action = indice 4
        if index == 4:

            compare_final_action(
                original[index],
                clean[index],
                f"{path}[4]",
                result
            )

        else:

            compare_exact(
                original[index],
                clean[index],
                f"{path}[{index}]",
                result
            )


# ==================================================================
# Confronto Trajectory
# ==================================================================

def compare_trajectory(
    original,
    clean,
    result
):

    if type(original) is not type(clean):

        add_difference(
            result,
            "traj",
            f"classe Trajectory differente"
        )
        return

    original_attrs = vars(
        original
    )

    clean_attrs = vars(
        clean
    )

    original_keys = set(
        original_attrs.keys()
    )

    clean_keys = set(
        clean_attrs.keys()
    )

    if original_keys != clean_keys:

        add_difference(
            result,
            "traj",
            f"attributi Trajectory differenti. "
            f"original={original_keys}, "
            f"clean={clean_keys}"
        )

    # --------------------------------------------------------------
    # Tutti gli attributi eccetto _data
    # devono essere IDENTICI
    # --------------------------------------------------------------

    for key in (
        original_keys
        & clean_keys
    ):

        if key == "_data":
            continue

        compare_exact(
            original_attrs[key],
            clean_attrs[key],
            f"traj.{key}",
            result
        )

    # --------------------------------------------------------------
    # _data
    # --------------------------------------------------------------

    if "_data" not in original_attrs:
        return

    if "_data" not in clean_attrs:

        add_difference(
            result,
            "traj._data",
            "_data assente nel clean"
        )
        return

    original_data = original_attrs["_data"]
    clean_data = clean_attrs["_data"]

    if len(original_data) != len(clean_data):

        add_difference(
            result,
            "traj._data",
            f"numero step differente: "
            f"{len(original_data)} vs {len(clean_data)}"
        )
        return

    if len(original_data) == 0:
        return

    last_index = len(
        original_data
    ) - 1

    # --------------------------------------------------------------
    # TUTTI gli step precedenti devono essere identici
    # --------------------------------------------------------------

    for index in range(last_index):

        compare_exact(
            original_data[index],
            clean_data[index],
            f"traj._data[{index}]",
            result
        )

    # --------------------------------------------------------------
    # Ultimo step: consentiamo solo le due modifiche previste
    # --------------------------------------------------------------

    compare_final_step(
        original_data[last_index],
        clean_data[last_index],
        f"traj._data[{last_index}]",
        result
    )


# ==================================================================
# Confronto payload completo
# ==================================================================

def compare_payload(
    original,
    clean,
    result
):

    original_is_dict = (
        isinstance(original, dict)
        and "traj" in original
    )

    clean_is_dict = (
        isinstance(clean, dict)
        and "traj" in clean
    )

    # --------------------------------------------------------------
    # Payload dict {"traj": ..., metadata...}
    # --------------------------------------------------------------

    if original_is_dict:

        if not clean_is_dict:

            add_difference(
                result,
                "payload",
                "struttura payload differente"
            )
            return

        original_keys = set(
            original.keys()
        )

        clean_keys = set(
            clean.keys()
        )

        if original_keys != clean_keys:

            add_difference(
                result,
                "payload",
                f"chiavi payload differenti"
            )

        # Tutti i metadata devono essere identici
        for key in (
            original_keys
            & clean_keys
        ):

            if key == "traj":
                continue

            compare_exact(
                original[key],
                clean[key],
                f"payload[{key!r}]",
                result
            )

        compare_trajectory(
            original["traj"],
            clean["traj"],
            result
        )

    # --------------------------------------------------------------
    # Payload direttamente Trajectory
    # --------------------------------------------------------------

    else:

        compare_trajectory(
            original,
            clean,
            result
        )


# ==================================================================
# Normalizzazione nome traiettoria
# ==================================================================

def trajectory_key(path, root):

    relative_parent = (
        path.parent
        .relative_to(root)
    )

    match = re.fullmatch(
        r"traj_?(\d+)",
        path.stem
    )

    if match:

        trajectory_name = (
            f"traj_{int(match.group(1)):03d}"
        )

    else:

        trajectory_name = path.stem

    return (
        str(relative_parent),
        trajectory_name
    )


def index_pickles(root):

    result = {}

    for path in root.rglob("*.pkl"):

        key = trajectory_key(
            path,
            root
        )

        result[key] = path

    return result


# ==================================================================
# MAIN
# ==================================================================

def main():

    print(f"ORIGINAL: {ORIGINAL_ROOT}")
    print(f"CLEAN   : {CLEAN_ROOT}")
    print()

    original_files = index_pickles(
        ORIGINAL_ROOT
    )

    clean_files = index_pickles(
        CLEAN_ROOT
    )

    original_keys = set(
        original_files.keys()
    )

    clean_keys = set(
        clean_files.keys()
    )

    missing_files = sorted(
        original_keys - clean_keys
    )

    extra_files = sorted(
        clean_keys - original_keys
    )

    common_files = sorted(
        original_keys & clean_keys
    )

    print(
        f"Originali : {len(original_files)}"
    )

    print(
        f"Clean     : {len(clean_files)}"
    )

    print(
        f"Da confrontare: {len(common_files)}"
    )

    print()

    report = {

        "original_root": str(
            ORIGINAL_ROOT
        ),

        "clean_root": str(
            CLEAN_ROOT
        ),

        "original_files": len(
            original_files
        ),

        "clean_files": len(
            clean_files
        ),

        "missing_files": [
            f"{task}/{traj}"
            for task, traj in missing_files
        ],

        "extra_files": [
            f"{task}/{traj}"
            for task, traj in extra_files
        ],

        "files": [],
    }

    files_ok = 0
    files_failed = 0

    # ==============================================================
    # Confronto file per file
    # ==============================================================

    for key in common_files:

        original_path = original_files[
            key
        ]

        clean_path = clean_files[
            key
        ]

        task, trajectory = key

        print(
            f"{task}/{trajectory}"
        )

        result = {

            "task": task,
            "trajectory": trajectory,

            "original": str(
                original_path
            ),

            "clean": str(
                clean_path
            ),

            "difference_count": 0,

            "differences": [],

            "arrays_compared": 0,

            "array_bytes_compared": 0,

            "bytes_objects_compared": 0,

            "raw_bytes_compared": 0,

            "expected_gripper_changes": 0,

            "expected_action_changes": 0,
        }

        try:

            original_payload = load_pickle(
                original_path
            )

            clean_payload = load_pickle(
                clean_path
            )

            compare_payload(
                original_payload,
                clean_payload,
                result
            )

            result["ok"] = (
                result["difference_count"] == 0
            )

            if result["ok"]:

                files_ok += 1

                print(
                    "  OK - contenuto identico "
                    "salvo modifiche previste"
                )

            else:

                files_failed += 1

                print(
                    f"  ERRORE - "
                    f"{result['difference_count']} "
                    f"differenze non consentite"
                )

                for difference in result[
                    "differences"
                ][:5]:

                    print(
                        f"    {difference['path']}: "
                        f"{difference['message']}"
                    )

        except Exception as exc:

            files_failed += 1

            result["ok"] = False

            result["difference_count"] += 1

            result["differences"].append({
                "path": "file",
                "message":
                    f"{type(exc).__name__}: {exc}"
            })

            print(
                f"  ERRORE: "
                f"{type(exc).__name__}: {exc}"
            )

        report["files"].append(
            result
        )

    # ==================================================================
    # SUMMARY
    # ==================================================================

    report["summary"] = {

        "files_ok": files_ok,

        "files_failed": files_failed,

        "missing_files": len(
            missing_files
        ),

        "extra_files": len(
            extra_files
        ),

        "all_ok": (
            files_failed == 0
            and len(missing_files) == 0
            and len(extra_files) == 0
        )
    }

    with REPORT_FILE.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=4,
            ensure_ascii=False
        )

    # ==================================================================
    # Risultato finale
    # ==================================================================

    print()
    print("=" * 70)
    print("RISULTATO VERIFICA")
    print("=" * 70)

    print(
        f"File corretti          : {files_ok}"
    )

    print(
        f"File con differenze    : {files_failed}"
    )

    print(
        f"File mancanti nel clean: {len(missing_files)}"
    )

    print(
        f"File extra nel clean   : {len(extra_files)}"
    )

    print()

    if report["summary"]["all_ok"]:

        print(
            "OK: dataset_clean coincide con il dataset originale "
            "salvo ESCLUSIVAMENTE le modifiche previste."
        )

    else:

        print(
            "ATTENZIONE: sono state trovate differenze "
            "non previste."
        )

    print()
    print(
        f"Report completo:\n{REPORT_FILE}"
    )

    # Exit code utile anche da terminale
    if not report["summary"]["all_ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()