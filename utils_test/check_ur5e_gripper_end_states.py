#!/usr/bin/env python3

import json
from pathlib import Path


# ================================================================
# CONFIGURAZIONE
# ================================================================

ROOT_DIR = Path(
    "/home/asus-mivia/Desktop/Multi-Task-LFD/repo/osvi-wm/dataset_clean_json"
)

OUTPUT_FILE = ROOT_DIR / "analisi.json"

TARGET_ACTION_GRIPPER = 0


# ================================================================
# Utility
# ================================================================

def get_action_gripper(action):
    """
    Restituisce l'ultimo elemento del vettore action.

    Esempio:

        "action": [0.01, -0.02, 0.0, 0.0]

    restituisce:

        0.0
    """

    if action is None:
        return None

    if isinstance(action, list):
        if len(action) == 0:
            return None

        return action[-1]

    return None


# ================================================================
# Main
# ================================================================

def main():

    if not ROOT_DIR.exists():
        raise FileNotFoundError(
            f"Directory non trovata: {ROOT_DIR}"
        )

    trajectories_found = []

    count_per_task = {}

    total_analyzed = 0
    total_matching = 0

    # ------------------------------------------------------------
    # Cerca tutte le directory task_XX
    # ------------------------------------------------------------

    task_dirs = sorted(
        directory
        for directory in ROOT_DIR.iterdir()
        if directory.is_dir()
        and directory.name.startswith("task_")
    )

    print(f"Trovati {len(task_dirs)} task\n")

    # ------------------------------------------------------------
    # Analizza ciascun task
    # ------------------------------------------------------------

    for task_dir in task_dirs:

        task_name = task_dir.name

        count_per_task[task_name] = 0

        trajectory_files = sorted(
            task_dir.glob("traj_*.json")
        )

        print(
            f"{task_name}: "
            f"{len(trajectory_files)} traiettorie"
        )

        for trajectory_path in trajectory_files:

            total_analyzed += 1

            try:

                with trajectory_path.open(
                    "r",
                    encoding="utf-8"
                ) as f:
                    trajectory = json.load(f)

                # ------------------------------------------------
                # File vuoto
                # ------------------------------------------------

                if not trajectory:
                    print(
                        f"  ATTENZIONE: "
                        f"{trajectory_path.name} è vuoto"
                    )
                    continue

                # ------------------------------------------------
                # Ultimo step
                # ------------------------------------------------

                last_step = trajectory[-1]

                action = last_step.get("action")

                action_gripper = get_action_gripper(
                    action
                )

                # ------------------------------------------------
                # Controllo action[-1] == 0
                # ------------------------------------------------

                if action_gripper == TARGET_ACTION_GRIPPER:

                    trajectories_found.append({
                        "task": task_name,
                        "trajectory": trajectory_path.name,
                        "last_step": last_step.get("step"),
                        "action_gripper": action_gripper,
                    })

                    count_per_task[task_name] += 1
                    total_matching += 1

                    print(
                        f"  MATCH: "
                        f"{trajectory_path.name} "
                        f"(step={last_step.get('step')}, "
                        f"action[-1]={action_gripper})"
                    )

            except Exception as exc:

                print(
                    f"  ERRORE {trajectory_path.name}: "
                    f"{type(exc).__name__}: {exc}"
                )

    # ============================================================
    # Costruzione analisi.json
    # ============================================================

    analysis = {

        "target_action_gripper": TARGET_ACTION_GRIPPER,

        "trajectories": trajectories_found,

        "count_per_task": count_per_task,

        "total_matching_trajectories": total_matching,

        "total_analyzed_trajectories": total_analyzed,
    }

    # ------------------------------------------------------------
    # Salvataggio
    # ------------------------------------------------------------

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            analysis,
            f,
            indent=4,
            ensure_ascii=False
        )

    # ============================================================
    # Riepilogo terminale
    # ============================================================

    print("\n" + "=" * 60)
    print("RISULTATI")
    print("=" * 60)

    for task_name, count in count_per_task.items():

        print(
            f"{task_name}: "
            f"{count} traiettorie con action[-1] finale = "
            f"{TARGET_ACTION_GRIPPER}"
        )

    print("-" * 60)

    print(
        f"Totale traiettorie con action[-1] finale = "
        f"{TARGET_ACTION_GRIPPER}: {total_matching}"
    )

    print(
        f"Totale traiettorie analizzate: "
        f"{total_analyzed}"
    )

    print()

    print("Analisi salvata in:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()