import argparse
import csv
import json
import sys
import tkinter as tk
from collections import defaultdict
from pathlib import Path
from tkinter import messagebox


FIELDS = [
    "object_correct",
    "bin_correct",
    "trajectory_semantic_correct",
    "gripper_release_correct",
]

FIELD_LABELS = {
    "object_correct": "Oggetto giusto",
    "bin_correct": "Bin giusto",
    "trajectory_semantic_correct": "Traiettoria plausibile",
    "gripper_release_correct": "Gripper/rilascio plausibile",
}

OUTPUT_FIELDS = [
    "mode",
    "task",
    "sample_id",
    "dataset_index",
    "agent_traj_index",
    "teacher_traj_index",
    "checkpoint_epoch",
    "output_png",
    *FIELDS,
    "notes",
]


def row_key(row):
    return "|".join(
        str(row.get(field, ""))
        for field in ["mode", "task", "dataset_index", "agent_traj_index", "teacher_traj_index", "output_png"]
    )


def resolve_image_path(index_path, value):
    path = Path(value)
    if path.is_absolute():
        return path
    return (index_path.parent / path).resolve()


def load_rows(index_path):
    with index_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for i, row in enumerate(rows):
        row.setdefault("sample_id", str(i))
    return rows


def load_annotations(output_csv):
    if not output_csv.exists():
        return {}
    with output_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return {row_key(row): row for row in rows}


def write_annotations(output_csv, annotations):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = list(annotations.values())
    rows.sort(key=lambda row: (row.get("mode", ""), row.get("task", ""), int(row.get("dataset_index") or 0)))
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize_annotations(annotations):
    groups = defaultdict(list)
    for row in annotations.values():
        groups["overall"].append(row)
        groups[f"{row.get('mode')}:{row.get('task')}"].append(row)

    summary = {}
    for key, rows in groups.items():
        item = {"annotated": len(rows)}
        for field in FIELDS:
            valid = [row[field] for row in rows if row.get(field) in {"yes", "no"}]
            yes = sum(value == "yes" for value in valid)
            item[f"{field}_yes"] = yes
            item[f"{field}_no"] = len(valid) - yes
            item[f"{field}_n"] = len(valid)
            item[f"{field}_rate"] = (yes / len(valid)) if valid else None
        complete = 0
        complete_n = 0
        for row in rows:
            values = [row.get(field) for field in FIELDS]
            if all(value in {"yes", "no"} for value in values):
                complete_n += 1
                complete += int(all(value == "yes" for value in values))
        item["all_semantic_success_n"] = complete_n
        item["all_semantic_success_rate"] = (complete / complete_n) if complete_n else None
        summary[key] = item
    return summary


def write_summary(output_json, annotations):
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w") as f:
        json.dump(summarize_annotations(annotations), f, indent=2)


class AnnotatorApp:
    def __init__(self, root, args):
        self.root = root
        self.index_path = Path(args.index).expanduser().resolve()
        self.rows = load_rows(self.index_path)
        self.output_csv = Path(args.output_csv).expanduser().resolve()
        self.output_json = Path(args.output_json).expanduser().resolve()
        self.annotations = load_annotations(self.output_csv)
        self.review_all = args.review_all
        self.max_image_width = args.window_width - 40
        self.max_image_height = args.window_height - 260
        self.position = 0
        self.current_photo = None
        self.current_row = None
        self.vars = {field: tk.StringVar(value="unclear") for field in FIELDS}

        self.root.title("OSVI waypoint semantic annotator")
        self.root.geometry(f"{args.window_width}x{args.window_height}")

        self.header = tk.Label(root, text="", font=("Arial", 13, "bold"))
        self.header.pack(fill="x", padx=8, pady=(8, 2))

        self.image_label = tk.Label(root, bg="#222")
        self.image_label.pack(fill="both", expand=True, padx=8, pady=4)

        form = tk.Frame(root)
        form.pack(fill="x", padx=8, pady=4)
        for field in FIELDS:
            frame = tk.LabelFrame(form, text=FIELD_LABELS[field], padx=6, pady=4)
            frame.pack(side="left", fill="x", expand=True, padx=3)
            for label, value in [("si", "yes"), ("no", "no"), ("dubbio", "unclear")]:
                tk.Radiobutton(frame, text=label, variable=self.vars[field], value=value).pack(anchor="w")

        notes_frame = tk.Frame(root)
        notes_frame.pack(fill="x", padx=8, pady=4)
        tk.Label(notes_frame, text="Note").pack(side="left")
        self.notes = tk.Entry(notes_frame)
        self.notes.pack(side="left", fill="x", expand=True, padx=(6, 0))

        controls = tk.Frame(root)
        controls.pack(fill="x", padx=8, pady=(4, 8))
        tk.Button(controls, text="Tutto si", command=self.mark_all_yes).pack(side="left")
        tk.Button(controls, text="Salva e avanti", command=self.save_and_next).pack(side="left", padx=4)
        tk.Button(controls, text="Salta", command=self.next_row).pack(side="left", padx=4)
        tk.Button(controls, text="Indietro", command=self.previous_row).pack(side="left", padx=4)
        tk.Button(controls, text="Salva summary", command=self.save_summary).pack(side="right")

        root.bind("<Return>", lambda _event: self.save_and_next())
        root.bind("<Right>", lambda _event: self.next_row())
        root.bind("<Left>", lambda _event: self.previous_row())
        root.bind("a", lambda _event: self.mark_all_yes())
        root.bind("q", lambda _event: self.root.destroy())

        self.seek_next_unannotated()
        self.show_current()

    def seek_next_unannotated(self):
        if self.review_all:
            return
        while self.position < len(self.rows) and row_key(self.rows[self.position]) in self.annotations:
            self.position += 1

    def image_for_display(self, path):
        try:
            from PIL import Image, ImageTk

            image = Image.open(path)
            max_w = max(400, self.max_image_width)
            max_h = max(300, self.max_image_height)
            image.thumbnail((max_w, max_h))
            return ImageTk.PhotoImage(image)
        except Exception:
            return tk.PhotoImage(file=str(path))

    def show_current(self):
        if self.position >= len(self.rows):
            write_annotations(self.output_csv, self.annotations)
            write_summary(self.output_json, self.annotations)
            messagebox.showinfo("Finito", f"Annotazioni salvate:\n{self.output_csv}\n{self.output_json}")
            self.root.destroy()
            return

        row = self.rows[self.position]
        self.current_row = row
        key = row_key(row)
        previous = self.annotations.get(key, {})
        for field in FIELDS:
            self.vars[field].set(previous.get(field, "unclear") or "unclear")
        self.notes.delete(0, tk.END)
        self.notes.insert(0, previous.get("notes", ""))

        image_path = resolve_image_path(self.index_path, row["output_png"])
        self.header.config(
            text=(
                f"{self.position + 1}/{len(self.rows)} | {row.get('mode')} {row.get('task')} | "
                f"idx={row.get('dataset_index')} | agent={row.get('agent_traj_index')} | "
                f"teacher={row.get('teacher_traj_index')}"
            )
        )
        if not image_path.exists():
            self.image_label.config(text=f"Immagine non trovata:\n{image_path}", image="")
            self.current_photo = None
            return
        self.current_photo = self.image_for_display(image_path)
        self.image_label.config(image=self.current_photo, text="")

    def annotation_from_current(self):
        row = dict(self.current_row)
        out = {field: row.get(field, "") for field in OUTPUT_FIELDS}
        for field in FIELDS:
            out[field] = self.vars[field].get()
        out["notes"] = self.notes.get()
        return out

    def save_and_next(self):
        if self.current_row is None:
            return
        self.annotations[row_key(self.current_row)] = self.annotation_from_current()
        write_annotations(self.output_csv, self.annotations)
        write_summary(self.output_json, self.annotations)
        self.position += 1
        self.seek_next_unannotated()
        self.show_current()

    def next_row(self):
        self.position += 1
        self.show_current()

    def previous_row(self):
        self.position = max(0, self.position - 1)
        self.show_current()

    def mark_all_yes(self):
        for field in FIELDS:
            self.vars[field].set("yes")

    def save_summary(self):
        write_annotations(self.output_csv, self.annotations)
        write_summary(self.output_json, self.annotations)
        messagebox.showinfo("Salvato", f"Summary aggiornata:\n{self.output_json}")


def main():
    parser = argparse.ArgumentParser(description="Manual semantic annotation tool for OSVI overlay images.")
    parser.add_argument("--index", required=True, help="index.csv produced by overlay_ur5e_waypoints.py")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--review-all", action="store_true", help="Review already annotated rows too.")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--window-width", type=int, default=1500)
    parser.add_argument("--window-height", type=int, default=1000)
    args = parser.parse_args()

    output_csv = Path(args.output_csv).expanduser().resolve()
    if args.output_json is None:
        args.output_json = str(output_csv.with_suffix(".summary.json"))

    if args.summarize_only:
        annotations = load_annotations(output_csv)
        write_summary(Path(args.output_json).expanduser().resolve(), annotations)
        print(f"Wrote summary: {args.output_json}")
        return

    if not Path(args.index).exists():
        raise SystemExit(f"Index not found: {args.index}")

    root = tk.Tk()
    AnnotatorApp(root, args)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
