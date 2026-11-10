#!/usr/bin/env python3
"""
Worm Segmenter for Body-size Selection  -  GUI

Run:
    python worm_segmenter_gui.py

Hidden developer option (not visible in the GUI):
    python worm_segmenter_gui.py --debug        # step-by-step debug figures for first 3 images/folder
    python worm_segmenter_gui.py --debug 10     # ... first 10 images per folder
    python worm_segmenter_gui.py --debug -1     # ... all images

Pipeline per input folder X:
    1. micro-sam segmentation   -> <output>/X/segmented/*_mask.tif
    2. WormSizer measurements   -> <output>/X/wormsizer_results/
"""

import argparse
import contextlib
import dataclasses
import json
import queue
import threading
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

APP_TITLE = "Worm Segmenter for Body-size Selection"
IMAGE_EXTENSIONS = {".tif", ".tiff"}
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PARAMS_FILE = SCRIPT_DIR / "default_params.json"

DEFAULT_CONFIG = {
    "segmentation": {
        "model_type": "vit_b_lm",
        "segmentation_mode": "apg",
        "device": "cpu",
    },
    "wormsizer": {
        "um_per_px": 2.0,
        "interval_px": 10.0,
        "min_area_px": 0,
        "max_area_px": None,        # null = no upper limit
        "close_radius": 0,
        "smooth_sigma_px": 3.0,
    },
}


# --------------------------------------------------------------------------- #
# Helpers (no heavy imports here, so the window opens instantly)
# --------------------------------------------------------------------------- #
class Cancelled(BaseException):
    """Raised inside progress callbacks to abort the run."""


def has_images(folder: Path) -> bool:
    try:
        return any(p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS for p in folder.iterdir())
    except OSError:
        return False


def find_batch_folders(root: Path, exclude: Path = None):
    """Immediate subfolders of `root` that directly contain .tif/.tiff files."""
    out = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if exclude is not None and d.resolve() == exclude.resolve():
            continue
        if has_images(d):
            out.append(d)
    return out


def ensure_default_params_file():
    if not DEFAULT_PARAMS_FILE.exists():
        DEFAULT_PARAMS_FILE.write_text(json.dumps(DEFAULT_CONFIG, indent=2), encoding="utf-8")


def load_config(path: Path) -> dict:
    """Read a parameters JSON; missing keys fall back to defaults."""
    with open(path, "r", encoding="utf-8") as fh:
        user = json.load(fh)
    if not isinstance(user, dict):
        raise ValueError("parameters file must contain a JSON object")
    cfg = {sec: dict(vals) for sec, vals in DEFAULT_CONFIG.items()}
    for sec in cfg:
        if sec in user:
            if not isinstance(user[sec], dict):
                raise ValueError(f'section "{sec}" must be a JSON object')
            cfg[sec].update(user[sec])
    return cfg


class QueueWriter:
    """File-like object that forwards printed lines to the GUI log."""
    def __init__(self, q):
        self.q = q
        self._buf = ""

    def write(self, s):
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self.q.put(("log", line))
        return len(s)

    def flush(self):
        pass


# --------------------------------------------------------------------------- #
# Worker (runs in a background thread)
# --------------------------------------------------------------------------- #
def run_pipeline(jobs, out_root: Path, cfg: dict, debug_n: int, q: queue.Queue, cancel: threading.Event):
    def check_cancel():
        if cancel.is_set():
            raise Cancelled()

    summary = []
    try:
        with contextlib.redirect_stdout(QueueWriter(q)):
            q.put(("phase", "Loading model..."))
            q.put(("log", "Loading micro-sam model (first time may download weights)..."))
            from automatic_worm_segmentation import load_segmenter, segment_folder
            from wormsizer import Params, run_batch

            # WormSizer parameters
            valid = {f.name for f in dataclasses.fields(Params)}
            ws = {}
            for k, v in cfg["wormsizer"].items():
                if k not in valid:
                    q.put(("log", f"WARNING: unknown wormsizer parameter '{k}' ignored"))
                    continue
                if k == "max_area_px" and v is None:
                    v = float("inf")
                ws[k] = v
            params = Params(**ws)

            predictor, segmenter = load_segmenter(**cfg["segmentation"])
            q.put(("log", "Model loaded."))

            total = len(jobs)
            for fi, folder in enumerate(jobs, 1):
                check_cancel()
                out_dir = out_root / folder.name
                seg_dir = out_dir / "segmented"
                res_dir = out_dir / "wormsizer_results"
                q.put(("folder", fi, total, folder.name))
                q.put(("log", f"\n=== [{fi}/{total}] {folder.name} ==="))

                try:
                    # ---- 1. segmentation ----
                    q.put(("phase", "Segmenting"))
                    seg_fail = segment_folder(
                        folder, seg_dir, predictor, segmenter,
                        progress_cb=lambda d, t, n: (check_cancel(), q.put(("image", d, t, n))),
                        log_cb=lambda m: q.put(("log", m)),
                    )

                    if not any(seg_dir.glob("*_mask.tif")):
                        raise RuntimeError("no masks were produced")

                    # ---- 2. size analysis ----
                    q.put(("phase", "Analyzing size"))
                    df, failed = run_batch(
                        seg_dir, params, out_dir=str(res_dir), debug_n=debug_n,
                        progress_cb=lambda d, t, n: (check_cancel(), q.put(("image", d, t, n))),
                    )
                    summary.append(
                        f"{folder.name}: {len(df)} measured, {len(failed)} analysis failures, "
                        f"{len(seg_fail)} segmentation failures")
                except Cancelled:
                    raise
                except Exception as e:
                    q.put(("log", f"ERROR in folder {folder.name}: {e!r}"))
                    q.put(("log", traceback.format_exc()))
                    summary.append(f"{folder.name}: FAILED ({e!r})")

            q.put(("folder", total + 1, total, ""))   # completes the folder bar
        q.put(("done", summary))
    except Cancelled:
        q.put(("cancelled", summary))
    except Exception:
        q.put(("error", traceback.format_exc()))


# --------------------------------------------------------------------------- #
# GUI
# --------------------------------------------------------------------------- #
class App(tk.Tk):
    def __init__(self, debug_n=0):
        super().__init__()
        self.title(APP_TITLE)
        self.minsize(720, 200)
        self.debug_n = debug_n
        self.q = queue.Queue()
        self.cancel_event = threading.Event()
        self.worker = None
        self.input_dir = None
        self.output_dir = None
        self.params_file = DEFAULT_PARAMS_FILE
        ensure_default_params_file()

        self._build()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._poll)

    # ---------- layout ---------- #
    def _build(self):
        pad = dict(padx=10, pady=6)
        self.columnconfigure(0, weight=1)

        # Step 1: mode
        f1 = ttk.LabelFrame(self, text="1 · What do you want to process?")
        f1.grid(row=0, column=0, sticky="ew", **pad)
        self.mode = tk.StringVar(value="")
        ttk.Radiobutton(f1, text="One imaging folder", value="single",
                        variable=self.mode, command=self._on_mode).pack(side="left", padx=10, pady=6)
        ttk.Radiobutton(f1, text="Batch: root folder with sub-folders", value="batch",
                        variable=self.mode, command=self._on_mode).pack(side="left", padx=10, pady=6)

        # Step 2: folders + run (hidden until a mode is chosen)
        self.f2 = ttk.LabelFrame(self, text="2 · Folders")
        self.f2.grid(row=1, column=0, sticky="ew", **pad)
        self.f2.columnconfigure(1, weight=1)

        self.btn_in = ttk.Button(self.f2, text="Choose input folder...", command=self._choose_input)
        self.btn_in.grid(row=0, column=0, sticky="ew", padx=8, pady=4)
        self.lbl_in = ttk.Label(self.f2, text="(none)", foreground="gray")
        self.lbl_in.grid(row=0, column=1, sticky="w", padx=4)

        self.btn_out = ttk.Button(self.f2, text="Choose output folder...", command=self._choose_output)
        self.btn_out.grid(row=1, column=0, sticky="ew", padx=8, pady=4)
        self.lbl_out = ttk.Label(self.f2, text="(none)", foreground="gray")
        self.lbl_out.grid(row=1, column=1, sticky="w", padx=4)

        btns = ttk.Frame(self.f2)
        btns.grid(row=1, column=2, padx=8)
        self.btn_run = ttk.Button(btns, text="Run", command=self._start, state="disabled")
        self.btn_run.pack(side="left", padx=2)
        self.btn_cancel = ttk.Button(btns, text="Cancel", command=self._cancel, state="disabled")
        self.btn_cancel.pack(side="left", padx=2)

        self.btn_par = ttk.Button(self.f2, text="Parameters file...", command=self._choose_params)
        self.btn_par.grid(row=2, column=0, sticky="ew", padx=8, pady=4)
        self.lbl_par = ttk.Label(self.f2, text=self._short(self.params_file) + "  (default)")
        self.lbl_par.grid(row=2, column=1, sticky="w", padx=4)

        self.lbl_info = ttk.Label(self.f2, text="", foreground="#1a5fb4")
        self.lbl_info.grid(row=3, column=0, columnspan=3, sticky="w", padx=8, pady=(0, 4))
        self.f2.grid_remove()

        # Progress (hidden until Run)
        self.f3 = ttk.LabelFrame(self, text="Progress")
        self.f3.grid(row=2, column=0, sticky="nsew", **pad)
        self.f3.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)
        self.f3.rowconfigure(5, weight=1)

        self.lbl_phase = ttk.Label(self.f3, text="", font=("TkDefaultFont", 11, "bold"))
        self.lbl_phase.grid(row=0, column=0, sticky="w", padx=8, pady=(6, 0))
        self.lbl_folder = ttk.Label(self.f3, text="")
        self.lbl_folder.grid(row=1, column=0, sticky="w", padx=8)
        self.bar_folder = ttk.Progressbar(self.f3, mode="determinate")
        self.bar_folder.grid(row=2, column=0, sticky="ew", padx=8, pady=2)
        self.lbl_image = ttk.Label(self.f3, text="")
        self.lbl_image.grid(row=3, column=0, sticky="w", padx=8)
        self.bar_image = ttk.Progressbar(self.f3, mode="determinate")
        self.bar_image.grid(row=4, column=0, sticky="ew", padx=8, pady=2)

        logf = ttk.Frame(self.f3)
        logf.grid(row=5, column=0, sticky="nsew", padx=8, pady=6)
        logf.columnconfigure(0, weight=1)
        logf.rowconfigure(0, weight=1)
        self.log = tk.Text(logf, height=10, state="disabled", wrap="none")
        sb = ttk.Scrollbar(logf, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        self.log.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")
        self.f3.grid_remove()

    # ---------- small utilities ---------- #
    @staticmethod
    def _short(p, n=70):
        s = str(p)
        return s if len(s) <= n else "..." + s[-(n - 3):]

    def _log(self, msg):
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _running(self):
        return self.worker is not None and self.worker.is_alive()

    def _update_run_state(self):
        ok = self.input_dir and self.output_dir and not self._running()
        self.btn_run.configure(state="normal" if ok else "disabled")

    def _set_controls(self, enabled):
        st = "normal" if enabled else "disabled"
        for b in (self.btn_in, self.btn_out, self.btn_par):
            b.configure(state=st)
        self.btn_cancel.configure(state="disabled" if enabled else "normal")
        self._update_run_state()

    # ---------- callbacks ---------- #
    def _on_mode(self):
        if self._running():
            return
        self.input_dir = None
        self.lbl_in.configure(text="(none)", foreground="gray")
        self.lbl_info.configure(text="")
        self.f2.grid()
        self._update_run_state()

    def _choose_input(self):
        title = ("Choose the imaging folder" if self.mode.get() == "single"
                 else "Choose the root folder (containing the imaging sub-folders)")
        d = filedialog.askdirectory(title=title)
        if not d:
            return
        self.input_dir = Path(d)
        self.lbl_in.configure(text=self._short(d), foreground="black")
        if self.mode.get() == "batch":
            found = find_batch_folders(self.input_dir, self.output_dir)
            self.lbl_info.configure(text=f"Found {len(found)} sub-folder(s) with .tif/.tiff images.")
        else:
            n = sum(1 for p in self.input_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
            self.lbl_info.configure(text=f"Found {n} .tif/.tiff image(s).")
        self._update_run_state()

    def _choose_output(self):
        d = filedialog.askdirectory(title="Choose the output folder")
        if not d:
            return
        self.output_dir = Path(d)
        self.lbl_out.configure(text=self._short(d), foreground="black")
        self._update_run_state()

    def _choose_params(self):
        f = filedialog.askopenfilename(title="Choose a parameters file",
                                       initialdir=str(SCRIPT_DIR),
                                       filetypes=[("JSON", "*.json"), ("All files", "*.*")])
        if not f:
            return
        self.params_file = Path(f)
        tag = "  (default)" if self.params_file == DEFAULT_PARAMS_FILE else ""
        self.lbl_par.configure(text=self._short(f) + tag)

    def _start(self):
        # parameters
        try:
            cfg = load_config(self.params_file)
        except Exception as e:
            messagebox.showerror("Parameters file", f"Could not read\n{self.params_file}\n\n{e}")
            return

        # jobs
        if self.mode.get() == "single":
            jobs = [self.input_dir] if has_images(self.input_dir) else []
        else:
            jobs = find_batch_folders(self.input_dir, self.output_dir)
        if not jobs:
            messagebox.showwarning("Nothing to do", "No folders with .tif/.tiff images were found.")
            return

        # existing output?
        existing = [j.name for j in jobs
                    if (self.output_dir / j.name).exists() and any((self.output_dir / j.name).iterdir())]
        if existing:
            shown = "\n".join(existing[:15]) + (f"\n... and {len(existing) - 15} more" if len(existing) > 15 else "")
            if not messagebox.askyesno(
                    "Output folder already exists",
                    f"{len(existing)} output folder(s) already exist in\n{self.output_dir}\n\n{shown}\n\n"
                    "Existing results with the same names will be overwritten.\nAre you sure you want to continue?",
                    icon="warning"):
                return

        # go
        self.cancel_event.clear()
        self.log.configure(state="normal"); self.log.delete("1.0", "end"); self.log.configure(state="disabled")
        self.bar_folder.configure(maximum=len(jobs), value=0)
        self.bar_image.configure(maximum=1, value=0)
        self.lbl_phase.configure(text="Starting...")
        self.lbl_folder.configure(text=f"Folders: 0 / {len(jobs)}")
        self.lbl_image.configure(text="")
        self.f3.grid()
        self._log(f"Parameters: {self.params_file}")
        self._log(f"Folders to process ({len(jobs)}): " + ", ".join(j.name for j in jobs))
        self._set_controls(False)

        self.worker = threading.Thread(
            target=run_pipeline,
            args=(jobs, self.output_dir, cfg, self.debug_n, self.q, self.cancel_event),
            daemon=True)
        self.worker.start()

    def _cancel(self):
        self.cancel_event.set()
        self.btn_cancel.configure(state="disabled")
        self.lbl_phase.configure(text="Cancelling (after the current image)...")

    def _on_close(self):
        if self._running():
            if not messagebox.askyesno("Quit", "A run is in progress. Quit anyway?"):
                return
            self.cancel_event.set()
        self.destroy()

    # ---------- queue polling ---------- #
    def _poll(self):
        try:
            while True:
                msg = self.q.get_nowait()
                kind = msg[0]
                if kind == "log":
                    self._log(msg[1])
                elif kind == "phase":
                    self.lbl_phase.configure(text=msg[1])
                    if msg[1] == "Loading model...":
                        self.lbl_image.configure(text="")
                elif kind == "folder":
                    _, idx, total, name = msg
                    self.bar_folder.configure(value=idx - 1)
                    if name:
                        self.lbl_folder.configure(text=f"Folders: {idx - 1} done, now {idx} / {total}  ({name})")
                elif kind == "image":
                    _, done, total, name = msg
                    self.bar_image.configure(maximum=max(total, 1), value=done)
                    self.lbl_image.configure(
                        text=f"Images processed: {done} / {total}" + (f"   (current: {name})" if name else ""))
                elif kind == "done":
                    self._finish("Finished", msg[1])
                    messagebox.showinfo("Finished", "All folders processed.\n\n" + "\n".join(msg[1]))
                elif kind == "cancelled":
                    self._finish("Cancelled", msg[1])
                elif kind == "error":
                    self._finish("Error", [])
                    self._log(msg[1])
                    messagebox.showerror("Error", "The run stopped with an error (see log):\n\n" + msg[1][-800:])
        except queue.Empty:
            pass
        self.after(100, self._poll)

    def _finish(self, label, summary):
        self.lbl_phase.configure(text=label)
        self._log(f"\n--- {label} ---")
        for s in summary:
            self._log(s)
        self._set_controls(True)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=APP_TITLE)
    ap.add_argument("--debug", nargs="?", const=3, type=int, default=0, metavar="N",
                    help="save WormSizer debug figures for the first N images per folder "
                         "(no value = 3, -1 = all). Hidden from the GUI.")
    args, _ = ap.parse_known_args()
    App(debug_n=args.debug).mainloop()


if __name__ == "__main__":
    main()
