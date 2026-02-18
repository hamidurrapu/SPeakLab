from __future__ import annotations

import io
import os
import sys
import tempfile
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import List, Optional

import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Optional: Pillow for better PNG handling/resizing; falls back to Tk if unavailable
try:
    from PIL import Image, ImageTk  # type: ignore
except Exception:
    Image = None
    ImageTk = None

from speaklab_core import (
    FitConfig, PeakSpec, process_file, save_per_file_outputs, 
    run_batch, setup_logging, logger
)


HALF_WIDTH_SINGLE_NORM = 0.5  # half-width used when user provides a single value for normalization range


def require_dir(path: str) -> bool:
    """Validate that a path is a valid directory."""
    if not path or not os.path.isdir(path):
        messagebox.showerror("Invalid directory", "Please choose a valid working directory.")
        return False
    return True


def choose_directory(entry: tk.Entry):
    """Open directory browser and update entry widget."""
    d = filedialog.askdirectory()
    if d:
        entry.delete(0, tk.END)
        entry.insert(0, d)


def choose_file(entry: tk.Entry, directory_entry: tk.Entry):
    """Open file browser and update entry widget."""
    initialdir = directory_entry.get() if os.path.isdir(directory_entry.get()) else os.getcwd()
    f = filedialog.askopenfilename(
        title="Select input .txt",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        initialdir=initialdir,
    )
    if f:
        entry.delete(0, tk.END)
        entry.insert(0, f)
        directory_entry.delete(0, tk.END)
        directory_entry.insert(0, os.path.dirname(f))


def parse_range(s: str) -> Optional[tuple[float, float]]:
    """Parse a range string like '100 200' or '150' into (min, max) tuple.
       Returns None for blank input."""
    try:
        s = s.strip().replace(",", " ")
        if s == "":
            return None
        parts = [p for p in s.split() if p]
        if len(parts) == 1:
            # Interpret single value as [0, val]
            val = float(parts[0])
            return (0.0, val)
        elif len(parts) >= 2:
            a, b = float(parts[0]), float(parts[1])
            return (min(a, b), max(a, b))
    except Exception:
        return None
    return None


def parse_positive_float(s: str) -> Optional[float]:
    """Parse a string to a positive float, handling various formats."""
    if s is None:
        return None
    s = s.strip()
    if s == "":
        return None
    try:
        s = s.replace("−", "-").replace("–", "-")
        token = "".join(ch for ch in s if (ch.isdigit() or ch in "+-.,eE"))
        token = token.replace(",", ".")
        if token == "" or token in "+-.":
            return None
        val = float(token)
        if val > 0 and np.isfinite(val):
            return val
    except Exception:
        return None
    return None


class PeakHistory:
    """Undo/Redo history for peak entries."""
    
    def __init__(self, max_history: int = 20):
        # store quadruples: (label, center_range, fwhm_range, amplitude_range)
        self.history: List[List[tuple[str, str, str, str]]] = []
        self.current_index = -1
        self.max_history = max_history
    
    def save_state(self, peak_data: List[tuple[str, str, str, str]]):
        """Save current peak configuration."""
        # Remove any forward history
        self.history = self.history[:self.current_index + 1]
        self.history.append(peak_data.copy())
        if len(self.history) > self.max_history:
            self.history.pop(0)
        else:
            self.current_index += 1
    
    def undo(self) -> Optional[List[tuple[str, str, str, str]]]:
        """Undo to previous state."""
        if self.current_index > 0:
            self.current_index -= 1
            return self.history[self.current_index]
        return None
    
    def redo(self) -> Optional[List[tuple[str, str, str, str]]]:
        """Redo to next state."""
        if self.current_index < len(self.history) - 1:
            self.current_index += 1
            return self.history[self.current_index]
        return None


class RamanFitGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Bol Spectra")
        self.geometry("1320x820")
        
        # Initialize logging
        setup_logging(verbose=True)

        self._set_window_icon()

        self.configure(bg="white")
        style = ttk.Style(self)
        style.configure("White.TFrame", background="white")
        style.configure("White.TLabel", background="white")
        style.configure(".", background="white")

        self.current_fig = None
        self.current_df: Optional[pd.DataFrame] = None
        self.current_input_filename = None
        self.logo_img = None
        self.peak_history = PeakHistory()

        # State for background work
        self.worker_thread: Optional[threading.Thread] = None
        our_cancel = threading.Event()
        self.cancel_event: Optional[threading.Event] = our_cancel
        self._is_running = False

        self._build_header()

        self.left = ttk.Frame(self, padding=(10, 50, 10, 10), style="White.TFrame")
        self.left.pack(side=tk.LEFT, fill=tk.Y)
        right = ttk.Frame(self, padding=(10, 0, 10, 10), style="White.TFrame")
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Mode
        ttk.Label(self.left, text="Mode:", style="White.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 10))
        self.mode_var = tk.StringVar(value="single")
        ttk.Radiobutton(self.left, text="Batch", variable=self.mode_var, value="batch",
                        command=self._on_mode_changed).grid(row=0, column=1, sticky="w", pady=(0, 10))
        ttk.Radiobutton(self.left, text="Single File", variable=self.mode_var, value="single",
                        command=self._on_mode_changed).grid(row=0, column=2, sticky="w", pady=(0, 10))

        # Batch Files Directory (row 1)
        self.dir_label = ttk.Label(self.left, text="Batch Files Directory", style="White.TLabel")
        self.dir_entry = ttk.Entry(self.left, width=50)
        self.dir_browse_btn = ttk.Button(self.left, text="Browse…", command=lambda: choose_directory(self.dir_entry))
        self.dir_label.grid(row=1, column=0, columnspan=7, pady=(0, 0), sticky="w")
        self.dir_entry.grid(row=1, column=1, columnspan=5, sticky="we", pady=(0, 0))
        self.dir_browse_btn.grid(row=1, column=6, sticky="w")
        self.batch_dir_widgets = [self.dir_label, self.dir_entry, self.dir_browse_btn]

        # Raman Spectra File (row 2)
        self.file_label = ttk.Label(self.left, text="Raman Spectra File", style="White.TLabel")
        self.file_entry = ttk.Entry(self.left, width=50)
        self.file_browse_btn = ttk.Button(self.left, text="Browse…", command=lambda: choose_file(self.file_entry, self.dir_entry))
        self.file_label.grid(row=2, column=0, sticky="w")
        self.file_entry.grid(row=2, column=1, columnspan=5, sticky="we")
        self.file_browse_btn.grid(row=2, column=6, sticky="w")
        self.single_file_widgets = [self.file_label, self.file_entry, self.file_browse_btn]

        # Fit range
        ttk.Label(self.left, text="Fit Range:", style="White.TLabel").grid(row=3, column=0, sticky="w", pady=(10, 0))
        self.fit_range_entry = ttk.Entry(self.left, width=18)
        self.fit_range_entry.insert(0, "300 480")
        self.fit_range_entry.grid(row=3, column=1, sticky="w", pady=(10, 0))

        # Baseline
        ttk.Label(self.left, text="Baseline:", style="White.TLabel").grid(row=4, column=0, sticky="w", pady=(10, 0))
        self.baseline_var = tk.StringVar(value="linear")
        ttk.Radiobutton(self.left, text="Linear", variable=self.baseline_var, value="linear").grid(row=4, column=1, sticky="w", pady=(10, 0))
        ttk.Radiobutton(self.left, text="Shirley", variable=self.baseline_var, value="shirley").grid(row=4, column=2, sticky="w", pady=(10, 0))

        # Si Peak Shift + Si Range
        ttk.Label(self.left, text="Si Peak Shift:", style="White.TLabel").grid(row=5, column=0, sticky="w", pady=(10, 0))
        self.align_si_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.left, text="Align to", variable=self.align_si_var).grid(row=5, column=1, sticky="w", pady=(10, 0))
        self.si_target_entry = ttk.Entry(self.left, width=10)
        self.si_target_entry.insert(0, "520.0")
        self.si_target_entry.grid(row=5, column=2, sticky="w", pady=(10, 0))
        ttk.Label(self.left, text="  Si Range:", style="White.TLabel").grid(row=5, column=3, sticky="w", pady=(10, 0))
        self.si_range_entry = ttk.Entry(self.left, width=18)
        self.si_range_entry.insert(0, "510 540")
        self.si_range_entry.grid(row=5, column=4, sticky="w", pady=(10, 0))

        # Normalize
        ttk.Label(self.left, text="Normalize to:", style="White.TLabel").grid(row=6, column=0, sticky="w", pady=(10, 0))
        self.norm_var = tk.StringVar(value="si")
        ttk.Radiobutton(self.left, text="None", variable=self.norm_var, value="none", command=self._on_norm_changed).grid(row=6, column=1, sticky="w", pady=(10, 0))
        ttk.Radiobutton(self.left, text="Si", variable=self.norm_var, value="si", command=self._on_norm_changed).grid(row=6, column=2, sticky="w", pady=(10, 0))
        ttk.Radiobutton(self.left, text="Range:", variable=self.norm_var, value="range", command=self._on_norm_changed).grid(row=6, column=3, sticky="w", pady=(10, 0))
        self.norm_range_entry = ttk.Entry(self.left, width=18)
        self.norm_range_entry.insert(0, "400.0 415.0")
        self.norm_range_entry.grid(row=6, column=4, sticky="w", pady=(10, 0))

        # Instrument Broadening mode
        ttk.Label(self.left, text="Instrument Broadening:", style="White.TLabel").grid(row=7, column=0, sticky="w", pady=(10, 0))
        self.gauss_mode_var = tk.StringVar(value="fixed")
        ttk.Radiobutton(self.left, text="Fixed", variable=self.gauss_mode_var, value="fixed",
                        command=self._on_gauss_mode_changed).grid(row=7, column=1, sticky="w", pady=(10, 0))
        ttk.Radiobutton(self.left, text="Fit", variable=self.gauss_mode_var, value="fit",
                        command=self._on_gauss_mode_changed).grid(row=7, column=2, sticky="w", pady=(10, 0))

        # Broadening Source
        self.broadening_source_label = ttk.Label(self.left, text="", style="White.TLabel")
        self.source_var = tk.StringVar(value="fixed")
        self.source_range_rb = ttk.Radiobutton(
            self.left,
            text="Estimate from Peak Range",
            variable=self.source_var,
            value="range",
            command=self._on_source_changed
        )
        self.source_range_entry = ttk.Entry(self.left, width=18)
        self.source_range_entry.insert(0, "510 540")
        self.source_manual_rb = ttk.Radiobutton(
            self.left,
            text="Spectral Resolution (FWHM, cm⁻¹)",
            variable=self.source_var,
            value="fixed",
            command=self._on_source_changed
        )
        self.manual_fwhm_entry = ttk.Entry(self.left, width=10)
        self.manual_fwhm_entry.insert(0, "2.47")

        self.broadening_source_label.grid(row=8, column=0, sticky="w", pady=(0, 0))
        self.source_range_rb.grid(row=8, column=1, columnspan=2, sticky="w", pady=(0, 0))
        self.source_range_entry.grid(row=8, column=3, sticky="w", pady=(0, 0))
        self.source_manual_rb.grid(row=9, column=1, columnspan=2, sticky="w")
        self.manual_fwhm_entry.grid(row=9, column=3, sticky="w")

        self.broadening_source_widgets = [
            self.broadening_source_label,
            self.source_range_rb,
            self.source_range_entry,
            self.source_manual_rb,
            self.manual_fwhm_entry,
        ]

        # Peak Preset
        ttk.Label(self.left, text="Peak Preset:", style="White.TLabel").grid(row=12, column=0, sticky="w", pady=(15, 0))
        self.preset_var = tk.StringVar(value="MoS2")
        self.preset_combo = ttk.Combobox(self.left, values=["MoS2", "Custom"], textvariable=self.preset_var,
                                         state="readonly", width=10)
        self.preset_combo.grid(row=12, column=1, sticky="w", pady=(15, 0))
        self.preset_combo.bind("<<ComboboxSelected>>", lambda e: self._on_preset_changed())
        self.custom_preset_entry = ttk.Entry(self.left, width=36, state="disabled")
        self.custom_preset_entry.grid(row=12, column=2, columnspan=2, sticky="we", pady=(15, 0))
        self.custom_preset_btn = ttk.Button(self.left, text="Browse…", state="disabled", command=self._choose_preset_file)
        self.custom_preset_btn.grid(row=12, column=4, sticky="w", pady=(15, 0))

        # Number of peaks
        ttk.Label(self.left, text="Number of Peaks:", style="White.TLabel").grid(row=13, column=0, sticky="w", pady=(15, 0))
        self.num_peaks_var = tk.IntVar(value=5)
        self.num_peaks_spin = ttk.Spinbox(self.left, from_=1, to=20, textvariable=self.num_peaks_var,
                                          width=5, command=self._rebuild_peak_entries)
        self.num_peaks_spin.grid(row=13, column=1, sticky="w", pady=(15, 0))

        # Column headers
        ttk.Label(self.left, text="Peak Label", style="White.TLabel",
                  font=("Arial", 9, "bold")).grid(row=15, column=0, sticky="w", padx=(6, 0), pady=(10, 4))
        ttk.Label(self.left, text="Range", style="White.TLabel",
                  font=("Arial", 9, "bold")).grid(row=15, column=0, sticky="w", padx=(112, 0), pady=(10, 4))
        ttk.Label(self.left, text="FWHM", style="White.TLabel",
                  font=("Arial", 9, "bold")).grid(row=15, column=1, sticky="w", padx=(42, 0), pady=(10, 4))
        ttk.Label(self.left, text="Area", style="White.TLabel",
                  font=("Arial", 9, "bold")).grid(row=15, column=2, sticky="w", padx=(13, 0), pady=(10, 4))
                  
        # Peak rows container
        self.peaks_frame = ttk.Frame(self.left, style="White.TFrame")
        self.peaks_frame.grid(row=16, column=0, columnspan=7, sticky="w")
        # each row is (label_entry, center_range_entry, fwhm_range_entry, amplitude_range_entry)
        self.peak_entries: List[tuple[tk.Entry, tk.Entry, tk.Entry, tk.Entry]] = []
        self._rebuild_peak_entries(first_time=True)

        self.save_plots_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.left, text="Save plots during batch", variable=self.save_plots_var).grid(
            row=17, column=0, columnspan=7, sticky="w", pady=(6, 0))

        self.generate_btn = ttk.Button(self.left, text="Generate", command=self.on_generate)
        self.generate_btn.grid(row=18, column=0, columnspan=2, pady=(14, 0), sticky="we")

        # Cancel button (shown during background work, especially batch)
        self.cancel_btn = ttk.Button(self.left, text="Cancel", command=self.on_cancel)
        self.cancel_btn.grid(row=18, column=2, columnspan=2, pady=(14, 0), sticky="we")
        self.cancel_btn.grid_remove()

        for i in range(7):
            self.left.grid_columnconfigure(i, weight=1)

        # Plot area (on the right)
        self.canvas_frame = ttk.Frame(right, style="White.TFrame")
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        self.canvas: Optional[FigureCanvasTkAgg] = None

        # Bottom buttons (on the right)
        bottom = ttk.Frame(right, style="White.TFrame")
        bottom.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(bottom, text="Copy Image", command=self.copy_image).pack(side=tk.LEFT)
        ttk.Button(bottom, text="Copy Data", command=self.copy_data).pack(side=tk.LEFT, padx=8)
        ttk.Button(bottom, text="Save Plot", command=self.save_plot).pack(side=tk.RIGHT)

        # Status and batch progress (moved to the left column under controls)
        status_container = ttk.Frame(self.left, style="White.TFrame")
        status_container.grid(row=19, column=0, columnspan=7, sticky="we", pady=(10, 0))
        status_container.grid_columnconfigure(0, weight=1)
        status_container.grid_columnconfigure(1, weight=1)

        # Left status bar (general status)
        status_left_frame = ttk.Frame(status_container, style="White.TFrame")
        status_left_frame.grid(row=0, column=0, sticky="we")
        status_left_frame.grid_columnconfigure(0, weight=1)
        
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(
            status_left_frame, 
            textvariable=self.status_var, 
            relief=tk.SUNKEN, 
            anchor=tk.W, 
            padding=(5, 2)
        )
        status_bar.grid(row=0, column=0, sticky="we")

        # Batch progress status (initially hidden)
        self.batch_progress_frame = ttk.Frame(status_container, style="White.TFrame")
        self.batch_progress_frame.grid(row=0, column=1, sticky="we", padx=(10, 0))
        self.batch_progress_frame.grid_columnconfigure(0, weight=1)
        self.batch_progress_frame.grid_columnconfigure(1, weight=0)
        
        self.batch_progress_var = tk.StringVar(value="")
        self.batch_progress_label = ttk.Label(
            self.batch_progress_frame, 
            textvariable=self.batch_progress_var,
            relief=tk.SUNKEN, 
            anchor=tk.W,
            padding=(5, 2)
        )
        self.batch_progress_label.grid(row=0, column=0, sticky="we")
        
        self.batch_progress_bar = ttk.Progressbar(
            self.batch_progress_frame, 
            length=200, 
            mode='determinate'
        )
        self.batch_progress_bar.grid(row=0, column=1, padx=(5, 5), sticky="we")
        
        # Initially hide batch progress (now using grid)
        self.batch_progress_frame.grid_remove()

        # Keyboard shortcuts
        self.bind('<Control-g>', lambda e: self.on_generate())
        self.bind('<Control-s>', lambda e: self.save_plot())
        self.bind('<Control-c>', lambda e: self.copy_data())
        self.bind('<Control-z>', lambda e: self._undo_peaks())
        self.bind('<Control-y>', lambda e: self._redo_peaks())

        self.dir_entry.insert(0, os.getcwd())
        self._on_norm_changed()
        self._on_mode_changed()
        self._on_gauss_mode_changed()
        self._on_source_changed()
        self._on_preset_changed()

    def show_batch_progress(self, show: bool = True):
        """Show or hide the batch progress indicators."""
        if show:
            self.batch_progress_frame.grid()
            self.batch_progress_bar['value'] = 0
        else:
            self.batch_progress_frame.grid_remove()
            self.batch_progress_var.set("")

    def update_batch_progress(self, current: int, total: int, filename: str):
        """Update batch progress display."""
        try:
            self.batch_progress_var.set(f"Processing {current + 1}/{total}: {filename}")
            self.batch_progress_bar['value'] = ((current + 1) / total) * 100
            self.update_idletasks()
        except Exception:
            pass

    def _threadsafe_progress(self, current: int, total: int, filename: str):
        """Wrapper to update progress safely from worker thread."""
        self.after(0, self.update_batch_progress, current, total, filename)

    def update_status(self, message: str):
        """Update status bar message."""
        self.status_var.set(message)
        self.update_idletasks()
        logger.info(f"Status: {message}")

    def _resource_path(self, relative: str) -> str:
        base = getattr(sys, "_MEIPASS", os.path.abspath(os.path.dirname(__file__)))
        return os.path.join(base, relative)

    def _set_window_icon(self):
        icon_env = os.environ.get("speaklab_ICON")
        candidates = []
        if icon_env:
            candidates.append(icon_env)
        candidates.append(self._resource_path("assets/speaklab_icon.ico"))
        candidates.append(self._resource_path("assets/speaklab_icon.png"))
        for path in candidates:
            if not path or not os.path.exists(path):
                continue
            try:
                if sys.platform.startswith("win") and path.lower().endswith(".ico"):
                    self.iconbitmap(path)
                    return
                if path.lower().endswith((".png", ".gif")):
                    img = tk.PhotoImage(file=path)
                    self.iconphoto(True, img)
                    self._icon_ref = img
                    return
                if Image and ImageTk:
                    img_pil = Image.open(path)
                    tk_img = ImageTk.PhotoImage(img_pil)
                    self.iconphoto(True, tk_img)
                    self._icon_ref = tk_img
                    return
            except Exception:
                pass

    def _build_header(self):
        header = ttk.Frame(self, padding=(10, 10, 10, 0), style="White.TFrame")
        header.pack(side=tk.TOP, fill=tk.X)
        logo_rel = os.environ.get("speaklab_HEADER", "assets/speaklab_header.png")
        logo_path = self._resource_path(logo_rel)
        if os.path.exists(logo_path):
            try:
                if Image and ImageTk:
                    img = Image.open(logo_path)
                    target_h = 100
                    w = int(img.width * (target_h / img.height))
                    img = img.resize((w, target_h), Image.LANCZOS)
                    self.logo_img = ImageTk.PhotoImage(img)
                else:
                    self.logo_img = tk.PhotoImage(file=logo_path)
                ttk.Label(header, image=self.logo_img, style="White.TLabel").pack(side=tk.LEFT)
            except Exception as e:
                logger.warning(f"Could not load header image: {e}")

    def _on_source_changed(self):
        if self.gauss_mode_var.get() != 'fixed':
            self.source_range_entry.configure(state="disabled")
            self.manual_fwhm_entry.configure(state="disabled")
            return

        if self.source_var.get() == "range":
            self.source_range_entry.configure(state="normal")
            self.manual_fwhm_entry.configure(state="disabled")
        else:
            self.source_range_entry.configure(state="disabled")
            self.manual_fwhm_entry.configure(state="normal")

    def _on_norm_changed(self):
        if self.norm_var.get() == 'range':
            self.norm_range_entry.configure(state="normal")
        else:
            self.norm_range_entry.configure(state="disabled")

    def _on_mode_changed(self):
        mode = self.mode_var.get()
        if mode == "batch":
            for w in self.batch_dir_widgets:
                w.grid()
            for w in self.single_file_widgets:
                w.grid_remove()
        else:
            for w in self.single_file_widgets:
                w.grid()
            for w in self.batch_dir_widgets:
                w.grid_remove()

    def _on_gauss_mode_changed(self):
        mode = self.gauss_mode_var.get()
        if mode == 'fixed':
            for w in self.broadening_source_widgets:
                w.grid()
            self._on_source_changed()
        else:
            for w in self.broadening_source_widgets:
                w.grid_remove()
            self.source_range_entry.configure(state="disabled")
            self.manual_fwhm_entry.configure(state="disabled")

    def _on_preset_changed(self):
        preset = self.preset_var.get()
        if preset == "MoS2":
            self.custom_preset_btn.configure(state="disabled")
            self.custom_preset_entry.configure(state="disabled")
            self._apply_mos2_preset()
        else:
            self.custom_preset_btn.configure(state="normal")
            self.custom_preset_entry.configure(state="readonly")
            path = self.custom_preset_entry.get().strip()
            if path and os.path.isfile(path):
                self._apply_custom_preset_from_file(path)

    def _apply_mos2_preset(self):
        # Leave FWHM blank by default to trigger robust guess
        mos2 = [
            ("TO(M)", "355 360", "", ""),
            ("LO(M)", "366 376", "", ""),
            ("E2g",  "376 390", "", ""),
            ("A1g",  "400 412", "", ""),
            ("ZO(M)", "410.7 413.1", "", ""),
        ]
        self._fill_peaks(mos2)

    def _fill_peaks(self, peaks: List[tuple[str, str, str, str]]):
        """
        Fill table rows with (label, center_range, fwhm_range, amplitude_range) quadruplets.
        Backward compatible: if triples are provided, fwhm and amplitude are left blank (auto).
        """
        self.num_peaks_var.set(len(peaks))
        self._rebuild_peak_entries(first_time=False)
        for i, widgets in enumerate(self.peak_entries):
            le, re, ge, ae = widgets
            if i < len(peaks):
                row = peaks[i]
                # Allow short tuples and fill blanks as needed
                lbl = row[0] if len(row) >= 1 else f"Peak{i+1}"
                rng = row[1] if len(row) >= 2 else "0 0"
                fwhm = row[2] if len(row) >= 3 else ""
                ampl = row[3] if len(row) >= 4 else ""
            else:
                lbl, rng, fwhm, ampl = (f"Peak{i+1}", "0 0", "", "")
            for w, val in ((le, lbl), (re, rng), (ge, fwhm), (ae, ampl)):
                w.delete(0, tk.END)
                w.insert(0, val)
        self._save_peak_state()

    def _choose_preset_file(self):
        initialdir = self.dir_entry.get() if os.path.isdir(self.dir_entry.get()) else os.getcwd()
        f = filedialog.askopenfilename(
            title="Select peak preset file",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialdir=initialdir,
        )
        if not f:
            return
        self.custom_preset_entry.configure(state="normal")
        self.custom_preset_entry.delete(0, tk.END)
        self.custom_preset_entry.insert(0, f)
        self.custom_preset_entry.configure(state="readonly")
        self._apply_custom_preset_from_file(f)
    
    def _apply_custom_preset_from_file(self, path: str):
        try:
            n, peaks = self._parse_preset_file(path)
        except Exception as e:
            messagebox.showerror("Preset file error", f"Could not read preset file:\n{e}")
            logger.error(f"Failed to parse preset file {path}: {e}")
            return
        if n < 1 or not peaks:
            messagebox.showerror("Preset file error", "No peak definitions found in the file.")
            return
        if len(peaks) != n:
            peaks = peaks[:n]
        # If FWHM is missing in file, leave blank to use robust guess; amplitude blank by default
        peaks = [(lbl, rng, (gr if gr and gr.strip() else ""), "") for (lbl, rng, gr) in peaks]
        self._fill_peaks(peaks)
        self.update_status(f"Loaded preset from {os.path.basename(path)}")

    def _parse_preset_file(self, path: str) -> tuple[int, List[tuple[str, str, str]]]:
        """
        Parse a custom preset file.
        Expected format (whitespace-delimited):
          Line 1: integer N (number of peaks) [optional]
          Next N lines: <label tokens...> <start> <end> [fwhm_min fwhm_max]
        Returns: (N, [(label, "start end", "fwhm_min fwhm_max"), ...])
        """
        with open(path, "r", encoding="utf-8") as fh:
            lines = [ln.strip() for ln in fh if ln.strip()]

        if not lines:
            raise ValueError("File is empty.")

        def to_float(tok: str) -> float:
            tok = tok.replace(",", ".").replace("−", "-").replace("–", "-")
            return float(tok)

        n = None
        try:
            n = int(lines[0].split()[0])
            content = lines[1:]
        except Exception:
            content = lines

        peaks: List[tuple[str, str, str]] = []
        for ln in content:
            parts = ln.split()
            if len(parts) < 3:
                continue
            label = ""
            try:
                # case with FWHM: ... start end fmin fmax
                if len(parts) >= 5:
                    start = to_float(parts[-4]); end = to_float(parts[-3])
                    fmin = to_float(parts[-2]); fmax = to_float(parts[-1])
                    label = " ".join(parts[:-4]).strip() or "Peak"
                    peaks.append((label, f"{start} {end}", f"{fmin} {fmax}"))
                    continue
            except Exception:
                pass
            try:
                # fallback: only center range provided -> leave FWHM blank
                start = to_float(parts[-2]); end = to_float(parts[-1])
                label = " ".join(parts[:-2]).strip() or "Peak"
                peaks.append((label, f"{start} {end}", ""))
            except Exception:
                continue

        if n is None:
            n = len(peaks)
        return n, peaks

    def _rebuild_peak_entries(self, first_time: bool = False):
        for child in self.peaks_frame.winfo_children():
            child.destroy()
        self.peak_entries.clear()
        n = self.num_peaks_var.get()

        # FWHM blank by default to trigger robust guess; amplitude blank for auto
        defaults = [
            ("TO(M)", "355 360", "", ""),
            ("LO(M)", "366 376", "", ""),
            ("E2g",  "376 390", "", ""),
            ("A1g",  "400 412", "", ""),
            ("ZO(M)", "410.7 413.1", "", ""),
        ]

        for i in range(n):
            le = ttk.Entry(self.peaks_frame, width=10)
            re = ttk.Entry(self.peaks_frame, width=15)
            ge = ttk.Entry(self.peaks_frame, width=8)  # FWHM range
            ae = ttk.Entry(self.peaks_frame, width=15)  # amplitude range
            if first_time and i < len(defaults):
                le.insert(0, defaults[i][0])
                re.insert(0, defaults[i][1])
                ge.insert(0, defaults[i][2])  # blank = robust guess
                ae.insert(0, defaults[i][3])  # blank = auto amplitude bounds
            else:
                le.insert(0, f"Peak{i+1}")
                re.insert(0, "0 0")
                ge.insert(0, "")  # blank
                ae.insert(0, "")  # blank
            # Tight, side-by-side
            le.grid(row=i, column=0, sticky="w", padx=(6, 8))
            re.grid(row=i, column=1, sticky="w", padx=(0, 8))
            ge.grid(row=i, column=2, sticky="w", padx=(0, 8))
            ae.grid(row=i, column=3, sticky="w", padx=(0, 0))
            self.peak_entries.append((le, re, ge, ae))

        # Keep snug columns
        for c in range(4):
            self.peaks_frame.grid_columnconfigure(c, weight=0)
        
        if not first_time:
            self._save_peak_state()

    def _save_peak_state(self):
        """Save current peak configuration to history."""
        peak_data = [(le.get(), re.get(), ge.get(), ae.get()) for le, re, ge, ae in self.peak_entries]
        self.peak_history.save_state(peak_data)

    def _undo_peaks(self):
        """Undo to previous peak configuration."""
        prev_state = self.peak_history.undo()
        if prev_state:
            self._restore_peak_state(prev_state)
            self.update_status("Undo peak configuration")
        else:
            self.update_status("No more undo history")

    def _redo_peaks(self):
        """Redo to next peak configuration."""
        next_state = self.peak_history.redo()
        if next_state:
            self._restore_peak_state(next_state)
            self.update_status("Redo peak configuration")
        else:
            self.update_status("No more redo history")

    def _restore_peak_state(self, state: List[tuple[str, str, str, str]]):
        """Restore peak entries from saved state."""
        self.num_peaks_var.set(len(state))
        self._rebuild_peak_entries(first_time=False)
        for i, (le, re, ge, ae) in enumerate(self.peak_entries):
            if i < len(state):
                le.delete(0, tk.END); le.insert(0, state[i][0])
                re.delete(0, tk.END); re.insert(0, state[i][1])
                ge.delete(0, tk.END); ge.insert(0, state[i][2])
                ae.delete(0, tk.END); ae.insert(0, state[i][3])

    def build_config(self) -> Optional[FitConfig]:
        """Build configuration from GUI inputs."""
        wd = self.dir_entry.get()
        if not require_dir(wd):
            return None
        si_r = parse_range(self.si_range_entry.get()) or (510.0, 540.0)
        fit_r = parse_range(self.fit_range_entry.get()) or (300.0, 480.0)

        norm_choice = self.norm_var.get()
        if norm_choice == "none":
            norm_val = None
            norm_range = None
        elif norm_choice == "si":
            norm_val = "si"
            norm_range = None
        elif norm_choice == "range":
            r = parse_range(self.norm_range_entry.get())
            if r is None or r[0] == r[1]:
                messagebox.showerror("Normalization range error",
                                     f"Invalid normalization range entry: '{self.norm_range_entry.get()}'")
                return None
            norm_val = "range"
            norm_range = r
        else:
            norm_val = None
            norm_range = None

        try:
            si_target = float(self.si_target_entry.get())
        except Exception:
            si_target = 520.0

        gauss_mode = self.gauss_mode_var.get()
        if gauss_mode == 'fixed':
            source_choice = self.source_var.get()
            if source_choice == 'range':
                instrument_source = "Si"
                r = parse_range(self.source_range_entry.get())
                if r is None or r[0] == r[1]:
                    messagebox.showerror("Peak Range error",
                                         f"Invalid peak range entry for estimation: '{self.source_range_entry.get()}'")
                    return None
                si_r = r
                manual_sigma = None
            else:
                instrument_source = "Manual"
                fwhm_text = self.manual_fwhm_entry.get().strip()
                fwhm_val = parse_positive_float(fwhm_text)
                if fwhm_val is None:
                    messagebox.showerror("Invalid FWHM",
                                         f"Spectral Resolution (FWHM) could not be parsed from: '{fwhm_text}'. Enter a positive number (e.g. 2.4).")
                    return None
                manual_sigma = fwhm_val / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        else:
            instrument_source = "Manual"
            manual_sigma = None

        peaks: List[PeakSpec] = []
        for le, re, ge, ae in self.peak_entries:
            label = le.get().strip()
            r = parse_range(re.get())
            fr = parse_range(ge.get())  # None if blank -> robust FWHM guess
            ar = parse_range(ae.get()) if ae.get().strip() != "" else None
            if not label or r is None or r[0] == r[1]:
                messagebox.showerror("Peak entry error",
                                     f"Invalid peak specification:\nLabel='{label}' Center Range='{re.get()}'\nEnsure non-empty label and two numbers.")
                return None
            if fr is not None:
                if fr[0] < 0 or fr[0] == fr[1]:
                    messagebox.showerror("FWHM range error",
                                         f"Invalid FWHM range for '{label}': '{ge.get()}'. Provide two numbers like '0 20' with min < max and min >= 0, or leave blank for auto.")
                    return None
            if ar is not None:
                if ar[0] < 0 or ar[0] == ar[1]:
                    messagebox.showerror("Amplitude range error",
                                         f"Invalid Amplitude range for '{label}': '{ae.get()}'. Provide two numbers like '0 1000' with min < max and min >= 0. Leave blank for auto.")
                    return None
            peaks.append(PeakSpec(label=label, center_range=r, fwhm_range=fr, amplitude_range=ar))

        cfg = FitConfig(
            instrument_source=instrument_source,
            si_range=si_r,
            fit_range=fit_r,
            manual_instr_sigma=manual_sigma,
            peaks=peaks,
            enforce_pure_instrument_gauss=(gauss_mode == "fixed"),
            baseline_model=self.baseline_var.get(),
            normalize=norm_val,
            normalize_range=norm_range,
            align_si=self.align_si_var.get(),
            si_target_cm1=si_target,
            save_plots=self.save_plots_var.get(),
            show_plots=False,
            summary_file="fit_summary.csv",
            verbose=True,
        )
        
        # Validate configuration
        is_valid, error_msg = cfg.validate()
        if not is_valid:
            messagebox.showerror("Configuration Error", f"Invalid configuration:\n{error_msg}")
            return None

        # Warn if peaks are outside the fit range
        out_peaks = []
        for p in cfg.peaks:
            if p.center_range[1] < cfg.fit_range[0] or p.center_range[0] > cfg.fit_range[1]:
                out_peaks.append(f"'{p.label}' {p.center_range}")
        if out_peaks:
            messagebox.showwarning("Peak outside fit range",
                                   "These peaks are outside the fit range and will be ignored by the fit:\n" +
                                   "\n".join(out_peaks))

        return cfg

    def _set_running(self, running: bool, is_batch: bool):
        """Enable/disable UI during background work."""
        self._is_running = running
        try:
            # Disable all interactive controls on the left except status/progress
            self._set_children_state(self.left, "disabled" if running else "normal",
                                     exclude_widgets={self.cancel_btn})
            # Explicitly manage Generate/Cancel buttons and progress
            if running:
                self.generate_btn.configure(text="Running...", state="disabled")
                self.cancel_btn.grid()
                if is_batch:
                    self.show_batch_progress(True)
            else:
                self.generate_btn.configure(text="Generate", state="normal")
                self.cancel_btn.grid_remove()
                self.show_batch_progress(False)
        except Exception:
            pass

    def _set_children_state(self, parent: tk.Widget, state: str, exclude_widgets: set[tk.Widget] = set()):
        """Recursively set state for child widgets that support it."""
        for child in parent.winfo_children():
            if child in exclude_widgets:
                continue
            try:
                # Some widgets (like Labels) don't have 'state'
                child.configure(state=state)
            except Exception:
                pass
            # Recurse into frames/containers
            self._set_children_state(child, state, exclude_widgets)

    def on_cancel(self):
        """Request cancellation of background work."""
        if self.cancel_event and not self.cancel_event.is_set():
            self.cancel_event.set()
            self.update_status("Cancellation requested...")
            self.cancel_btn.configure(state="disabled")

    def on_generate(self):
        """Handle Generate button click."""
        if self._is_running:
            return  # Ignore while running

        self.update_status("Building configuration...")
        cfg = self.build_config()
        if cfg is None:
            self.update_status("Configuration error")
            return
        
        mode = self.mode_var.get()
        wd = self.dir_entry.get()

        # Prepare for background execution
        self.cancel_event = threading.Event()

        if mode == "single":
            file_path = self.file_entry.get()
            if not file_path:
                messagebox.showerror("Missing file", "Please select a file to process.")
                self.update_status("No file selected")
                return
            if not os.path.isfile(file_path):
                messagebox.showerror("File not found", f"Could not find:\n{file_path}")
                self.update_status("File not found")
                return

            self._set_running(True, is_batch=False)
            self.update_status(f"Processing {os.path.basename(file_path)}...")

            def worker():
                try:
                    result = process_file(file_path, cfg)
                    out_txt = save_per_file_outputs(result, os.path.dirname(file_path), cfg)
                    report_path = os.path.join(os.path.dirname(file_path), f"{result['input_filename']}_fit_report.txt")
                    self.after(0, self._on_single_done, True, result, out_txt, report_path, "")
                except Exception as e:
                    self.after(0, self._on_single_done, False, None, "", "", str(e))

            self.worker_thread = threading.Thread(target=worker, daemon=True)
            self.worker_thread.start()

        else:
            if not require_dir(wd):
                self.update_status("Invalid directory")
                return

            self._set_running(True, is_batch=True)
            self.update_status("Running batch processing...")

            def worker_batch():
                try:
                    summary_df, saved_txt, saved_plots = run_batch(
                        wd, cfg, progress_callback=self._threadsafe_progress, cancel_event=self.cancel_event
                    )
                    self.after(0, self._on_batch_done, True, summary_df, saved_txt, saved_plots, "")
                except Exception as e:
                    self.after(0, self._on_batch_done, False, pd.DataFrame(), [], [], str(e))

            self.worker_thread = threading.Thread(target=worker_batch, daemon=True)
            self.worker_thread.start()

    def _on_single_done(self, ok: bool, result, out_txt: str, report_path: str, error: str):
        """Finalize single-file processing on the main thread."""
        self._set_running(False, is_batch=False)
        if ok and result is not None:
            msg = f"Finished processing.\nSaved data: {out_txt}"
            if os.path.exists(report_path):
                msg += f"\nSaved report: {report_path}"
            self.current_fig = result["fig"]
            self.current_df = result["dataframe"]
            self.current_input_filename = result["input_filename"]
            self.render_figure(self.current_fig)
            self.update_status("Processing complete")
            messagebox.showinfo("Done", msg)
        else:
            messagebox.showerror("Processing error", error or "Unknown error")
            self.update_status("Processing failed")

    def _on_batch_done(self, ok: bool, summary_df: pd.DataFrame, saved_txt: List[str], saved_plots: List[str], error: str):
        """Finalize batch processing on the main thread."""
        cancelled = self.cancel_event.is_set() if self.cancel_event else False
        self._set_running(False, is_batch=True)
        if not ok:
            messagebox.showerror("Batch error", error or "Unknown error")
            self.update_status("Batch processing failed")
            return

        if cancelled:
            messagebox.showinfo("Batch", "Batch cancelled. Partial results may have been saved.")
            self.update_status("Batch cancelled")
        elif summary_df.empty:
            messagebox.showinfo("Batch", "No files processed.")
            self.update_status("No files processed")
        else:
            wd = self.dir_entry.get()
            msg = f"Processed {len(saved_txt)} files.\nSummary saved to: {os.path.join(wd, 'fit_summary.csv')}"
            if saved_plots:
                msg += f"\nSaved {len(saved_plots)} plot(s)."
            self.update_status(f"Batch complete: {len(saved_txt)} files")
            messagebox.showinfo("Batch complete", msg)
            
            # Show last file plot (best-effort)
            try:
                last_txt = saved_txt[-1]
                last_raw = last_txt.replace("_fit_plot.txt", ".txt")
                res = process_file(last_raw, summary_df.attrs.get("cfg", None) or FitConfig(
                    instrument_source="Manual", fit_range=(300.0, 480.0), peaks=[]
                ))
                self.current_fig = res["fig"]
                self.current_df = res["dataframe"]
                self.current_input_filename = res["input_filename"]
                self.render_figure(self.current_fig)
            except Exception as e:
                logger.warning(f"Could not display last plot: {e}")

    def render_figure(self, fig):
        """Render matplotlib figure in the canvas."""
        try:
            fig.patch.set_facecolor("white")
            for ax in fig.axes:
                ax.set_facecolor("white")
            fig.subplots_adjust(top=0.93)
        except Exception as e:
            logger.warning(f"Figure styling error: {e}")
        
        for child in self.canvas_frame.winfo_children():
            child.destroy()
        self.canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        self.canvas.draw()
        widget = self.canvas.get_tk_widget()
        widget.configure(bg="white", highlightthickness=0)
        widget.pack(fill=tk.BOTH, expand=True, side=tk.TOP, anchor="n", pady=0)

    def copy_image(self):
        """Copy current plot image to clipboard."""
        if not self.current_fig:
            messagebox.showwarning("No image", "Nothing to copy yet.")
            return
        
        buf = io.BytesIO()
        self.current_fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
        png_data = buf.getvalue()
        
        if sys.platform.startswith("win"):
            try:
                import win32clipboard
                from PIL import Image
                buf.seek(0)
                image = Image.open(buf).convert("RGB")
                output = io.BytesIO()
                image.save(output, "BMP")
                data = output.getvalue()[14:]
                output.close()
                win32clipboard.OpenClipboard()
                win32clipboard.EmptyClipboard()
                win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)
                win32clipboard.CloseClipboard()
                messagebox.showinfo("Copied", "Image copied to clipboard.")
                self.update_status("Image copied to clipboard")
                return
            except Exception as e:
                logger.warning(f"Windows clipboard copy failed: {e}")
        
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        with open(tmp.name, "wb") as f:
            f.write(png_data)
        self.clipboard_clear()
        self.clipboard_append(tmp.name)
        messagebox.showinfo("Copied (path)", f"Temporary image saved and its path copied:\n{tmp.name}")
        self.update_status("Image path copied to clipboard")

    def copy_data(self):
        """Copy current data to clipboard as TSV."""
        if self.current_df is None:
            messagebox.showwarning("No data", "No data to copy.")
            return
        
        tsv = self.current_df.to_csv(sep="\t", index=False)
        self.clipboard_clear()
        self.clipboard_append(tsv)
        messagebox.showinfo("Copied", "Data copied to clipboard (TSV).")
        self.update_status("Data copied to clipboard")

    def save_plot(self):
        """Save current plot to file."""
        if not self.current_fig:
            messagebox.showwarning("No image", "Nothing to save yet.")
            return
        
        initialdir = self.dir_entry.get() if os.path.isdir(self.dir_entry.get()) else os.getcwd()
        default_name = (self.current_input_filename or "plot") + "_fit.png"
        f = filedialog.asksaveasfilename(
            title="Save plot",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")],
            initialdir=initialdir,
            initialfile=default_name,
        )
        if f:
            self.current_fig.savefig(f, dpi=200, bbox_inches="tight", facecolor="white")
            messagebox.showinfo("Saved", f"Plot saved to:\n{f}")
            self.update_status(f"Plot saved: {os.path.basename(f)}")
            logger.info(f"Plot saved to {f}")


if __name__ == "__main__":
    app = RamanFitGUI()
    app.mainloop()