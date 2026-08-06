#!/usr/bin/env python3
"""
Loop Crossfade
==============
A standalone tool that replicates TwistedWave's "Loop Crossfade" effect:
it blends the tail of an audio file into its head so the file loops back
on itself with no click or pop at the seam -- plus:

  * Multi-format I/O: WAV, AIFF, MP3, MP4/M4A, FLAC (decode/encode via ffmpeg)
  * Optional transient-snap: finds the strongest attack near the start and
    end of the clip and trims to it, so the loop begins/ends on the beat
    or articulation instead of an arbitrary sample boundary
  * Automatic crossfade-length detection (or set it manually)
  * A dark, flat, modern GUI

Works on Windows, macOS, and Linux.

--------------------------------------------------------------------------
DEPENDENCIES
--------------------------------------------------------------------------
    pip install numpy
    Tkinter ships with the standard python.org installers on Mac/Windows,
    no separate install needed there.

    ffmpeg is required for any format other than plain WAV (MP3, MP4/M4A,
    FLAC, AIFF). Plain WAV works with zero extra dependencies.
        macOS:    brew install ffmpeg
        Windows:  https://www.gyan.dev/ffmpeg/builds/ (add the bin folder
                  to your PATH), or `winget install ffmpeg`
        Linux:    sudo apt install ffmpeg (or your distro's package manager)

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
GUI:
    python loop_crossfade_app.py

Command line:
    python loop_crossfade_app.py in.mp3 out.wav --auto-xfade
    python loop_crossfade_app.py in.wav out.flac --xfade 0.35 --curve linear
    python loop_crossfade_app.py in.wav out.wav --snap-transients --transient-window 0.25 --auto-xfade

--------------------------------------------------------------------------
PACKAGING AS A NATIVE APP
--------------------------------------------------------------------------
    pip install pyinstaller
    pyinstaller --onefile --windowed loop_crossfade_app.py
The result in dist/ is a standalone double-clickable app (still needs
ffmpeg on the target machine for non-WAV formats -- see notes above).
"""

import os
import sys
import wave
import shutil
import argparse
import tempfile
import subprocess
import numpy as np


def _find_ffmpeg():
    """Looks for a copy of ffmpeg bundled alongside a packaged (frozen)
    build of this app first, so a PyInstaller build with ffmpeg embedded
    needs nothing installed on the end user's machine. Falls back to
    whatever's on the system PATH for normal `python loop_crossfade_app.py`
    usage during development."""
    if getattr(sys, "frozen", False):
        bundle_dir = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
        candidate = os.path.join(bundle_dir, "ffmpeg.exe" if os.name == "nt" else "ffmpeg")
        if os.path.exists(candidate):
            return candidate
    return shutil.which("ffmpeg")


FFMPEG_PATH = _find_ffmpeg()
SUPPORTED_EXTS = {".wav", ".aif", ".aiff", ".mp3", ".mp4", ".m4a", ".flac"}

FFMPEG_ENCODE_ARGS = {
    ".mp3":  ["-c:a", "libmp3lame", "-q:a", "2"],
    ".flac": ["-c:a", "flac"],
    ".mp4":  ["-c:a", "aac", "-b:a", "192k"],
    ".m4a":  ["-c:a", "aac", "-b:a", "192k"],
    ".aif":  ["-c:a", "pcm_s16le"],
    ".aiff": ["-c:a", "pcm_s16le"],
}


def ffmpeg_available():
    return FFMPEG_PATH is not None


def _require_ffmpeg(context):
    if not ffmpeg_available():
        raise RuntimeError(
            f"ffmpeg is required to {context}, but wasn't found on your PATH.\n"
            "Install it and try again:\n"
            "  macOS:   brew install ffmpeg\n"
            "  Windows: https://www.gyan.dev/ffmpeg/builds/  (add to PATH)\n"
            "           or: winget install ffmpeg\n"
            "  Linux:   sudo apt install ffmpeg"
        )


# ---------------------------------------------------------------------------
# WAV I/O (stdlib `wave` module -- the common intermediate format)
# ---------------------------------------------------------------------------

def read_wav(path):
    """Returns (data, samplerate, sampwidth); data is float64 in [-1, 1],
    shape (n_samples, n_channels)."""
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        samplerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float64)
        data = (data - 128.0) / 128.0
    elif sampwidth == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif sampwidth == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        as_int32 = (b[:, 0].astype(np.int32)
                    | (b[:, 1].astype(np.int32) << 8)
                    | (b[:, 2].astype(np.int32) << 16))
        as_int32 = np.where(as_int32 & 0x800000, as_int32 - 0x1000000, as_int32)
        data = as_int32.astype(np.float64) / 8388608.0
    elif sampwidth == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth} bytes")

    data = data.reshape(-1, n_channels)
    return data, samplerate, sampwidth


def write_wav(path, data, samplerate, sampwidth):
    data = np.clip(data, -1.0, 1.0)
    if data.ndim == 1:
        data = data[:, None]
    n_channels = data.shape[1]

    if sampwidth == 1:
        out = np.round(data * 127.0 + 128.0).astype(np.uint8)
        raw = out.tobytes()
    elif sampwidth == 2:
        out = np.round(data * 32767.0).astype("<i2")
        raw = out.tobytes()
    elif sampwidth == 3:
        as_int = np.round(data * 8388607.0).astype(np.int32)
        b0 = (as_int & 0xFF).astype(np.uint8)
        b1 = ((as_int >> 8) & 0xFF).astype(np.uint8)
        b2 = ((as_int >> 16) & 0xFF).astype(np.uint8)
        raw = np.stack([b0, b1, b2], axis=-1).tobytes()
    elif sampwidth == 4:
        out = np.round(data * 2147483647.0).astype("<i4")
        raw = out.tobytes()
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth} bytes")

    with wave.open(path, "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(samplerate)
        wf.writeframes(raw)


# ---------------------------------------------------------------------------
# Multi-format decode / encode (ffmpeg for anything that isn't plain WAV)
# ---------------------------------------------------------------------------

def decode_to_pcm(path):
    """Decode any supported input format to (data, sr, sampwidth) via a
    24-bit PCM WAV intermediate. Plain WAV files skip ffmpeg entirely."""
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXTS:
        raise ValueError(f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTS)}")

    if ext == ".wav":
        return read_wav(path)

    _require_ffmpeg(f"read '{ext}' files")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_wav = os.path.join(tmp, "decoded.wav")
        cmd = [FFMPEG_PATH, "-y", "-i", path, "-c:a", "pcm_s24le", tmp_wav]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed to decode '{path}':\n{result.stderr.decode(errors='ignore')}")
        return read_wav(tmp_wav)


def encode_from_pcm(data, sr, sampwidth, out_path):
    """Write processed PCM data out in whatever format out_path's
    extension indicates. Plain WAV skips ffmpeg entirely."""
    ext = os.path.splitext(out_path)[1].lower()
    if ext not in SUPPORTED_EXTS:
        raise ValueError(f"Unsupported output type '{ext}'. Supported: {sorted(SUPPORTED_EXTS)}")

    if ext == ".wav":
        write_wav(out_path, data, sr, sampwidth)
        return

    _require_ffmpeg(f"write '{ext}' files")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_wav = os.path.join(tmp, "processed.wav")
        write_wav(tmp_wav, data, sr, sampwidth)
        codec_args = FFMPEG_ENCODE_ARGS.get(ext, [])
        cmd = [FFMPEG_PATH, "-y", "-i", tmp_wav, *codec_args, out_path]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed to encode '{out_path}':\n{result.stderr.decode(errors='ignore')}")


# ---------------------------------------------------------------------------
# Transient detection (for beat / articulation alignment)
# ---------------------------------------------------------------------------

def _onset_frames(mono, sr, hop_sec=0.005, frame_sec=0.010):
    hop = max(1, int(hop_sec * sr))
    frame = max(hop * 2, int(frame_sec * sr))
    n = len(mono)
    if n <= frame:
        return np.array([]), np.array([])
    starts = np.arange(0, n - frame, hop)
    idx = starts[:, None] + np.arange(frame)[None, :]
    frames = mono[idx]
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    centers = starts + frame // 2
    return rms, centers


def find_strongest_transient(data, sr, search_window_sec, from_start=True):
    """Returns a sample index: the strongest attack (energy rise) within
    the first/last `search_window_sec` of the clip. Falls back to the very
    edge (0 or len) if nothing clearly stands out."""
    mono = data.mean(axis=1) if data.ndim > 1 else data
    n = len(mono)
    window_samples = min(int(search_window_sec * sr), n)
    if window_samples < int(0.02 * sr):
        return 0 if from_start else n

    if from_start:
        seg = mono[:window_samples]
        base_offset = 0
    else:
        seg = mono[n - window_samples:]
        base_offset = n - window_samples

    rms, centers = _onset_frames(seg, sr)
    if len(rms) < 2:
        return 0 if from_start else n

    flux = np.diff(rms, prepend=rms[0])
    flux = np.clip(flux, 0, None)
    if flux.max() <= 1e-9:
        return 0 if from_start else n

    best = int(np.argmax(flux))
    return int(base_offset + centers[best])


def snap_to_transients(data, sr, search_window_sec):
    """Trims the clip so it starts at the strongest onset found near the
    beginning, and ends right at the strongest onset found near the end
    (dropping any lead-in silence and trailing decay past that point).
    Returns (trimmed_data, samples_trimmed_from_start, samples_trimmed_from_end)."""
    n = data.shape[0]
    if search_window_sec <= 0 or n / sr < 0.05:
        return data, 0, 0

    window = min(search_window_sec, (n / sr) / 2.5)
    start_idx = find_strongest_transient(data, sr, window, from_start=True)
    end_idx = find_strongest_transient(data, sr, window, from_start=False)

    if end_idx <= start_idx + int(0.01 * sr):
        return data, 0, 0

    trimmed = data[start_idx:end_idx]
    return trimmed, start_idx, n - end_idx


# ---------------------------------------------------------------------------
# Core crossfade DSP
# ---------------------------------------------------------------------------

def _crossfade_core(data, xfade_n, curve):
    """data: 2D float64 array (n_samples, n_channels). Returns the
    crossfaded result as a 2D array."""
    n = data.shape[0]
    xfade_n = max(1, min(xfade_n, n // 2))

    head = data[:xfade_n]
    tail = data[n - xfade_n:]
    middle = data[xfade_n:n - xfade_n]

    t = np.linspace(0.0, 1.0, xfade_n, endpoint=False)
    if curve == "equal_power":
        fade_in = np.sin(t * np.pi / 2.0)
        fade_out = np.cos(t * np.pi / 2.0)
    elif curve == "linear":
        fade_in = t
        fade_out = 1.0 - t
    else:
        raise ValueError("curve must be 'equal_power' or 'linear'")

    blended = head * fade_in[:, None] + tail * fade_out[:, None]
    return np.concatenate([blended, middle], axis=0)


def loop_crossfade(data, samplerate, xfade_seconds, curve="equal_power"):
    was_1d = (data.ndim == 1)
    if was_1d:
        data = data[:, None]
    xfade_n = int(round(xfade_seconds * samplerate))
    result = _crossfade_core(data, xfade_n, curve)
    return result[:, 0] if was_1d else result


# ---------------------------------------------------------------------------
# Automatic crossfade-length selection
# ---------------------------------------------------------------------------

def _seam_cost(data, xfade_n, sr, length_penalty=0.06):
    """Lower is better. This is NOT based on the crossfaded output's edge
    samples -- measuring those is width-dependent in a way that trivially
    favors longer crossfades (finer ramp resolution shrinks the measured
    edge derivative without the audio actually sounding any better).

    Instead this scores how well-matched the head and tail segments being
    blended actually are:
      - waveform correlation (do they line up in phase/shape?)
      - RMS energy match (is the blend a similar loudness throughout?)
    plus a small penalty for length, since a shorter crossfade preserves
    more of the original transient content and TwistedWave-style loop
    crossfades are typically tens to a few hundred ms, not seconds.
    """
    n = data.shape[0]
    xfade_n = max(1, min(xfade_n, n // 2))
    head = data[:xfade_n]
    tail = data[n - xfade_n:]

    h = head.reshape(xfade_n, -1)
    tl = tail.reshape(xfade_n, -1)

    h_c = h - h.mean(axis=0, keepdims=True)
    t_c = tl - tl.mean(axis=0, keepdims=True)
    denom = (np.linalg.norm(h_c, axis=0) * np.linalg.norm(t_c, axis=0)) + 1e-9
    corr = np.mean(np.sum(h_c * t_c, axis=0) / denom)  # -1..1, higher = better match

    rms_h = np.sqrt(np.mean(h ** 2) + 1e-12)
    rms_t = np.sqrt(np.mean(tl ** 2) + 1e-12)
    energy_mismatch = abs(rms_h - rms_t) / max(rms_h, rms_t, 1e-9)

    length_cost = length_penalty * (xfade_n / sr)

    return float(-corr + energy_mismatch + length_cost)


def auto_select_xfade(data, sr, curve="equal_power",
                       min_sec=0.008, max_sec=0.4, n_candidates=24):
    """Searches crossfade durations on a log scale and picks the one that
    best matches head/tail content (see _seam_cost) -- i.e. the shortest
    crossfade that still blends two genuinely similar-sounding regions,
    which is what makes a crossfade perceptually "disappear."
    `curve` is accepted for API symmetry with loop_crossfade but doesn't
    affect the search, since head/tail similarity is curve-independent."""
    n = data.shape[0]
    max_sec = max(min_sec * 2, min(max_sec, (n / sr) / 2 - 0.001))
    if max_sec <= min_sec:
        return max(0.005, (n / sr) / 4)

    candidates = np.geomspace(min_sec, max_sec, n_candidates)
    best_sec, best_cost = None, None
    for c in candidates:
        xfade_n = max(1, min(int(round(c * sr)), n // 2))
        cost = _seam_cost(data, xfade_n, sr)
        if best_cost is None or cost < best_cost:
            best_cost, best_sec = cost, c
    return float(best_sec)


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

def process_file(in_path, out_path, xfade_seconds=None, curve="equal_power",
                  snap_transients=False, transient_window=0.25, auto_xfade=False):
    data, sr, sampwidth = decode_to_pcm(in_path)
    if data.ndim == 1:
        data = data[:, None]

    start_trim = end_trim = 0
    if snap_transients:
        data, start_trim, end_trim = snap_to_transients(data, sr, transient_window)

    if auto_xfade or xfade_seconds is None:
        xfade_seconds = auto_select_xfade(data, sr, curve=curve)

    result = loop_crossfade(data, sr, xfade_seconds, curve)
    encode_from_pcm(result, sr, sampwidth, out_path)

    return {
        "n_samples": result.shape[0],
        "samplerate": sr,
        "xfade_seconds": xfade_seconds,
        "start_trim_samples": start_trim,
        "end_trim_samples": end_trim,
    }


# ---------------------------------------------------------------------------
# GUI (Tkinter, dark theme)
# ---------------------------------------------------------------------------

def launch_gui():
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    # ---- palette: flat, dark, modern (in the spirit of Audacity 4 / other
    # dark-themed pro-audio apps -- not a pixel copy of any specific app) ----
    BG = "#1e1f22"
    PANEL = "#26282c"
    FIELD_BG = "#2c2f34"
    FG = "#e6e6e8"
    MUTED = "#9a9da3"
    ACCENT = "#4f8cff"
    ACCENT_HOVER = "#6da0ff"
    BORDER = "#37393e"

    root = tk.Tk()
    root.title("Loop Crossfade")
    root.geometry("520x460")
    root.minsize(520, 460)
    root.configure(bg=BG)

    style = ttk.Style(root)
    style.theme_use("clam")

    style.configure("TFrame", background=BG)
    style.configure("Panel.TFrame", background=PANEL)
    style.configure("TLabel", background=BG, foreground=FG, font=("Segoe UI", 10))
    style.configure("Muted.TLabel", background=BG, foreground=MUTED, font=("Segoe UI", 9))
    style.configure("Heading.TLabel", background=BG, foreground=FG, font=("Segoe UI", 13, "bold"))
    style.configure("TCheckbutton", background=BG, foreground=FG, font=("Segoe UI", 10))
    style.map("TCheckbutton", background=[("active", BG)], foreground=[("disabled", MUTED)])
    style.configure("TEntry", fieldbackground=FIELD_BG, foreground=FG,
                     insertcolor=FG, bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER)
    style.configure("TCombobox", fieldbackground=FIELD_BG, background=FIELD_BG,
                     foreground=FG, arrowcolor=FG, bordercolor=BORDER)
    style.map("TCombobox", fieldbackground=[("readonly", FIELD_BG)],
              foreground=[("readonly", FG)])
    style.configure("TButton", background=PANEL, foreground=FG, borderwidth=0,
                     focusthickness=0, padding=8, font=("Segoe UI", 10))
    style.map("TButton", background=[("active", BORDER)])
    style.configure("Accent.TButton", background=ACCENT, foreground="#ffffff",
                     borderwidth=0, padding=10, font=("Segoe UI", 11, "bold"))
    style.map("Accent.TButton", background=[("active", ACCENT_HOVER)])

    in_path_var = tk.StringVar()
    out_path_var = tk.StringVar()
    xfade_var = tk.StringVar(value="0.30")
    curve_var = tk.StringVar(value="Equal power")
    auto_xfade_var = tk.BooleanVar(value=True)
    snap_var = tk.BooleanVar(value=False)
    window_var = tk.StringVar(value="0.25")
    status_var = tk.StringVar(value="Choose an audio file to begin.")

    filetypes = [
        ("Audio files", "*.wav *.aif *.aiff *.mp3 *.mp4 *.m4a *.flac"),
        ("All files", "*.*"),
    ]

    def choose_input():
        path = filedialog.askopenfilename(title="Choose audio file", filetypes=filetypes)
        if path:
            in_path_var.set(path)
            if not out_path_var.get():
                root_name, ext = os.path.splitext(path)
                out_path_var.set(root_name + "_loop" + ext)

    def choose_output():
        path = filedialog.asksaveasfilename(
            title="Save processed file as", defaultextension=".wav",
            filetypes=[("WAV", "*.wav"), ("AIFF", "*.aiff"), ("MP3", "*.mp3"),
                       ("MP4/M4A", "*.m4a"), ("FLAC", "*.flac")],
        )
        if path:
            out_path_var.set(path)

    def toggle_xfade_entry(*_):
        xfade_entry.configure(state="disabled" if auto_xfade_var.get() else "normal")

    def toggle_window_entry(*_):
        window_entry.configure(state="normal" if snap_var.get() else "disabled")

    def run_process():
        in_path, out_path = in_path_var.get(), out_path_var.get()
        if not in_path or not out_path:
            messagebox.showerror("Loop Crossfade", "Choose an input file and an output location first.")
            return

        xfade_seconds = None
        if not auto_xfade_var.get():
            try:
                xfade_seconds = float(xfade_var.get())
                if xfade_seconds <= 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Loop Crossfade", "Crossfade duration must be a positive number of seconds.")
                return

        try:
            transient_window = float(window_var.get()) if snap_var.get() else 0.25
        except ValueError:
            messagebox.showerror("Loop Crossfade", "Transient search window must be a number of seconds.")
            return

        curve = "equal_power" if curve_var.get() == "Equal power" else "linear"

        try:
            status_var.set("Processing...")
            root.update_idletasks()
            info = process_file(
                in_path, out_path,
                xfade_seconds=xfade_seconds,
                curve=curve,
                snap_transients=snap_var.get(),
                transient_window=transient_window,
                auto_xfade=auto_xfade_var.get(),
            )
            duration = info["n_samples"] / info["samplerate"]
            msg = (f"Done -- {duration:.2f}s loop saved to:\n{out_path}\n"
                   f"Crossfade used: {info['xfade_seconds']*1000:.0f} ms")
            if snap_var.get():
                msg += (f"\nTrimmed to transients: "
                        f"{info['start_trim_samples']/info['samplerate']*1000:.0f} ms from start, "
                        f"{info['end_trim_samples']/info['samplerate']*1000:.0f} ms from end")
            status_var.set(msg)
        except Exception as e:
            status_var.set("Failed. See error dialog.")
            messagebox.showerror("Loop Crossfade", str(e))

    outer = ttk.Frame(root, padding=20)
    outer.pack(fill="both", expand=True)

    ttk.Label(outer, text="Loop Crossfade", style="Heading.TLabel").pack(anchor="w", pady=(0, 2))
    ttk.Label(outer, text="Seamless, beat-aware loop points.", style="Muted.TLabel").pack(anchor="w", pady=(0, 16))

    row = ttk.Frame(outer); row.pack(fill="x", pady=4)
    ttk.Label(row, text="Input", width=8).pack(side="left")
    ttk.Entry(row, textvariable=in_path_var).pack(side="left", fill="x", expand=True, padx=6)
    ttk.Button(row, text="Browse", command=choose_input).pack(side="left")

    row = ttk.Frame(outer); row.pack(fill="x", pady=4)
    ttk.Label(row, text="Save as", width=8).pack(side="left")
    ttk.Entry(row, textvariable=out_path_var).pack(side="left", fill="x", expand=True, padx=6)
    ttk.Button(row, text="Browse", command=choose_output).pack(side="left")

    ttk.Frame(outer, height=1, style="Panel.TFrame").pack(fill="x", pady=16)

    row = ttk.Frame(outer); row.pack(fill="x", pady=4)
    ttk.Checkbutton(row, text="Snap loop points to transients (beat / articulation alignment)",
                     variable=snap_var, command=toggle_window_entry).pack(side="left")

    row = ttk.Frame(outer); row.pack(fill="x", pady=(0, 12))
    ttk.Label(row, text="Search window (s):", style="Muted.TLabel").pack(side="left", padx=(24, 6))
    window_entry = ttk.Entry(row, textvariable=window_var, width=8, state="disabled")
    window_entry.pack(side="left")

    row = ttk.Frame(outer); row.pack(fill="x", pady=4)
    ttk.Checkbutton(row, text="Auto-detect crossfade length", variable=auto_xfade_var,
                     command=toggle_xfade_entry).pack(side="left")

    row = ttk.Frame(outer); row.pack(fill="x", pady=(0, 4))
    ttk.Label(row, text="Manual crossfade (s):", style="Muted.TLabel").pack(side="left", padx=(24, 6))
    xfade_entry = ttk.Entry(row, textvariable=xfade_var, width=8, state="disabled")
    xfade_entry.pack(side="left")

    row = ttk.Frame(outer); row.pack(fill="x", pady=(8, 4))
    ttk.Label(row, text="Curve:", style="Muted.TLabel").pack(side="left", padx=(24, 6))
    ttk.Combobox(row, textvariable=curve_var, values=["Equal power", "Linear"],
                 state="readonly", width=14).pack(side="left")

    ttk.Frame(outer, height=1, style="Panel.TFrame").pack(fill="x", pady=16)

    ttk.Button(outer, text="Process & Save", style="Accent.TButton",
               command=run_process).pack(fill="x", pady=(0, 12))

    ttk.Label(outer, textvariable=status_var, style="Muted.TLabel",
              wraplength=470, justify="left").pack(anchor="w", fill="x")

    if not ffmpeg_available():
        ttk.Label(outer, text="Note: ffmpeg not found -- only plain WAV will work until it's installed.",
                  style="Muted.TLabel", foreground="#e2a33d").pack(anchor="w", pady=(8, 0))

    root.mainloop()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) == 1:
        launch_gui()
        return

    parser = argparse.ArgumentParser(
        description="Crossfade a clip's tail into its head so it loops seamlessly. "
                     "Supports WAV, AIFF, MP3, MP4/M4A, FLAC."
    )
    parser.add_argument("input", help="Path to input audio file")
    parser.add_argument("output", help="Path to write the processed audio file")
    parser.add_argument("--xfade", type=float, default=None,
                         help="Crossfade duration in seconds (ignored if --auto-xfade is set)")
    parser.add_argument("--auto-xfade", action="store_true",
                         help="Automatically choose the crossfade length that minimizes seam discontinuity")
    parser.add_argument("--curve", choices=["equal_power", "linear"], default="equal_power")
    parser.add_argument("--snap-transients", action="store_true",
                         help="Trim to the strongest transient near the start/end for beat/articulation alignment")
    parser.add_argument("--transient-window", type=float, default=0.25,
                         help="Search window in seconds for transient snapping (default: 0.25)")
    args = parser.parse_args()

    if args.xfade is None and not args.auto_xfade:
        args.auto_xfade = True  # sensible default if the user specified neither

    info = process_file(
        args.input, args.output,
        xfade_seconds=args.xfade, curve=args.curve,
        snap_transients=args.snap_transients, transient_window=args.transient_window,
        auto_xfade=args.auto_xfade,
    )
    dur = info["n_samples"] / info["samplerate"]
    print(f"Wrote {args.output}: {dur:.3f}s, crossfade {info['xfade_seconds']*1000:.0f} ms")
    if args.snap_transients:
        print(f"  trimmed {info['start_trim_samples']} samples from start, "
              f"{info['end_trim_samples']} from end")


if __name__ == "__main__":
    main()
