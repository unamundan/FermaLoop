#!/usr/bin/env python3
"""
Loop Crossfade
==============
A standalone tool that replicates TwistedWave's "Loop Crossfade" effect:
it blends the tail of an audio file into its head so the file loops back
on itself with no click or pop at the seam -- plus:

  * Multi-format I/O: WAV, AIFF, MP3, MP4/M4A, FLAC (decode/encode via ffmpeg)
  * A waveform view: drag to select a region, drag the edges to adjust it,
    crop to it, and preview before/after committing
  * Transport controls (Play/Pause, Stop, Rewind, Loop) with remappable
    keyboard shortcuts (see the "Keyboard Shortcuts..." button in-app)
  * Drag-and-drop file loading onto the window (falls back to Browse if
    tkinterdnd2 isn't installed -- see dependencies below)
  * Optional transient-snap: finds the strongest attack near the start and
    end of the clip and trims to it, so the loop begins/ends on the beat
    or articulation instead of an arbitrary sample boundary
  * Crossfade length: manual by default, or an auto-detect option you can
    switch on (finds the length where the head/tail actually sound alike)
  * A dark, flat, modern GUI

Works on Windows, macOS, and Linux.

--------------------------------------------------------------------------
DEPENDENCIES
--------------------------------------------------------------------------
    pip install numpy
    Tkinter ships with the standard python.org installers on Mac/Windows,
    no separate install needed there.

    For waveform playback (Play/Pause/Stop/Rewind/Loop):
        pip install sounddevice
    Without it, everything else still works -- playback buttons just show
    a one-time message telling you to install it.

    For drag-and-drop file loading:
        pip install tkinterdnd2
    Without it, use the Browse buttons instead -- nothing else is affected.

    For smoother (anti-aliased) waveform rendering:
        pip install Pillow
    Without it, the waveform still draws, just with plain 1px canvas lines
    instead of a supersampled, smoothed fill.

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

Command line (unaffected by the GUI changes above):
    python loop_crossfade_app.py in.mp3 out.wav --auto-xfade
    python loop_crossfade_app.py in.wav out.flac --xfade 0.35 --curve linear
    python loop_crossfade_app.py in.wav out.wav --snap-transients --transient-window 0.25 --auto-xfade

--------------------------------------------------------------------------
PACKAGING AS A NATIVE APP
--------------------------------------------------------------------------
    pip install pyinstaller sounddevice tkinterdnd2 Pillow
    pyinstaller --onefile --windowed --collect-all sounddevice --collect-all tkinterdnd2 loop_crossfade_app.py
The result in dist/ is a standalone double-clickable app (still needs
ffmpeg on the target machine for non-WAV formats, unless bundled -- see
the project's build.yml for a version that embeds ffmpeg too).
"""

import os
import re
import sys
import json
import time
import wave
import struct
import shutil
import argparse
import tempfile
import threading
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
    shape (n_samples, n_channels).

    This parses the RIFF/WAVE header directly rather than using the
    stdlib `wave` module, because `wave` did not support the
    WAVE_FORMAT_EXTENSIBLE header variant (format tag 0xFFFE) until
    Python 3.12 -- and ffmpeg commonly writes that variant for pcm_s24le
    output, which is exactly the intermediate format this app uses when
    decoding MP3/FLAC/MP4 inputs. Parsing it ourselves avoids depending
    on which Python version happened to build/run the app."""
    with open(path, "rb") as f:
        riff = f.read(12)
        if riff[0:4] != b"RIFF" or riff[8:12] != b"WAVE":
            raise ValueError(f"Not a valid WAV file: {path}")

        fmt_chunk = None
        data_chunk = None
        while fmt_chunk is None or data_chunk is None:
            header = f.read(8)
            if len(header) < 8:
                break
            chunk_id = header[0:4]
            chunk_size = struct.unpack("<I", header[4:8])[0]
            chunk_data = f.read(chunk_size)
            if chunk_size % 2 == 1:
                f.read(1)  # chunks are word-aligned; skip the pad byte
            if chunk_id == b"fmt ":
                fmt_chunk = chunk_data
            elif chunk_id == b"data":
                data_chunk = chunk_data

        if fmt_chunk is None or data_chunk is None:
            raise ValueError(f"WAV file is missing a 'fmt ' or 'data' chunk: {path}")

        (format_tag, n_channels, samplerate, _byte_rate,
         _block_align, bits_per_sample) = struct.unpack("<HHIIHH", fmt_chunk[:16])

        actual_format = format_tag
        if format_tag == 0xFFFE:  # WAVE_FORMAT_EXTENSIBLE
            if len(fmt_chunk) >= 26:
                # bytes 24:26 of the extended fmt chunk hold the first two
                # bytes of the SubFormat GUID, which encode the real tag
                # (1 = PCM, 3 = IEEE float) the same way the plain tag does.
                actual_format = struct.unpack("<H", fmt_chunk[24:26])[0]
            else:
                actual_format = 1  # malformed/truncated extension; assume PCM

        if actual_format not in (1, 3):
            raise ValueError(f"Unsupported WAV format tag: {actual_format}")

        if actual_format == 3 and bits_per_sample == 32:
            data = np.frombuffer(data_chunk, dtype="<f4").astype(np.float64)
            sampwidth = 3  # no float writer path; downstream write goes out as 24-bit PCM
        elif actual_format == 3 and bits_per_sample == 64:
            data = np.frombuffer(data_chunk, dtype="<f8").astype(np.float64)
            sampwidth = 3
        else:
            sampwidth = bits_per_sample // 8
            if sampwidth == 1:
                data = np.frombuffer(data_chunk, dtype=np.uint8).astype(np.float64)
                data = (data - 128.0) / 128.0
            elif sampwidth == 2:
                data = np.frombuffer(data_chunk, dtype="<i2").astype(np.float64) / 32768.0
            elif sampwidth == 3:
                b = np.frombuffer(data_chunk, dtype=np.uint8)
                b = b[: (len(b) // 3) * 3].reshape(-1, 3)
                as_int32 = (b[:, 0].astype(np.int32)
                            | (b[:, 1].astype(np.int32) << 8)
                            | (b[:, 2].astype(np.int32) << 16))
                as_int32 = np.where(as_int32 & 0x800000, as_int32 - 0x1000000, as_int32)
                data = as_int32.astype(np.float64) / 8388608.0
            elif sampwidth == 4:
                data = np.frombuffer(data_chunk, dtype="<i4").astype(np.float64) / 2147483648.0
            else:
                raise ValueError(f"Unsupported bits per sample: {bits_per_sample}")

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

# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

def _run_pipeline(data, sr, xfade_seconds, curve, snap_transients, transient_window, auto_xfade):
    """Shared core: snap -> (auto-)crossfade. Operates purely in memory so
    the GUI can run it on a cropped in-memory buffer without writing/
    re-reading a file, and process_file() below can run it on freshly
    decoded data -- same logic either way."""
    if data.ndim == 1:
        data = data[:, None]

    start_trim = end_trim = 0
    if snap_transients:
        data, start_trim, end_trim = snap_to_transients(data, sr, transient_window)

    if auto_xfade or xfade_seconds is None:
        xfade_seconds = auto_select_xfade(data, sr, curve=curve)

    result = loop_crossfade(data, sr, xfade_seconds, curve)
    return result, xfade_seconds, start_trim, end_trim


def process_file(in_path, out_path, xfade_seconds=None, curve="equal_power",
                  snap_transients=False, transient_window=0.25, auto_xfade=False):
    data, sr, sampwidth = decode_to_pcm(in_path)
    result, xfade_seconds, start_trim, end_trim = _run_pipeline(
        data, sr, xfade_seconds, curve, snap_transients, transient_window, auto_xfade)
    encode_from_pcm(result, sr, sampwidth, out_path)

    return {
        "n_samples": result.shape[0],
        "samplerate": sr,
        "xfade_seconds": xfade_seconds,
        "start_trim_samples": start_trim,
        "end_trim_samples": end_trim,
    }


# ---------------------------------------------------------------------------
# Keyboard shortcuts (user-remappable, persisted to a small JSON file)
# ---------------------------------------------------------------------------

DEFAULT_SHORTCUTS = {
    "play_pause": "space",
    "stop": "s",
    "rewind": "Home",
    "loop_toggle": "l",
    "crop": "c",
    "audition": "a",
    "undo": "Control-z",
    "redo": "Control-y",
    "zoom_in": "equal",
    "zoom_out": "minus",
    "zoom_fit": "0",
}

SHORTCUT_LABELS = {
    "play_pause": "Play / Pause",
    "stop": "Stop",
    "rewind": "Rewind",
    "loop_toggle": "Toggle Loop",
    "crop": "Crop to Selection",
    "audition": "Audition Loop",
    "undo": "Undo",
    "redo": "Redo",
    "zoom_in": "Zoom In",
    "zoom_out": "Zoom Out",
    "zoom_fit": "Zoom to Fit",
}

SHORTCUTS_PATH = os.path.join(os.path.expanduser("~"), ".loop_crossfade_shortcuts.json")


def load_shortcuts(path=SHORTCUTS_PATH):
    if os.path.exists(path):
        try:
            with open(path) as f:
                saved = json.load(f)
            merged = dict(DEFAULT_SHORTCUTS)
            merged.update({k: v for k, v in saved.items() if k in DEFAULT_SHORTCUTS})
            return merged
        except Exception:
            return dict(DEFAULT_SHORTCUTS)
    return dict(DEFAULT_SHORTCUTS)


def save_shortcuts(shortcuts, path=SHORTCUTS_PATH):
    try:
        with open(path, "w") as f:
            json.dump(shortcuts, f, indent=2)
    except OSError:
        pass  # non-fatal -- shortcuts just won't persist this session


# ---------------------------------------------------------------------------
# Waveform rendering helper (pure numpy, no GUI dependency)
# ---------------------------------------------------------------------------

def compute_waveform_peaks(data, target_width):
    """Returns (mins, maxs): a mono-mixed min/max amplitude envelope, one
    pair per pixel column, for fast waveform drawing regardless of clip
    length."""
    mono = data.mean(axis=1) if data.ndim > 1 else data
    n = len(mono)
    if n == 0:
        return np.zeros(1), np.zeros(1)
    target_width = max(1, min(target_width, n))
    edges = np.linspace(0, n, target_width + 1).astype(int)
    mins = np.empty(target_width)
    maxs = np.empty(target_width)
    for i in range(target_width):
        a, b = edges[i], max(edges[i] + 1, edges[i + 1])
        chunk = mono[a:b]
        mins[i] = chunk.min()
        maxs[i] = chunk.max()
    return mins, maxs


# ---------------------------------------------------------------------------
# Audio playback engine (sounddevice). Isolated from Tkinter so the
# looping/stop logic can be (and was) unit tested without real audio
# hardware -- see the callback structure, it mirrors that test exactly.
# ---------------------------------------------------------------------------

try:
    import sounddevice as _sd
    SOUNDDEVICE_AVAILABLE = True
except (ImportError, OSError):
    # OSError happens when the package is installed but its native
    # PortAudio library isn't found (a real, common case, especially on
    # Linux) -- either way, fall back gracefully instead of crashing.
    _sd = None
    SOUNDDEVICE_AVAILABLE = False


class AudioPlayer:
    """Selection-aware playback: play/pause/stop/rewind, optional
    continuous loop over [sel_start, sel_end). Works on whatever buffer
    is currently loaded (pre- or post-crop -- the GUI just calls load()
    again after cropping)."""

    def __init__(self):
        self.data = None          # float32, shape (n, channels)
        self.sr = 44100
        self.sel_start = 0
        self.sel_end = 0
        self.cursor = 0
        self.loop = False
        self.playing = False
        self.lock = threading.Lock()
        self.stream = None
        self.on_natural_stop = None  # optional callback, called from GUI thread via `after`

    def load(self, data, sr):
        self.stop()
        if data.ndim == 1:
            data = data[:, None]
        self.data = np.ascontiguousarray(data.astype(np.float32))
        self.sr = sr
        with self.lock:
            self.sel_start, self.sel_end, self.cursor = 0, len(self.data), 0

    def set_selection(self, start, end):
        with self.lock:
            if self.data is None:
                return
            self.sel_start = max(0, min(start, len(self.data)))
            self.sel_end = max(self.sel_start, min(end, len(self.data)))
            self.cursor = max(self.sel_start, min(self.cursor, self.sel_end))

    def set_loop(self, value):
        self.loop = bool(value)

    def _callback(self, outdata, frames, time_info, status):
        with self.lock:
            if self.data is None:
                outdata[:] = 0
                raise _sd.CallbackStop
            remaining = self.sel_end - self.cursor
            if remaining <= 0:
                if self.loop:
                    self.cursor = self.sel_start
                    remaining = self.sel_end - self.cursor
                else:
                    outdata[:] = 0
                    self.playing = False
                    raise _sd.CallbackStop
            n = min(frames, remaining)
            chunk = self.data[self.cursor:self.cursor + n]
            ch = outdata.shape[1]
            if chunk.shape[1] != ch:
                chunk = np.tile(chunk[:, :1], (1, ch)) if chunk.shape[1] == 1 else chunk[:, :ch]
            outdata[:n] = chunk
            self.cursor += n
            if n < frames:
                if self.loop:
                    self.cursor = self.sel_start
                    n2 = min(frames - n, self.sel_end - self.sel_start)
                    chunk2 = self.data[self.sel_start:self.sel_start + n2]
                    if chunk2.shape[1] != ch:
                        chunk2 = np.tile(chunk2[:, :1], (1, ch)) if chunk2.shape[1] == 1 else chunk2[:, :ch]
                    outdata[n:n + n2] = chunk2
                    outdata[n + n2:] = 0
                    self.cursor += n2
                else:
                    outdata[n:] = 0
                    self.playing = False
                    raise _sd.CallbackStop

    def play(self):
        if not SOUNDDEVICE_AVAILABLE or self.data is None:
            return
        with self.lock:
            if not (self.sel_start <= self.cursor < self.sel_end):
                self.cursor = self.sel_start
        channels = self.data.shape[1]
        self.stream = _sd.OutputStream(samplerate=self.sr, channels=channels,
                                        callback=self._callback, dtype="float32")
        self.stream.start()
        self.playing = True

    def pause(self):
        if self.stream is not None:
            try:
                self.stream.stop()
            except Exception:
                pass
        self.playing = False

    def stop(self):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        with self.lock:
            self.cursor = self.sel_start
        self.playing = False

    def rewind(self):
        with self.lock:
            self.cursor = self.sel_start

    def get_cursor(self):
        with self.lock:
            return self.cursor


# ---------------------------------------------------------------------------
# Optional drag-and-drop support (tkinterdnd2). The app works fully via the
# Browse buttons without it -- this is purely additive.
# ---------------------------------------------------------------------------

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    DND_AVAILABLE = True
except ImportError:
    DND_AVAILABLE = False

try:
    from PIL import Image, ImageDraw, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def _parse_dnd_paths(raw):
    """tkinterdnd2 gives dropped paths as a single string, space-separated,
    with any path containing spaces wrapped in {curly braces}."""
    tokens = re.findall(r"\{[^}]*\}|\S+", raw)
    return [t[1:-1] if t.startswith("{") and t.endswith("}") else t for t in tokens]


# ---------------------------------------------------------------------------
# GUI (Tkinter, dark theme, waveform editor)
# ---------------------------------------------------------------------------

# palette: flat, dark, modern (in the spirit of Audacity 4 / other
# dark-themed pro-audio apps -- not a pixel copy of any specific app)
BG = "#1e1f22"
PANEL = "#26282c"
FIELD_BG = "#2c2f34"
FG = "#e6e6e8"
MUTED = "#9a9da3"
ACCENT = "#4f8cff"
ACCENT_HOVER = "#6da0ff"
BORDER = "#37393e"
WAVEFORM_COLOR = "#5b8cff"
SELECTION_COLOR = "#3a4a6b"
PLAYHEAD_COLOR = "#ff5c5c"
HANDLE_COLOR = "#e6e6e8"


def _hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def format_time(seconds):
    """MM:SS.mmm"""
    seconds = max(0.0, seconds)
    m = int(seconds // 60)
    s = seconds - m * 60
    return f"{m:02d}:{s:06.3f}"


def pick_tick_interval(span_sec, target_ticks=8):
    """Chooses a 'nice' timeline tick spacing (in seconds) for a given
    visible time span, aiming for roughly target_ticks marks."""
    candidates = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5,
                  1, 2, 5, 10, 15, 30, 60, 120, 300, 600]
    raw = span_sec / target_ticks
    for c in candidates:
        if c >= raw:
            return c
    return candidates[-1]


def render_waveform_image(data, w, h, bg_hex, wave_hex, supersample=3):
    """Renders a waveform envelope as a supersampled-then-downsampled PIL
    Image for smooth (anti-aliased) edges, instead of many individual
    1px canvas lines. Returns a PIL Image in RGB mode, sized (w, h)."""
    sw, sh = max(1, w * supersample), max(1, h * supersample)
    mins, maxs = compute_waveform_peaks(data, sw)
    mid = sh / 2.0
    amp_scale = (sh / 2.0) * 0.9
    img = Image.new("RGB", (sw, sh), _hex_to_rgb(bg_hex))
    draw = ImageDraw.Draw(img)
    top = mid - maxs * amp_scale
    bot = mid - mins * amp_scale
    poly = list(zip(range(sw), top)) + list(zip(range(sw - 1, -1, -1), bot[::-1]))
    draw.polygon(poly, fill=_hex_to_rgb(wave_hex))
    return img.resize((w, h), Image.LANCZOS)


def render_rounded_box_image(w, h, radius, fill_hex, border_hex, supersample=4):
    """Anti-aliased rounded rectangle, for entry/panel backgrounds. Plain
    canvas create_polygon(smooth=True) isn't actually anti-aliased and
    looks jaggy at small sizes -- rendering via PIL and downsampling is."""
    w, h = max(1, int(w)), max(1, int(h))
    sw, sh = w * supersample, h * supersample
    img = Image.new("RGBA", (sw, sh), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([0, 0, sw - 1, sh - 1], radius=radius * supersample,
                            fill=_hex_to_rgb(fill_hex) + (255,),
                            outline=_hex_to_rgb(border_hex) + (255,), width=max(1, supersample))
    return img.resize((w, h), Image.LANCZOS)


def render_checkbox_image(w, h, radius, checked, field_bg_hex, accent_hex, border_hex,
                           check_hex="#ffffff", supersample=4):
    """Anti-aliased rounded checkbox indicator with a smooth checkmark
    stroke (the hand-drawn canvas version looked jaggy/misshapen)."""
    w, h = max(1, int(w)), max(1, int(h))
    sw, sh = w * supersample, h * supersample
    img = Image.new("RGBA", (sw, sh), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    fill = _hex_to_rgb(accent_hex) if checked else _hex_to_rgb(field_bg_hex)
    draw.rounded_rectangle([0, 0, sw - 1, sh - 1], radius=radius * supersample,
                            fill=fill + (255,), outline=_hex_to_rgb(border_hex) + (255,),
                            width=max(1, supersample))
    if checked:
        pts = [(w * 0.22 * supersample, h * 0.52 * supersample),
               (w * 0.42 * supersample, h * 0.72 * supersample),
               (w * 0.8 * supersample, h * 0.28 * supersample)]
        draw.line(pts, fill=_hex_to_rgb(check_hex) + (255,), width=max(2, supersample), joint="curve")
    return img.resize((w, h), Image.LANCZOS)


def _rounded_rect_points(w, h, r):
    r = min(r, w / 2, h / 2)
    return [r, 0, w - r, 0, w, 0, w, r, w, h - r, w, h,
            w - r, h, r, h, 0, h, 0, h - r, 0, r, 0, 0]


class RoundedEntry:
    """A ttk.Entry substitute with actual rounded, anti-aliased corners
    (drawn via PIL when available; falls back to a canvas polygon, which
    looks blockier since Tk's canvas doesn't anti-alias, if Pillow isn't
    installed). Resizes responsively via <Configure>."""

    def __init__(self, parent, textvariable, bg, field_bg, fg, border, height=32, radius=10):
        import tkinter as tk
        self.tk = tk
        self.radius = radius
        self.bg, self.field_bg, self.border, self.fg = bg, field_bg, border, fg
        self.frame = tk.Frame(parent, bg=bg)
        self.canvas = tk.Canvas(self.frame, height=height, bg=bg, highlightthickness=0)
        self.canvas.pack(fill="x", expand=True)
        self.entry = tk.Entry(self.canvas, textvariable=textvariable, bg=field_bg, fg=fg,
                               insertbackground=fg, relief="flat", highlightthickness=0,
                               bd=0, font=("Segoe UI", 10))
        self._bg_photo = None
        self.canvas.bind("<Configure>", self._redraw)

    def _redraw(self, event=None):
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w < 2 or h < 2:
            return
        self.canvas.delete("bg")
        if PIL_AVAILABLE:
            img = render_rounded_box_image(w, h, self.radius, self.field_bg, self.border)
            self._bg_photo = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor="nw", image=self._bg_photo, tags="bg")
        else:
            pts = _rounded_rect_points(w, h, self.radius)
            self.canvas.create_polygon(pts, smooth=True, fill=self.field_bg,
                                        outline=self.border, tags="bg")
        self.canvas.tag_lower("bg")
        self.canvas.delete("entrywin")
        self.canvas.create_window(self.radius, h / 2, window=self.entry, anchor="w",
                                   width=max(10, w - 2 * self.radius), tags="entrywin")

    def pack(self, **kw):
        self.frame.pack(**kw)


class RoundedCheckbutton:
    """A ttk.Checkbutton substitute with an anti-aliased rounded box
    indicator and a smooth checkmark stroke (drawn via PIL when
    available)."""

    def __init__(self, parent, text, variable, bg, fg, field_bg, accent, border,
                 command=None, box=18, radius=5):
        import tkinter as tk
        self.tk = tk
        self.variable, self.command = variable, command
        self.box, self.radius = box, radius
        self.bg, self.fg, self.field_bg, self.accent, self.border = bg, fg, field_bg, accent, border
        self._photo = None

        self.frame = tk.Frame(parent, bg=bg)
        self.canvas = tk.Canvas(self.frame, width=box, height=box, bg=bg, highlightthickness=0)
        self.canvas.pack(side="left")
        self.label = tk.Label(self.frame, text=text, bg=bg, fg=fg, font=("Segoe UI", 10))
        self.label.pack(side="left", padx=(6, 0))
        self.canvas.bind("<Button-1>", self._toggle)
        self.label.bind("<Button-1>", self._toggle)
        self._draw()

    def _draw(self):
        self.canvas.delete("all")
        w = h = self.box
        checked = self.variable.get()
        if PIL_AVAILABLE:
            img = render_checkbox_image(w, h, self.radius, checked, self.field_bg, self.accent, self.border)
            self._photo = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor="nw", image=self._photo)
        else:
            pts = _rounded_rect_points(w, h, self.radius)
            fill = self.accent if checked else self.field_bg
            self.canvas.create_polygon(pts, smooth=True, fill=fill, outline=self.border)
            if checked:
                self.canvas.create_line(w * 0.22, h * 0.52, w * 0.42, h * 0.72, w * 0.8, h * 0.28,
                                         fill="#ffffff", width=2, capstyle="round", joinstyle="round")

    def _toggle(self, event=None):
        self.variable.set(not self.variable.get())
        self._draw()
        if self.command:
            self.command()

    def pack(self, **kw):
        self.frame.pack(**kw)


class LoopCrossfadeGUI:
    HANDLE_HIT_PX = 6
    CLICK_SLOP_PX = 3

    def __init__(self):
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
        self.tk, self.filedialog, self.messagebox, self.ttk = tk, filedialog, messagebox, ttk

        self.root = TkinterDnD.Tk() if DND_AVAILABLE else tk.Tk()
        self.root.title("Loop Crossfade")
        self.root.geometry("760x620")
        self.root.minsize(700, 600)
        self.root.configure(bg=BG)

        self._build_style()

        # ---- state ----
        self.data = None           # currently loaded (possibly cropped) audio, float64 (n, ch)
        self.sr = 44100
        self.sampwidth = 2
        self.loaded_path = None
        self.cropped = False
        self.sel_start = 0
        self.sel_end = 0
        self.zoom_start = 0        # visible window into self.data, in samples
        self.zoom_end = 0
        self.canvas_width = 700
        self.canvas_height = 160
        self.drag_mode = None      # None | "start" | "end" | "new" | "pending"
        self.drag_anchor_x = None
        self.pre_drag_selection = None
        self.undo_stack = []
        self.redo_stack = []
        self.preview_mode = False  # True while the player holds a processed preview, not raw audio
        self._wave_photo = None    # keep a reference so PIL's PhotoImage isn't garbage collected
        self.player = AudioPlayer()
        self.shortcuts = load_shortcuts()

        self.in_path_var = tk.StringVar()
        self.out_path_var = tk.StringVar()
        self.xfade_var = tk.StringVar(value="0.30")
        self.curve_var = tk.StringVar(value="Equal power")
        self.auto_xfade_var = tk.BooleanVar(value=False)   # OFF by default, per spec
        self.snap_var = tk.BooleanVar(value=False)
        self.window_var = tk.StringVar(value="0.25")
        self.loop_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Drag an audio file onto this window, or click Browse.")
        self.play_label_var = tk.StringVar(value="Play")
        self.time_var = tk.StringVar(value="00:00.000")
        self.selection_duration_var = tk.StringVar(value="Selection: --")
        self._click_flag = None       # (x_pixel, time_str) or None
        self._click_flag_after_id = None

        self._build_widgets()
        self._bind_shortcuts()
        if DND_AVAILABLE:
            self._enable_drag_and_drop()

        self._poll_playhead()

    # ---------------- styling ----------------

    def _build_style(self):
        style = self.ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("TLabel", background=BG, foreground=FG, font=("Segoe UI", 10))
        style.configure("Muted.TLabel", background=BG, foreground=MUTED, font=("Segoe UI", 9))
        style.configure("Heading.TLabel", background=BG, foreground=FG, font=("Segoe UI", 13, "bold"))
        style.configure("TCheckbutton", background=BG, foreground=FG, font=("Segoe UI", 10))
        style.map("TCheckbutton", background=[("active", BG)], foreground=[("disabled", MUTED)])
        style.configure("TEntry", fieldbackground=FIELD_BG, foreground=FG, insertcolor=FG,
                         bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER)
        style.configure("TCombobox", fieldbackground=FIELD_BG, background=FIELD_BG,
                         foreground=FG, arrowcolor=FG, bordercolor=BORDER)
        style.map("TCombobox", fieldbackground=[("readonly", FIELD_BG)], foreground=[("readonly", FG)])
        style.configure("TButton", background=PANEL, foreground=FG, borderwidth=0,
                         focusthickness=0, padding=8, font=("Segoe UI", 10))
        style.map("TButton", background=[("active", BORDER)])
        style.configure("Accent.TButton", background=ACCENT, foreground="#ffffff",
                         borderwidth=0, padding=10, font=("Segoe UI", 11, "bold"))
        style.map("Accent.TButton", background=[("active", ACCENT_HOVER)])
        style.configure("Toggle.TButton", background=PANEL, foreground=FG, borderwidth=0, padding=8)
        style.configure("ToggleOn.TButton", background=ACCENT, foreground="#ffffff", borderwidth=0, padding=8)

    # ---------------- widget layout ----------------

    def _build_widgets(self):
        tk, ttk = self.tk, self.ttk
        outer = ttk.Frame(self.root, padding=16)
        outer.pack(fill="both", expand=True)

        ttk.Label(outer, text="Loop Crossfade", style="Heading.TLabel").pack(anchor="w")
        ttk.Label(outer, text="Drag & drop a file, select a region, audition the loop, then save.",
                  style="Muted.TLabel").pack(anchor="w", pady=(0, 10))

        # file row (rounded entries)
        row = ttk.Frame(outer); row.pack(fill="x", pady=3)
        ttk.Label(row, text="Input", width=7).pack(side="left")
        in_entry = RoundedEntry(row, self.in_path_var, BG, FIELD_BG, FG, BORDER)
        in_entry.pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(row, text="Browse", command=self.choose_input).pack(side="left")

        row = ttk.Frame(outer); row.pack(fill="x", pady=3)
        ttk.Label(row, text="Save as", width=7).pack(side="left")
        out_entry = RoundedEntry(row, self.out_path_var, BG, FIELD_BG, FG, BORDER)
        out_entry.pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(row, text="Browse", command=self.choose_output).pack(side="left")

        # timeline ruler (shared coordinate space with the waveform below)
        self.timeline_canvas = tk.Canvas(outer, height=22, bg=BG, highlightthickness=0)
        self.timeline_canvas.pack(fill="x", pady=(12, 0))
        self.timeline_canvas.bind("<Configure>", lambda e: self._redraw())

        # waveform canvas
        canvas_frame = ttk.Frame(outer, style="Panel.TFrame")
        canvas_frame.pack(fill="x", pady=(0, 4))
        self.canvas = tk.Canvas(canvas_frame, height=self.canvas_height, bg=PANEL,
                                 highlightthickness=0)
        self.canvas.pack(fill="x", expand=True, padx=4, pady=4)
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)     # Windows / macOS
        self.canvas.bind("<Button-4>", self._on_mousewheel)       # Linux scroll up
        self.canvas.bind("<Button-5>", self._on_mousewheel)       # Linux scroll down
        self._draw_placeholder()

        # timer + selection duration readouts
        info_row = ttk.Frame(outer); info_row.pack(fill="x", pady=(2, 4))
        ttk.Label(info_row, textvariable=self.time_var, style="Muted.TLabel",
                  font=("Segoe UI", 10, "bold")).pack(side="left")
        ttk.Label(info_row, textvariable=self.selection_duration_var, style="Muted.TLabel").pack(side="right")

        ttk.Label(outer, text="Drag to select, drag edges to adjust, click to move the playhead. Scroll to zoom.",
                  style="Muted.TLabel").pack(anchor="w", pady=(0, 4))

        # zoom row
        zoom_row = ttk.Frame(outer); zoom_row.pack(fill="x", pady=(0, 8))
        ttk.Button(zoom_row, text="Zoom to Selection", command=self.zoom_to_selection).pack(side="left", padx=(0, 4))
        ttk.Button(zoom_row, text="Zoom to Fit", command=self.zoom_to_fit).pack(side="left", padx=4)
        ttk.Button(zoom_row, text="\u21b6 Undo", command=self.undo).pack(side="right", padx=4)
        ttk.Button(zoom_row, text="\u21b7 Redo", command=self.redo).pack(side="right", padx=(4, 0))

        # transport
        transport = ttk.Frame(outer); transport.pack(fill="x", pady=(4, 4))
        self.btn_rewind = ttk.Button(transport, text="\u23ee Rewind", command=self.on_rewind)
        self.btn_rewind.pack(side="left", padx=(0, 4))
        self.btn_play = ttk.Button(transport, textvariable=self.play_label_var, command=self.on_play_pause)
        self.btn_play.pack(side="left", padx=4)
        self.btn_stop = ttk.Button(transport, text="\u23f9 Stop", command=self.on_stop)
        self.btn_stop.pack(side="left", padx=4)
        self.btn_loop = ttk.Button(transport, text="\U0001f501 Loop", command=self.on_loop_toggle)
        self.btn_loop.pack(side="left", padx=4)
        ttk.Button(transport, text="\u25b6 Audition Loop", style="Accent.TButton",
                   command=self.on_audition).pack(side="left", padx=(16, 4))
        ttk.Button(transport, text="Crop to Selection", command=self.on_crop).pack(side="left", padx=4)
        ttk.Button(transport, text="Keyboard Shortcuts...", command=self.open_shortcuts_dialog).pack(side="right")

        ttk.Label(outer, text="Audition previews the current selection's crossfade without saving or cropping. "
                               "Crop is only needed once you want to commit the working range.",
                  style="Muted.TLabel").pack(anchor="w", pady=(2, 0))

        ttk.Frame(outer, height=1, style="Panel.TFrame").pack(fill="x", pady=14)

        # processing options (rounded checkboxes)
        row = ttk.Frame(outer); row.pack(fill="x", pady=4)
        RoundedCheckbutton(row, "Snap loop points to transients (beat / articulation alignment)",
                            self.snap_var, BG, FG, FIELD_BG, ACCENT, BORDER,
                            command=self._toggle_window_entry).pack(side="left")
        row = ttk.Frame(outer); row.pack(fill="x", pady=(0, 10))
        ttk.Label(row, text="Search window (s):", style="Muted.TLabel").pack(side="left", padx=(24, 6))
        self.window_entry = ttk.Entry(row, textvariable=self.window_var, width=8, state="disabled")
        self.window_entry.pack(side="left")

        row = ttk.Frame(outer); row.pack(fill="x", pady=4)
        RoundedCheckbutton(row, "Auto-detect crossfade length", self.auto_xfade_var,
                            BG, FG, FIELD_BG, ACCENT, BORDER,
                            command=self._toggle_xfade_entry).pack(side="left")
        row = ttk.Frame(outer); row.pack(fill="x", pady=(0, 4))
        ttk.Label(row, text="Manual crossfade (s):", style="Muted.TLabel").pack(side="left", padx=(24, 6))
        self.xfade_entry = ttk.Entry(row, textvariable=self.xfade_var, width=8, state="normal")
        self.xfade_entry.pack(side="left")

        row = ttk.Frame(outer); row.pack(fill="x", pady=(8, 4))
        ttk.Label(row, text="Curve:", style="Muted.TLabel").pack(side="left", padx=(24, 6))
        ttk.Combobox(row, textvariable=self.curve_var, values=["Equal power", "Linear"],
                     state="readonly", width=14).pack(side="left")

        ttk.Frame(outer, height=1, style="Panel.TFrame").pack(fill="x", pady=14)

        ttk.Button(outer, text="Process & Save", style="Accent.TButton",
                   command=self.run_process).pack(fill="x", pady=(0, 10))

        ttk.Label(outer, textvariable=self.status_var, style="Muted.TLabel",
                  wraplength=700, justify="left").pack(anchor="w", fill="x")

        notes = []
        if not ffmpeg_available():
            notes.append("ffmpeg not found -- only plain WAV will work until it's installed.")
        if not SOUNDDEVICE_AVAILABLE:
            notes.append("sounddevice not installed -- playback controls are disabled (pip install sounddevice).")
        if not DND_AVAILABLE:
            notes.append("tkinterdnd2 not installed -- drag & drop is disabled, use Browse instead (pip install tkinterdnd2).")
        if not PIL_AVAILABLE:
            notes.append("Pillow not installed -- waveform draws with plain lines instead of smoothed fill (pip install Pillow).")
        if notes:
            ttk.Label(outer, text=" / ".join(notes), style="Muted.TLabel",
                      foreground="#e2a33d", wraplength=700, justify="left").pack(anchor="w", pady=(8, 0))

    def _toggle_xfade_entry(self):
        self.xfade_entry.configure(state="disabled" if self.auto_xfade_var.get() else "normal")

    def _toggle_window_entry(self):
        self.window_entry.configure(state="normal" if self.snap_var.get() else "disabled")

    # ---------------- file loading ----------------

    def choose_input(self):
        path = self.filedialog.askopenfilename(
            title="Choose audio file",
            filetypes=[("Audio files", "*.wav *.aif *.aiff *.mp3 *.mp4 *.m4a *.flac"), ("All files", "*.*")],
        )
        if path:
            self.load_file(path)

    def choose_output(self):
        path = self.filedialog.asksaveasfilename(
            title="Save processed file as", defaultextension=".wav",
            filetypes=[("WAV", "*.wav"), ("AIFF", "*.aiff"), ("MP3", "*.mp3"),
                       ("MP4/M4A", "*.m4a"), ("FLAC", "*.flac")],
        )
        if path:
            self.out_path_var.set(path)

    def _enable_drag_and_drop(self):
        for widget in (self.root, self.canvas):
            widget.drop_target_register(DND_FILES)
            widget.dnd_bind("<<Drop>>", self._on_drop)

    def _on_drop(self, event):
        paths = _parse_dnd_paths(event.data)
        if paths:
            self.load_file(paths[0])

    def load_file(self, path):
        ext = os.path.splitext(path)[1].lower()
        if ext not in SUPPORTED_EXTS:
            self.messagebox.showerror("Loop Crossfade", f"Unsupported file type: {ext}")
            return
        try:
            self.status_var.set("Loading...")
            self.root.update_idletasks()
            data, sr, sampwidth = decode_to_pcm(path)
        except Exception as e:
            self.status_var.set("Failed to load file.")
            self.messagebox.showerror("Loop Crossfade", str(e))
            return

        self.data, self.sr, self.sampwidth = data, sr, sampwidth
        self.loaded_path = path
        self.cropped = False
        self.sel_start, self.sel_end = 0, len(data)
        self.zoom_start, self.zoom_end = 0, len(data)
        self.preview_mode = False
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.player.load(data, sr)

        self.in_path_var.set(path)
        # auto-fill Save As to the same directory as the loaded file
        root_name, ext = os.path.splitext(path)
        self.out_path_var.set(root_name + "_loop" + ext)

        self._click_flag = None
        self._redraw()
        self._update_selection_duration_label()
        dur = len(data) / sr
        self.status_var.set(f"Loaded {os.path.basename(path)} ({dur:.2f}s). Select a region, Audition to preview the loop, then Process & Save.")

    # ---------------- undo / redo ----------------

    def _snapshot(self):
        return {"data": self.data, "sel_start": self.sel_start, "sel_end": self.sel_end,
                "cropped": self.cropped, "zoom_start": self.zoom_start, "zoom_end": self.zoom_end}

    def _restore(self, snap):
        self.data = snap["data"]
        self.sel_start, self.sel_end = snap["sel_start"], snap["sel_end"]
        self.cropped = snap["cropped"]
        self.zoom_start, self.zoom_end = snap["zoom_start"], snap["zoom_end"]
        self.preview_mode = False
        self.player.load(self.data, self.sr)
        self.player.set_selection(self.sel_start, self.sel_end)
        self._redraw()
        self._update_selection_duration_label()

    def push_undo(self):
        self.undo_stack.append(self._snapshot())
        self.redo_stack.clear()

    def undo(self):
        if not self.undo_stack or self.data is None:
            return
        self.redo_stack.append(self._snapshot())
        self._restore(self.undo_stack.pop())
        self.status_var.set("Undid last change.")

    def redo(self):
        if not self.redo_stack or self.data is None:
            return
        self.undo_stack.append(self._snapshot())
        self._restore(self.redo_stack.pop())
        self.status_var.set("Redid change.")

    # ---------------- waveform canvas ----------------

    def _draw_placeholder(self):
        self.canvas.delete("all")
        w = self.canvas.winfo_width() or self.canvas_width
        h = self.canvas.winfo_height() or self.canvas_height
        self.canvas.create_text(w // 2, h // 2, text="No audio loaded",
                                 fill=MUTED, font=("Segoe UI", 10))
        self.timeline_canvas.delete("all")

    def _on_canvas_resize(self, event):
        self.canvas_width, self.canvas_height = event.width, event.height
        self._redraw()

    def _visible_range(self):
        if self.data is None:
            return 0, 0
        return self.zoom_start, self.zoom_end

    def _display_cursor_sample(self):
        """Maps the player's cursor into self.data-space for on-screen
        display. In preview mode the player holds a shorter, processed
        buffer derived from [sel_start, sel_end) rather than self.data
        itself -- without this mapping the playhead would be drawn using
        the wrong buffer's coordinate space and appear stuck near the
        start of the file regardless of actual (correct) playback
        position -- that was the reported "jumps to the beginning" bug."""
        cursor = self.player.get_cursor()
        if not self.preview_mode or self.player.data is None:
            return cursor
        total = max(1, self.player.data.shape[0])
        frac = min(1.0, cursor / total)
        return int(self.sel_start + frac * (self.sel_end - self.sel_start))

    def _redraw(self):
        """Single entry point that keeps the timeline ruler and the
        waveform in sync -- call this instead of the two draw methods
        directly."""
        self._redraw_timeline()
        self._redraw_waveform()

    def _redraw_timeline(self):
        self.timeline_canvas.delete("all")
        if self.data is None:
            return
        w = max(1, self.timeline_canvas.winfo_width() or self.canvas_width)
        h = max(1, self.timeline_canvas.winfo_height() or 22)
        vs, ve = self._visible_range()
        span_sec = max(1e-6, (ve - vs) / self.sr)
        interval = pick_tick_interval(span_sec)

        first_tick = (int(vs / self.sr / interval)) * interval
        t = first_tick
        while t <= ve / self.sr + interval:
            samp = t * self.sr
            x = self._sample_to_x(samp, w)
            if -20 <= x <= w + 20:
                self.timeline_canvas.create_line(x, h - 6, x, h, fill=MUTED)
                label = format_time(t) if interval < 1 else f"{int(t // 60):02d}:{int(t % 60):02d}"
                self.timeline_canvas.create_text(x, h - 8, text=label, fill=MUTED,
                                                  font=("Segoe UI", 8), anchor="s")
            t += interval

        if self._click_flag is not None:
            fx, ftext = self._click_flag
            self._draw_flag(fx, ftext, w)

    def _draw_flag(self, x, text, canvas_w):
        pad_x, pad_y = 6, 3
        text_w = 7 * len(text)  # rough monospace-ish estimate, good enough for a small flag
        box_w = text_w + pad_x * 2
        box_x = max(2, min(canvas_w - box_w - 2, x - box_w / 2))
        self.timeline_canvas.create_rectangle(box_x, 0, box_x + box_w, 14,
                                               fill=ACCENT, outline="")
        self.timeline_canvas.create_polygon(x - 4, 14, x + 4, 14, x, 18, fill=ACCENT, outline="")
        self.timeline_canvas.create_text(box_x + box_w / 2, 7, text=text, fill="#ffffff",
                                          font=("Segoe UI", 8, "bold"))

    def _redraw_waveform(self):
        if self.data is None:
            self._draw_placeholder()
            return
        self.canvas.delete("all")
        w = max(1, self.canvas.winfo_width() or self.canvas_width)
        h = max(1, self.canvas.winfo_height() or self.canvas_height)
        vs, ve = self._visible_range()
        segment = self.data[vs:ve] if ve > vs else self.data[0:1]

        if PIL_AVAILABLE:
            img = render_waveform_image(segment, w, h, PANEL, WAVEFORM_COLOR)
            self._wave_photo = ImageTk.PhotoImage(img)  # keep a reference, or Tk garbage-collects it
            self.canvas.create_image(0, 0, anchor="nw", image=self._wave_photo)
        else:
            mins, maxs = compute_waveform_peaks(segment, w)
            mid = h / 2
            scale = (h / 2) * 0.9
            for x in range(len(maxs)):
                y1 = mid - maxs[x] * scale
                y2 = mid - mins[x] * scale
                self.canvas.create_line(x, y1, x, y2, fill=WAVEFORM_COLOR)

        sx = self._sample_to_x(self.sel_start, w)
        ex = self._sample_to_x(self.sel_end, w)
        self.canvas.create_rectangle(sx, 0, ex, h, fill=SELECTION_COLOR, outline="", stipple="gray50")
        self.canvas.create_line(sx, 0, sx, h, fill=HANDLE_COLOR, width=2, tags="handle_start")
        self.canvas.create_line(ex, 0, ex, h, fill=HANDLE_COLOR, width=2, tags="handle_end")

        cursor = self._display_cursor_sample()
        cx = self._sample_to_x(cursor, w)
        self.canvas.create_line(cx, 0, cx, h, fill=PLAYHEAD_COLOR, width=1, tags="playhead")

    def _sample_to_x(self, sample, width=None):
        width = width or self.canvas.winfo_width() or self.canvas_width
        vs, ve = self._visible_range()
        span = max(1, ve - vs)
        x = ((sample - vs) / span) * width
        return max(-10, min(width + 10, x))  # allow slightly off-canvas so edge handles are still visible

    def _x_to_sample(self, x, width=None):
        width = width or self.canvas.winfo_width() or self.canvas_width
        vs, ve = self._visible_range()
        span = ve - vs
        if width == 0 or span <= 0:
            return vs
        frac = max(0.0, min(1.0, x / width))
        return int(vs + frac * span)

    def _update_selection_duration_label(self):
        if self.data is None or self.sel_end <= self.sel_start:
            self.selection_duration_var.set("Selection: --")
        else:
            dur = (self.sel_end - self.sel_start) / self.sr
            self.selection_duration_var.set(f"Selection: {format_time(dur)}")

    def _show_click_flag(self, x_pixel, sample):
        t = sample / self.sr
        self._click_flag = (x_pixel, format_time(t))
        if self._click_flag_after_id is not None:
            self.root.after_cancel(self._click_flag_after_id)
        self._click_flag_after_id = self.root.after(1500, self._clear_click_flag)

    def _clear_click_flag(self):
        self._click_flag = None
        self._click_flag_after_id = None
        self._redraw_timeline()

    def _on_canvas_press(self, event):
        if self.data is None:
            return
        w = self.canvas.winfo_width()
        sx = self._sample_to_x(self.sel_start, w)
        ex = self._sample_to_x(self.sel_end, w)
        self.pre_drag_selection = (self.sel_start, self.sel_end)
        if abs(event.x - sx) <= self.HANDLE_HIT_PX:
            self.drag_mode = "start"
        elif abs(event.x - ex) <= self.HANDLE_HIT_PX:
            self.drag_mode = "end"
        else:
            self.drag_mode = "pending"  # resolved to "new" selection or a playhead click on release
            self.drag_anchor_x = event.x

    def _on_canvas_drag(self, event):
        if self.data is None or self.drag_mode is None:
            return
        w = self.canvas.winfo_width()
        x = max(0, min(w, event.x))
        samp = self._x_to_sample(x, w)

        if self.drag_mode == "pending" and abs(event.x - self.drag_anchor_x) > self.CLICK_SLOP_PX:
            self.drag_mode = "new"
            self.sel_start = self._x_to_sample(self.drag_anchor_x, w)
            self.sel_end = self.sel_start

        if self.drag_mode == "start":
            self.sel_start = min(samp, self.sel_end)
        elif self.drag_mode == "end":
            self.sel_end = max(samp, self.sel_start)
        elif self.drag_mode == "new":
            anchor_samp = self._x_to_sample(self.drag_anchor_x, w)
            self.sel_start, self.sel_end = min(anchor_samp, samp), max(anchor_samp, samp)

        if self.drag_mode in ("start", "end", "new"):
            self._update_selection_duration_label()
            self._redraw()

    def _on_canvas_release(self, event):
        if self.data is None:
            self.drag_mode = None
            return
        if self.drag_mode == "pending":
            # a plain click (no meaningful drag): move the playhead there
            # and show a time flag on the timeline
            w = self.canvas.winfo_width()
            samp = self._x_to_sample(event.x, w)
            self.player.rewind()  # ensure stopped state doesn't fight the seek
            with self.player.lock:
                self.player.cursor = max(self.sel_start, min(samp, self.sel_end))
            self._show_click_flag(event.x, samp)
            self._redraw()
        elif self.drag_mode in ("start", "end", "new") and self.pre_drag_selection is not None:
            if (self.sel_start, self.sel_end) != self.pre_drag_selection:
                # push the PRE-drag selection so undo restores exactly where the drag began
                old_start, old_end = self.pre_drag_selection
                self.undo_stack.append({
                    "data": self.data, "sel_start": old_start, "sel_end": old_end,
                    "cropped": self.cropped, "zoom_start": self.zoom_start, "zoom_end": self.zoom_end,
                })
                self.redo_stack.clear()
                if self.preview_mode:
                    self._exit_preview_mode()
            self._update_selection_duration_label()
        self.drag_mode = None
        self.pre_drag_selection = None
        if self.sel_end > self.sel_start:
            self.player.set_selection(self.sel_start, self.sel_end)

    # ---------------- zoom ----------------

    MIN_ZOOM_SAMPLES = 256

    def _zoom_step(self, direction, center_x=None):
        if self.data is None:
            return
        w = self.canvas.winfo_width() or self.canvas_width
        center_x = w / 2 if center_x is None else center_x
        self._zoom_at(center_x, w, direction)
        self._redraw()

    def _zoom_at(self, x_pixel, w, direction):
        n = len(self.data)
        vs, ve = self.zoom_start, self.zoom_end
        span = ve - vs
        factor = 0.8 if direction > 0 else 1.25
        new_span = max(self.MIN_ZOOM_SAMPLES, min(n, int(span * factor)))
        if new_span >= n:
            self.zoom_start, self.zoom_end = 0, n
            return
        center_samp = vs + (x_pixel / w) * span
        left_frac = x_pixel / w
        new_vs = int(center_samp - left_frac * new_span)
        new_ve = new_vs + new_span
        if new_vs < 0:
            new_ve -= new_vs; new_vs = 0
        if new_ve > n:
            shift = new_ve - n
            new_vs = max(0, new_vs - shift); new_ve = n
        self.zoom_start, self.zoom_end = new_vs, new_ve

    def _on_mousewheel(self, event):
        if self.data is None:
            return
        direction = 1 if (getattr(event, "delta", 0) > 0 or getattr(event, "num", None) == 4) else -1
        w = self.canvas.winfo_width() or self.canvas_width
        self._zoom_at(event.x, w, direction)
        self._redraw()

    def zoom_to_fit(self):
        if self.data is None:
            return
        self.zoom_start, self.zoom_end = 0, len(self.data)
        self._redraw()

    def zoom_to_selection(self):
        if self.data is None or self.sel_end <= self.sel_start:
            return
        span = self.sel_end - self.sel_start
        pad = max(1, int(span * 0.1))
        self.zoom_start = max(0, self.sel_start - pad)
        self.zoom_end = min(len(self.data), self.sel_end + pad)
        self._redraw()

    # ---------------- transport ----------------

    def _exit_preview_mode(self):
        """Swap the player back to raw (un-processed) audio -- used whenever
        something invalidates a processed preview that's currently loaded."""
        if not self.preview_mode:
            return
        self.player.stop()
        self.player.load(self.data, self.sr)
        self.player.set_selection(self.sel_start, self.sel_end)
        self.preview_mode = False

    def on_play_pause(self):
        if self.data is None:
            return
        if not SOUNDDEVICE_AVAILABLE:
            self.messagebox.showinfo("Loop Crossfade", "Install the 'sounddevice' package to enable playback:\npip install sounddevice")
            return
        self._exit_preview_mode()  # plain Play always plays raw source audio
        if self.player.playing:
            self.player.pause()
            self.play_label_var.set("Play")
        else:
            self.player.set_selection(self.sel_start, self.sel_end)
            self.player.set_loop(self.loop_var.get())
            self.player.play()
            self.play_label_var.set("Pause")

    def on_stop(self):
        self.player.stop()
        self.play_label_var.set("Play")
        self._redraw()

    def on_rewind(self):
        self.player.rewind()
        self._redraw()

    def on_loop_toggle(self):
        self.loop_var.set(not self.loop_var.get())
        self.player.set_loop(self.loop_var.get())
        self.btn_loop.configure(style="ToggleOn.TButton" if self.loop_var.get() else "Toggle.TButton")

    def _read_process_params(self):
        """Validates and returns (xfade_seconds_or_None, curve, snap, window)
        from the current UI fields, or None if invalid (and shows an error)."""
        xfade_seconds = None
        if not self.auto_xfade_var.get():
            try:
                xfade_seconds = float(self.xfade_var.get())
                if xfade_seconds <= 0:
                    raise ValueError
            except ValueError:
                self.messagebox.showerror("Loop Crossfade", "Crossfade duration must be a positive number of seconds.")
                return None
        try:
            transient_window = float(self.window_var.get()) if self.snap_var.get() else 0.25
        except ValueError:
            self.messagebox.showerror("Loop Crossfade", "Transient search window must be a number of seconds.")
            return None
        curve = "equal_power" if self.curve_var.get() == "Equal power" else "linear"
        return xfade_seconds, curve, self.snap_var.get(), transient_window

    def on_audition(self):
        """Processes the CURRENT selection (crop not required) and plays it
        looped, without touching self.data or writing any file -- so you can
        hear whether the crossfade settings are right before committing."""
        if self.data is None:
            return
        if not SOUNDDEVICE_AVAILABLE:
            self.messagebox.showinfo("Loop Crossfade", "Install the 'sounddevice' package to enable playback:\npip install sounddevice")
            return
        if self.sel_end <= self.sel_start:
            self.messagebox.showerror("Loop Crossfade", "Select a region on the waveform first.")
            return
        params = self._read_process_params()
        if params is None:
            return
        xfade_seconds, curve, snap, window = params

        try:
            self.status_var.set("Auditioning...")
            self.root.update_idletasks()
            segment = self.data[self.sel_start:self.sel_end]
            t0 = time.time()
            preview, used_xfade, st, et = _run_pipeline(
                segment, self.sr, xfade_seconds, curve, snap, window, self.auto_xfade_var.get())
            elapsed = time.time() - t0

            self.player.stop()
            self.player.load(preview, self.sr)
            self.preview_mode = True
            self.loop_var.set(True)
            self.player.set_loop(True)
            self.btn_loop.configure(style="ToggleOn.TButton")
            self.player.play()
            self.play_label_var.set("Pause")

            dur = preview.shape[0] / self.sr
            self.status_var.set(
                f"Auditioning {dur:.2f}s loop (crossfade {used_xfade*1000:.0f} ms, computed in {elapsed*1000:.0f} ms). "
                f"Adjust settings and press Audition again, or Process & Save when it sounds right."
            )
        except Exception as e:
            self.status_var.set("Audition failed.")
            self.messagebox.showerror("Loop Crossfade", str(e))

    def on_crop(self):
        if self.data is None:
            return
        s, e = self.sel_start, self.sel_end
        if e <= s:
            self.messagebox.showerror("Loop Crossfade", "Select a region on the waveform first.")
            return
        self.push_undo()
        self.player.stop()
        self.data = self.data[s:e]
        self.sel_start, self.sel_end = 0, len(self.data)
        self.zoom_start, self.zoom_end = 0, len(self.data)
        self.cropped = True
        self.preview_mode = False
        self.player.load(self.data, self.sr)
        self._redraw()
        self._update_selection_duration_label()
        dur = len(self.data) / self.sr
        self.status_var.set(f"Cropped to {dur:.2f}s. (Cmd/Ctrl+Z to undo.)")

    def _poll_playhead(self):
        if self.data is not None and self.player.playing:
            self._redraw()
            display_samp = self._display_cursor_sample()
            self.time_var.set(format_time(display_samp / self.sr))
            if not self.player.playing:
                self.play_label_var.set("Play")
        self.root.after(50, self._poll_playhead)

    # ---------------- keyboard shortcuts ----------------

    def _action_map(self):
        return {
            "play_pause": self.on_play_pause,
            "stop": self.on_stop,
            "rewind": self.on_rewind,
            "loop_toggle": self.on_loop_toggle,
            "crop": self.on_crop,
            "audition": self.on_audition,
            "undo": self.undo,
            "redo": self.redo,
            "zoom_in": lambda: self._zoom_step(1),
            "zoom_out": lambda: self._zoom_step(-1),
            "zoom_fit": self.zoom_to_fit,
        }

    def _bind_shortcuts(self):
        for name, fn in self._action_map().items():
            self._bind_one(self.shortcuts.get(name, DEFAULT_SHORTCUTS[name]), fn)

    def _bind_one(self, key, fn):
        seq = f"<{key}>" if len(key) > 1 else f"<KeyPress-{key}>"
        try:
            self.root.bind(seq, lambda e: fn())
        except self.tk.TclError:
            pass  # an invalid/unsupported key sequence shouldn't crash the app

    def _rebind_all(self):
        for name, fn in self._action_map().items():
            key = self.shortcuts.get(name, DEFAULT_SHORTCUTS[name])
            self._bind_one(key, fn)

    @staticmethod
    def _event_to_key_string(event):
        """Builds a Tkinter-bindable key string (e.g. 'Control-z') from a
        KeyPress event, including modifiers -- needed so remapping Undo/
        Redo to Ctrl+<key> combos actually works, not just the bare key."""
        parts = []
        state = event.state
        if state & 0x0004:
            parts.append("Control")
        if state & 0x0001:
            parts.append("Shift")
        if state & 0x0008 or state & 0x20000:
            parts.append("Alt")
        keysym = event.keysym
        if keysym in ("Control_L", "Control_R", "Shift_L", "Shift_R", "Alt_L", "Alt_R"):
            return None  # a bare modifier key press isn't a usable shortcut on its own
        parts.append(keysym)
        return "-".join(parts)

    def open_shortcuts_dialog(self):
        tk, ttk = self.tk, self.ttk
        dlg = tk.Toplevel(self.root)
        dlg.title("Keyboard Shortcuts")
        dlg.configure(bg=BG)
        dlg.geometry("380x420")
        dlg.transient(self.root)

        rows = {}
        for i, (name, label) in enumerate(SHORTCUT_LABELS.items()):
            ttk.Label(dlg, text=label, background=BG, foreground=FG).grid(row=i, column=0, sticky="w", padx=10, pady=6)
            btn = ttk.Button(dlg, text=self.shortcuts.get(name, DEFAULT_SHORTCUTS[name]), width=14)
            btn.grid(row=i, column=1, padx=10, pady=6)
            rows[name] = btn

        def start_listening(name, btn):
            btn.configure(text="Press a key...")

            def capture(event):
                key = self._event_to_key_string(event)
                if key is None:
                    return  # ignore bare modifier presses, keep listening
                self.shortcuts[name] = key
                btn.configure(text=key)
                dlg.unbind("<KeyPress>")
                self._rebind_all()

            dlg.bind("<KeyPress>", capture)

        for name, btn in rows.items():
            btn.configure(command=lambda n=name, b=rows[n]: start_listening(n, b))

        def on_close():
            save_shortcuts(self.shortcuts)
            dlg.destroy()

        ttk.Button(dlg, text="Done", command=on_close, style="Accent.TButton").grid(
            row=len(rows), column=0, columnspan=2, pady=16, padx=10, sticky="ew")
        dlg.protocol("WM_DELETE_WINDOW", on_close)

    # ---------------- process & save ----------------

    def run_process(self):
        if self.data is None:
            self.messagebox.showerror("Loop Crossfade", "Load an audio file first.")
            return
        out_path = self.out_path_var.get()
        if not out_path:
            self.messagebox.showerror("Loop Crossfade", "Choose a Save As location first.")
            return
        if self.sel_end <= self.sel_start:
            self.messagebox.showerror("Loop Crossfade", "Select a region on the waveform first.")
            return

        params = self._read_process_params()
        if params is None:
            return
        xfade_seconds, curve, snap, window = params

        try:
            self.status_var.set("Processing...")
            self.root.update_idletasks()
            t0 = time.time()
            # operates on the current SELECTION, not necessarily the full
            # buffer -- Crop is optional; this is what makes "process and
            # save straight from a selection" work without cropping first
            segment = self.data[self.sel_start:self.sel_end]
            result, used_xfade, start_trim, end_trim = _run_pipeline(
                segment, self.sr, xfade_seconds, curve, snap, window, self.auto_xfade_var.get(),
            )
            encode_from_pcm(result, self.sr, self.sampwidth, out_path)
            elapsed = time.time() - t0
            duration = result.shape[0] / self.sr
            msg = (f"Done in {elapsed:.2f}s -- {duration:.2f}s loop saved to:\n{out_path}\n"
                   f"Crossfade used: {used_xfade * 1000:.0f} ms")
            if snap:
                msg += (f"\nTrimmed to transients: {start_trim / self.sr * 1000:.0f} ms from start, "
                        f"{end_trim / self.sr * 1000:.0f} ms from end")
            self.status_var.set(msg)
        except Exception as e:
            self.status_var.set("Failed. See error dialog.")
            self.messagebox.showerror("Loop Crossfade", str(e))

    def run(self):
        self.root.mainloop()


def launch_gui():
    app = LoopCrossfadeGUI()
    app.run()


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
