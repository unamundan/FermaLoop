#!/usr/bin/env python3
# Licensed under the MIT License. See LICENSE file in the project root.
"""
FermaLoop
=========
A tool for building seamless, glitch-free looping audio extensions --
material that can function as an infinite or extended fermata: a held
gesture with no fixed length, free to be released whenever a live
performance actually calls for it, rather than at a fixed, predetermined
timecode.

The idea it's built around: live theatrical sound design is rarely as
beat- or time-locked as a purely pre-recorded, playback-driven cue. A
held chord, a drone, an ambient bed under a scene -- these often need to
sustain for as long as the moment takes, not for a fixed duration decided
in advance. FermaLoop takes a short audio selection and blends its tail
into its head (crossfading away the seam) so it can loop indefinitely
with no click or pop at the join, and/or runs it through PaulXStretch for
extreme time-stretching, so a brief sound can become a long, evolving,
decaying, or sustained bed. Combined, these two features are a way to
build audio material that behaves less like a fixed backtrack and more
like something a live operator can hold, extend, or release on cue.

The resulting loops and stretches are meant to be exported and then
programmed into cue-based theatrical audio playback software --
where they can supplement the more rigid timing of pre-recorded
backtracks with something more organically responsive to what's actually
happening on stage: a way to introduce controllable, "conductable" timing
into material that would otherwise be locked to the clock.

Plus:

  * Multi-format I/O: WAV, AIFF, MP3, MP4/M4A, FLAC (decode/encode via ffmpeg)
  * A waveform view: drag to select a region, drag the edges to adjust it,
    crop to it, and preview before/after committing
  * Transport controls (Play/Pause, Stop, Rewind, Loop) with remappable
    keyboard shortcuts (see the "Hints & Keyboard Shortcuts" button in-app)
  * Drag-and-drop file loading onto the window (falls back to Browse if
    tkinterdnd2 isn't installed -- see dependencies below)
  * Optional transient-snap: finds the strongest attack near the start and
    end of the clip and trims to it, so the loop begins/ends on the beat
    or articulation instead of an arbitrary sample boundary
  * Crossfade length: manual by default, or an auto-detect option you can
    switch on (finds the length where the head/tail actually sound alike)
  * PaulXStretch: extreme time-stretching for turning a short selection
    into a long, evolving/decaying bed
  * A dark, flat, modern GUI

Runs from source on Windows, macOS, and Linux; packaged builds are
currently produced for Windows and macOS only (see build.yml).

GitHub: https://github.com/unamundan/FermaLoop

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

# Single source of truth for the app's version/author metadata -- read
# directly by build.yml (via a small extraction step) to populate the
# Windows .exe version resource and the macOS .app Info.plist, so the
# version only ever needs to be bumped in exactly one place, here.
APP_VERSION = "1.0.0"
APP_AUTHOR = "unamundan"
APP_COPYRIGHT = "\u00a9 unamundan 2026"
APP_DESCRIPTION = "Seamless audio loop crossfading and extreme time-stretching utility"
APP_URL = "https://github.com/unamundan/FermaLoop"

import os
import re
import io
import sys
import json
import time
import math
import wave
import struct
import base64
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

# The three output formats the GUI's Format selector offers (input loading
# still accepts the full SUPPORTED_EXTS set above).
FORMAT_OPTIONS = ["FLAC (Lossless)", "MP4 (Apple Lossless)", "MP3 (VBR)"]
FORMAT_EXT = {"FLAC (Lossless)": ".flac", "MP4 (Apple Lossless)": ".mp4", "MP3 (VBR)": ".mp3"}
MP3_QUALITY_INFO = {
    0: "best, ~220-260 kbps", 1: "~190-250 kbps", 2: "high quality, ~170-210 kbps",
    3: "~150-195 kbps", 4: "~140-185 kbps", 5: "~120-150 kbps",
    6: "~100-130 kbps", 7: "~80-120 kbps", 8: "~70-105 kbps", 9: "smallest, ~45-85 kbps",
}

FFMPEG_ENCODE_ARGS = {
    ".mp3":  ["-c:a", "libmp3lame", "-q:a", "2"],   # overridden dynamically by encode_from_pcm's mp3_quality param
    ".flac": ["-c:a", "flac", "-compression_level", "8"],  # lossless; 8 = highest compression (smaller file, same quality)
    ".mp4":  ["-c:a", "alac"],   # Apple Lossless -- verified bit-exact round-trip
    ".m4a":  ["-c:a", "alac"],   # same container family as .mp4, same lossless codec
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

_SUBPROCESS_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)  # the actual
# flag on Windows, a harmless no-op (0) everywhere else -- without this,
# every ffmpeg call below briefly flashes a visible console window on
# Windows specifically (a well-documented Windows+subprocess behavior:
# a console executable launched from a GUI app gets its own console
# window by default regardless of stdout/stderr being piped). Reported
# directly as a "phantom zooming window outline" appearing top-left and
# disappearing once loading finished -- exactly the shape of a console
# window's brief open/close animation. Shared between both ffmpeg call
# sites below so neither can be fixed without the other.


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
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 creationflags=_SUBPROCESS_NO_WINDOW)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed to decode '{path}':\n{result.stderr.decode(errors='ignore')}")
        return read_wav(tmp_wav)


def encode_from_pcm(data, sr, sampwidth, out_path, mp3_quality=2):
    """Write processed PCM data out in whatever format out_path's
    extension indicates. Plain WAV skips ffmpeg entirely.

    mp3_quality: LAME VBR quality, 0 (best/largest) to 9 (worst/smallest).
    Only used for .mp3 output; ignored otherwise."""
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
        if ext == ".mp3":
            q = max(0, min(9, int(round(mp3_quality))))
            codec_args = ["-c:a", "libmp3lame", "-q:a", str(q)]
        else:
            codec_args = FFMPEG_ENCODE_ARGS.get(ext, [])
        cmd = [FFMPEG_PATH, "-y", "-i", tmp_wav, *codec_args, out_path]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 creationflags=_SUBPROCESS_NO_WINDOW)
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


# Shared between declick_edges' own default below and AudioPlayer's
# wrap_declick_total (see load()) -- both exist to declick the SAME
# thing (a REPEAT loop's wrap point), just in two different contexts:
# this one bakes the fade into an EXPORTED file, the other applies it
# live during on-screen playback. They drifted apart once already: this
# was 0.05 (matching an earlier, since-changed live value) even after
# the live path moved to a shorter duration specifically because a
# fade audible once per loop cycle needs to be much shorter than one
# that only ever fires from a one-time seek -- confirmed directly, by
# measuring an exported file's actual fade length against a report of
# an "undesirably long" fade when looped externally. A single shared
# constant means the next change to either can't silently leave the
# other stale the same way.
#
# 0.005 (5ms) here, not 0.01 -- this is a TEST value specifically, to
# compare against the previous 10ms before deciding whether REPEAT
# stays in the app at all. Unlike LOOP's crossfade (which blends two
# overlapping copies of the audio, so total energy stays roughly
# constant through the overlap and there's no structural dip), this
# function fades to actual silence at both edges with nothing filling
# the gap -- shortening this reduces the DURATION of that dip but can't
# eliminate it, since eliminating it isn't what this function does.
# 5ms is close to the lower bound literature suggests before a
# fade/crossfade window measurably loses effectiveness at suppressing
# clicks -- shorter than this trades away real safety margin, not just
# "sounds different."
LOOP_WRAP_DECLICK_SECONDS = 0.005


def declick_edges(data, samplerate, fade_seconds=LOOP_WRAP_DECLICK_SECONDS):
    """Applies a short raised-cosine fade-IN at the very start and fade-OUT
    at the very end of `data`, WITHOUT shortening it or blending head into
    tail the way loop_crossfade() does -- this is REPEAT mode's export
    counterpart: same automatic, non-user-adjustable edge treatment as the
    live wrap declick in AudioPlayer._callback (same curve shape, same
    LOOP_WRAP_DECLICK_SECONDS duration), just baked into the file instead
    of applied live during playback, so what Repeat previews and what it
    exports sound the same way at the loop point when the file is looped
    externally.

    Deliberately the SAME curve as the live declick (1-cos(t*pi/2) for the
    fade-in, cos(t*pi/2) for the fade-out) rather than loop_crossfade's
    equal-power/linear options -- this path has no curve CHOICE, matching
    the "automatic, not user-adjustable" design for REPEAT exports."""
    n = len(data)
    fade_n = max(1, min(int(round(fade_seconds * samplerate)), n // 2))
    out = data.astype(np.float64, copy=True)
    t = np.linspace(0.0, 1.0, fade_n, endpoint=False)
    fade_in = 1.0 - np.cos(t * np.pi / 2)
    fade_out = np.cos(t * np.pi / 2)
    out[:fade_n] *= fade_in[:, None]
    out[n - fade_n:] *= fade_out[:, None]
    return out.astype(data.dtype, copy=False)


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
    more of the original transient content and loop crossfades of this
    kind are typically tens to a few hundred ms, not seconds.
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
# PaulXStretch -- extreme time-stretch via phase randomization
# ---------------------------------------------------------------------------

def _paulstretch_mono(data, samplerate, stretch_factor, window_seconds, rng=None):
    """Core single-channel PaulStretch algorithm (Nasca Octavian Paul's
    'Paul's Extreme Sound Stretch', the technique PaulXStretch is built on
    -- see the module docstring note below on feature-completeness vs. the
    actual PaulXStretch application).

    Unlike a phase vocoder or granular stretcher -- which try to preserve
    phase relationships between analysis frames and pay for it with
    metallic/comb-filtered artifacts at extreme ratios -- this deliberately
    RANDOMIZES the phase of every FFT bin per frame while keeping the
    magnitude spectrum intact. That destroys phase coherence (so
    transients and rhythm smear into a diffuse texture) but avoids the
    usual comb-filter/phasiness artifacts, which is why it holds up at
    stretch factors of 10x-100x+ where other methods fall apart. It's
    suited to ambient/drone/texture material, not rhythmic content --
    that smearing is inherent to the technique, not a bug.

    The frame is windowed on BOTH analysis and synthesis (a "squared"
    Hann), which -- unlike a single Hann window -- requires 75% overlap
    between frames (not 50%) to reconstruct at a constant level; using
    50% here previously produced a ~67% amplitude ripple cycling at the
    frame-hop rate, audible as pulsing/uneven loudness. Verified this
    numerically: periodic Hann (denominator N, not N-1) squared at 75%
    overlap sums to a constant to within floating-point precision.
    """
    if rng is None:
        rng = np.random.default_rng()
    data = np.asarray(data, dtype=np.float64)
    n = len(data)
    if n < 4:
        return data.copy()

    windowsize = int(window_seconds * samplerate)
    windowsize = max(16, windowsize)
    windowsize += windowsize % 4  # keep divisible by 4 so the output hop below is a whole number

    # periodic Hann (denominator = windowsize, not windowsize-1) -- the
    # form that actually satisfies constant-overlap-add for STFT use
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(windowsize) / windowsize)
    output_hop = windowsize // 4  # 75% overlap, required for a squared Hann window

    padded = np.concatenate([data, np.zeros(windowsize)])

    start_pos = 0.0
    displace_pos = output_hop / stretch_factor

    out_len_guess = int(n / max(displace_pos, 1e-9) * output_hop) + windowsize * 4
    output = np.zeros(out_len_guess)
    out_pos = 0

    while True:
        istart = int(start_pos)
        if istart + windowsize > len(padded):
            break
        buf = padded[istart:istart + windowsize] * window

        spec = np.fft.rfft(buf)
        magnitude = np.abs(spec)
        phase = rng.uniform(0, 2 * np.pi, size=magnitude.shape)
        new_spec = magnitude * np.exp(1j * phase)
        new_buf = np.fft.irfft(new_spec, n=windowsize) * window

        if out_pos + windowsize > len(output):
            output = np.concatenate([output, np.zeros(len(output) + windowsize)])
        output[out_pos:out_pos + windowsize] += new_buf

        out_pos += output_hop
        start_pos += displace_pos
        if start_pos >= n:
            break

    output = output[:out_pos]

    # Overlap-add with randomized phase alters overall level in a way that's
    # not a fixed constant (depends on window/overlap), so match RMS to the
    # input rather than deriving the exact theoretical compensation factor.
    in_rms = np.sqrt(np.mean(data ** 2) + 1e-12)
    out_rms = np.sqrt(np.mean(output ** 2) + 1e-12)
    if out_rms > 1e-9:
        output *= (in_rms / out_rms)
    return output


def paulstretch(data, samplerate, stretch_factor, window_seconds=0.25):
    """Multi-channel wrapper: stretches each channel independently (each
    gets its own random phase draw, same as the reference implementation --
    this is fine, even desirable, for wide ambient stereo texture, though
    it means stereo channels are no longer phase-correlated the way the
    original recording was)."""
    was_1d = data.ndim == 1
    if was_1d:
        data = data[:, None]
    n_channels = data.shape[1]
    stretched_channels = [
        _paulstretch_mono(data[:, ch], samplerate, stretch_factor, window_seconds)
        for ch in range(n_channels)
    ]
    min_len = min(len(c) for c in stretched_channels)
    result = np.stack([c[:min_len] for c in stretched_channels], axis=1)
    return result[:, 0] if was_1d else result


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
                  snap_transients=False, transient_window=0.25, auto_xfade=False, mp3_quality=2):
    data, sr, sampwidth = decode_to_pcm(in_path)
    result, xfade_seconds, start_trim, end_trim = _run_pipeline(
        data, sr, xfade_seconds, curve, snap_transients, transient_window, auto_xfade)
    encode_from_pcm(result, sr, sampwidth, out_path, mp3_quality=mp3_quality)

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

IS_MACOS = sys.platform == "darwin"
# macOS convention is Command, not Control, for these -- Tk's binding
# syntax for the Cmd key is "Command" (e.g. <Command-z>). The displayed
# label in the Shortcuts window also switches to the macOS glyph (see
# _display_key_label) so a remapped shortcut still reads naturally there.
_UNDO_KEY = "Command-z" if IS_MACOS else "Control-z"
_REDO_KEY = "Shift-Command-z" if IS_MACOS else "Shift-Control-z"

DEFAULT_SHORTCUTS = {
    "play_pause": "space",
    "stop": "s",
    "rewind": "Home",
    "loop_toggle": "r",
    "crop": "c",
    "audition": "l",
    "undo": _UNDO_KEY,
    "redo": _REDO_KEY,
    "zoom_in": "equal",
    "zoom_out": "minus",
    "zoom_fit": "0",
    "zoom_selection": "9",
    "stretch": "x",
}

SHORTCUT_LABELS = {
    "play_pause": "Play / Pause",
    "stop": "Stop",
    "rewind": "Rewind",
    "loop_toggle": "Repeat (declicked edges preview)",
    "crop": "Crop to Selection",
    "audition": "Loop (crossfaded edges preview)",
    "undo": "Undo",
    "redo": "Redo",
    "zoom_in": "Zoom In (Scroll Up)",
    "zoom_out": "Zoom Out (Scroll Down)",
    "zoom_fit": "Zoom to Fit",
    "zoom_selection": "Zoom to Selection",
    "stretch": "PaulXStretch...",
}

# single source of truth for transport-button hover hints (display name +
# description) -- referenced by both initial button creation and the
# refresh-after-remap logic, so a text change only ever needs to happen
# in one place instead of two staying manually in sync
TRANSPORT_HINTS = {
    "play_pause": ("Play / Pause", "Activate / freeze playback"),
    "stop": ("Stop", "Stop playback"),
    "loop_toggle": ("Repeat", "Declicked edges"),
    "audition": ("Loop", "Crossfaded edges"),
    "crop": ("Crop Selected", "Remove unselected audio"),
    "stretch": ("Stretch", "Open PaulXStretch for extreme time-stretching of the current selection"),
}

_MAC_MOD_SYMBOLS = {"Control": "\u2303", "Command": "\u2318", "Cmd": "\u2318",
                     "Shift": "\u21e7", "Alt": "\u2325", "Option": "\u2325"}
_OTHER_MOD_NAMES = {"Control": "Ctrl", "Command": "Cmd", "Shift": "Shift", "Alt": "Alt"}
_KEY_DISPLAY_NAMES = {"space": "Space", "equal": "=", "minus": "-", "Home": "Home",
                       "plus": "+", "Escape": "Esc", "Return": "Return"}
_MOD_DISPLAY_ORDER = ["Control", "Alt", "Option", "Shift", "Command", "Cmd"]


def format_key_for_display(key):
    """Formats a raw Tk key-binding string (e.g. 'Shift-Command-z') for
    on-screen display -- macOS gets its native modifier symbols
    (stacked, no separator, matching how macOS itself shows shortcuts
    e.g. "^\u2318Z"), other platforms get a readable 'Ctrl+Shift+Z' style
    string. Modifiers are always shown in a fixed, conventional order
    regardless of how they happen to be ordered in the raw binding
    string, so the display is consistent no matter how a shortcut was
    remapped. Used both in the Shortcuts window and in tooltip hints, so
    a remapped shortcut displays correctly everywhere it's referenced."""
    if not key:
        return key
    parts = key.split("-")
    base = parts[-1]
    mods = parts[:-1]
    mods = sorted(mods, key=lambda m: _MOD_DISPLAY_ORDER.index(m) if m in _MOD_DISPLAY_ORDER else 99)
    base_display = _KEY_DISPLAY_NAMES.get(base, base.upper() if len(base) == 1 else base.capitalize())
    if IS_MACOS:
        mod_str = "".join(_MAC_MOD_SYMBOLS.get(m, m) for m in mods)
        return f"{mod_str}{base_display}"
    else:
        mod_str = "+".join(_OTHER_MOD_NAMES.get(m, m) for m in mods)
        return f"{mod_str}+{base_display}" if mod_str else base_display


SHORTCUTS_PATH = os.path.join(os.path.expanduser("~"), ".loop_crossfade_shortcuts.json")


def load_shortcuts(path=SHORTCUTS_PATH):
    if os.path.exists(path):
        try:
            with open(path) as f:
                saved = json.load(f)
            merged = dict(DEFAULT_SHORTCUTS)
            merged.update({k: v for k, v in saved.items() if k in DEFAULT_SHORTCUTS})
            # One-time migration: a shortcuts file saved before this
            # platform had its own Command-key defaults (or saved while
            # testing on a different platform) can leave "undo"/"redo"
            # stuck on the old Control-based binding even on a Mac,
            # since a SAVED value always takes precedence over
            # DEFAULT_SHORTCUTS -- this app auto-saves shortcuts on
            # close regardless of whether anything was ever deliberately
            # remapped. Only corrects a value that EXACTLY matches the
            # old, known non-Mac default (not anything else the user
            # might have deliberately remapped it to), and only runs
            # once per file -- a deliberate remap back to Control-z
            # afterward is respected and won't keep getting overridden.
            if IS_MACOS and not saved.get("_mac_defaults_migrated"):
                if merged.get("undo") == "Control-z":
                    merged["undo"] = DEFAULT_SHORTCUTS["undo"]
                if merged.get("redo") == "Shift-Control-z":
                    merged["redo"] = DEFAULT_SHORTCUTS["redo"]
                merged["_mac_defaults_migrated"] = True
            # Same pattern, same reason, for the REPEAT/LOOP key swap
            # (loop_toggle "l"->"r", audition "a"->"l") -- since this app
            # auto-saves the full shortcuts set on every close, ANY
            # existing user's file already has the OLD values explicitly
            # stored even if they never touched these two, which would
            # otherwise silently keep overriding the new defaults
            # forever. Platform-independent (unlike the migration above),
            # since this swap isn't a per-OS default difference.
            if not saved.get("_repeat_loop_keys_migrated"):
                if merged.get("loop_toggle") == "l":
                    merged["loop_toggle"] = DEFAULT_SHORTCUTS["loop_toggle"]
                if merged.get("audition") == "a":
                    merged["audition"] = DEFAULT_SHORTCUTS["audition"]
                merged["_repeat_loop_keys_migrated"] = True
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
# Window size persistence (per named window, e.g. "main", "stretch")
# ---------------------------------------------------------------------------

WINDOW_SIZES_PATH = os.path.join(os.path.expanduser("~"), ".loop_crossfade_window_sizes.json")


def load_window_sizes(path=WINDOW_SIZES_PATH):
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_window_sizes(sizes, path=WINDOW_SIZES_PATH):
    try:
        with open(path, "w") as f:
            json.dump(sizes, f, indent=2)
    except OSError:
        pass


def resolve_window_size(required_w, required_h, saved):
    """Never launches smaller than what's needed to show all content, but
    respects a larger size if the person deliberately resized bigger last
    time."""
    if not saved:
        return required_w, required_h
    try:
        w = max(required_w, int(saved.get("width", required_w)))
        h = max(required_h, int(saved.get("height", required_h)))
        return w, h
    except (TypeError, ValueError):
        # malformed/stale saved data (e.g. from an older, different schema)
        # -- fall back to the freshly-measured size rather than crash
        return required_w, required_h


# ---------------------------------------------------------------------------
# Waveform rendering helper (pure numpy, no GUI dependency)
# ---------------------------------------------------------------------------

def compute_waveform_peaks(data, target_width):
    """Returns (mins, maxs): a mono-mixed min/max amplitude envelope, one
    pair per pixel column, for fast waveform drawing regardless of clip
    length.

    At extreme zoom (fewer samples in view than target_width), this used
    to cap the output to n columns and rely on a later Lanczos upscale to
    stretch it back out -- a 100x+ upscale from a handful of source
    columns produces visible ringing/banding artifacts (the "malformed
    shapes" at high magnification). Instead, when there are fewer samples
    than requested columns, map each output column to its nearest sample
    directly, producing a clean blocky "staircase" -- the correct way to
    represent individual samples at extreme zoom, and how professional
    audio editors render this case."""
    mono = data.mean(axis=1) if data.ndim > 1 else data
    n = len(mono)
    if n == 0:
        return np.zeros(1), np.zeros(1)
    target_width = max(1, target_width)
    if n >= target_width:
        edges = np.linspace(0, n, target_width + 1).astype(int)
        mins = np.empty(target_width)
        maxs = np.empty(target_width)
        for i in range(target_width):
            a, b = edges[i], max(edges[i] + 1, edges[i + 1])
            chunk = mono[a:b]
            mins[i] = chunk.min()
            maxs[i] = chunk.max()
    else:
        idx = np.linspace(0, n - 1, target_width).round().astype(int)
        vals = mono[idx]
        mins, maxs = vals.copy(), vals.copy()
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
    again after cropping).

    Important: sel_start/sel_end describe the marked LOOP region, not a
    hard boundary on where playback is allowed. Clicking outside that
    region should still be able to play -- so play() decides, at the
    moment it's called, whether the cursor is inside the loop region
    (play_start/play_end become the selection, honoring `loop`) or
    outside it (play_start/play_end become the whole buffer, never
    looping). That decision is frozen in play_start/play_end/play_loop
    for the duration of that playback, rather than re-evaluated per
    audio callback -- so scrubbing into the selection mid-playback
    doesn't suddenly start looping unexpectedly.
    """

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
        self.play_start = 0       # effective bounds for the CURRENT playback
        self.play_end = 0
        self.play_loop = False
        self.on_natural_stop = None  # optional callback, called from GUI thread via `after`
        self.declick_total = 1        # samples in the current declick fade-in ramp
        self.wrap_declick_total = 1   # samples in the (shorter) REPEAT-wrap fade -- both
                                       # get a real value in load(); 1 here is just a safe
                                       # default matching declick_total's own pattern above,
                                       # in case anything ever reads it before a file is loaded
        self.declick_remaining = 0    # samples of that ramp still left to apply
        self.declick_total_active = 1  # denominator for the CURRENT fade-in in
                                        # progress (declick_remaining counts down
                                        # against this) -- separate from
                                        # pending_jump_declick_total below so a
                                        # new jump starting mid-fade-in can't
                                        # change the denominator out from under
                                        # an already-in-progress fade-in curve
        self.pending_cursor = None    # target cursor once an in-progress fade-out completes
        self.fadeout_remaining = 0    # samples of the pre-jump fade-out still left to apply
        self.pending_jump_declick_total = 1  # denominator for the CURRENT pending
                                              # jump's fade-out/fade-in curves --
                                              # set to declick_total for an explicit
                                              # seek, or wrap_declick_total for a
                                              # REPEAT loop-wrap (see set_loop):
                                              # the same 50ms fade that's barely
                                              # noticeable on a one-off seek click
                                              # became an audible periodic "dip"
                                              # when the identical duration applied
                                              # to a wrap that repeats every loop
                                              # cycle -- reported directly (a wrap
                                              # dip that "sounds like a fade set to
                                              # a length that's too long")
        self.declick_loop_wrap = False  # True: REPEAT's raw loop -- wrap gets the same
                                         # fade-out/fade-in as a seek. False: an already
                                         # crossfade-PROCESSED preview buffer (Audition/LOOP),
                                         # whose wrap point is already seamless at the
                                         # waveform level -- declicking it again would be
                                         # redundant at best and could dip an already-smooth
                                         # loop's volume for no reason.

    def load(self, data, sr):
        self.stop()
        if data.ndim == 1:
            data = data[:, None]
        self.data = np.ascontiguousarray(data.astype(np.float32))
        self.sr = sr
        self.declick_total = max(1, int(sr * 0.05))  # 50ms fade, applied after an explicit seek
        self.wrap_declick_total = max(1, int(sr * LOOP_WRAP_DECLICK_SECONDS))  # applied at a REPEAT
                                                            # loop wrap specifically -- short enough
                                                            # to still fully mask the underlying
                                                            # discontinuity (the original problem
                                                            # was a genuine waveform discontinuity,
                                                            # not a need for any particular fade
                                                            # length) while being much shorter than
                                                            # the seek fade, since this one repeats
                                                            # every loop cycle rather than firing once.
                                                            # Shared with declick_edges' own default
                                                            # above (see LOOP_WRAP_DECLICK_SECONDS)
                                                            # so the live and exported-file versions
                                                            # of this same declick can't drift apart
        with self.lock:
            self.sel_start, self.sel_end, self.cursor = 0, len(self.data), 0
            self.play_start, self.play_end = 0, len(self.data)

    def _apply_bounds_from_cursor(self):
        """(Re)computes play_start/play_end/play_loop from the current
        cursor position relative to the marked selection. Called from
        play() to establish initial bounds, AND from set_selection(),
        set_loop(), and set_cursor() while a stream is already running --
        without this, changing the selection or toggling Repeat mid-
        playback had no effect on the ALREADY-RUNNING stream, since the
        bounds were previously frozen only at the moment play() was first
        called.

        The selection only constrains playback when self.loop is actually
        enabled (Repeat, or the Loop crossfade-preview, both set this).
        Previously this only checked cursor position, so merely HAVING a
        selection defined -- even with Repeat/Loop both off -- silently
        clipped ordinary playback to stop at the selection's end."""
        if self.loop and self.sel_start <= self.cursor < self.sel_end:
            self.play_start, self.play_end, self.play_loop = self.sel_start, self.sel_end, True
        else:
            self.play_start, self.play_end, self.play_loop = 0, len(self.data), False

    def set_selection(self, start, end):
        """Updates the marked loop region. Deliberately does NOT move the
        cursor -- callers that want to move the cursor (e.g. a click) do
        so explicitly, so this can't silently clamp a click outside the
        selection back into it."""
        with self.lock:
            if self.data is None:
                return
            self.sel_start = max(0, min(start, len(self.data)))
            self.sel_end = max(self.sel_start, min(end, len(self.data)))
            if self.playing:
                self._apply_bounds_from_cursor()

    def set_cursor(self, sample):
        with self.lock:
            if self.data is None:
                return
            target = max(0, min(sample, len(self.data)))
            if self.playing:
                # Don't jump immediately -- fade the CURRENTLY PLAYING
                # material out first (handled in _callback via
                # pending_cursor/fadeout_remaining), THEN jump and fade
                # the new material in via the existing declick ramp
                # below. Jumping immediately only smoothed the incoming
                # side of the cut; the outgoing side stopped abruptly at
                # whatever amplitude it happened to be, which could click
                # just as audibly as an unfaded start would have.
                self.pending_cursor = target
                self.fadeout_remaining = self.declick_total
                self.pending_jump_declick_total = self.declick_total
            else:
                self.cursor = target

    def set_loop(self, value, declick_wrap):
        # declick_wrap has NO default -- every caller must say explicitly
        # whether this loop's wrap point should be declicked (REPEAT's
        # raw selection) or left alone (an already-processed Audition/
        # LOOP preview buffer, whose seam is already seamless at the
        # waveform level). A silently-wrong default here would mean
        # either an unwanted dip in an already-smooth processed loop, or
        # REPEAT quietly going back to clicking -- both are exactly the
        # class of bug that's easy to miss without forcing every call
        # site to state its intent.
        self.loop = bool(value)
        self.declick_loop_wrap = bool(declick_wrap)
        with self.lock:
            if self.data is not None and self.playing:
                self._apply_bounds_from_cursor()

    def _callback(self, outdata, frames, time_info, status):
        with self.lock:
            if self.data is None:
                outdata[:] = 0
                raise _sd.CallbackStop

            # Proactively arm the SAME fade-out/jump/fade-in machinery
            # used for an explicit seek, as soon as the cursor comes
            # within one declick window of the loop's wrap point -- a
            # loop wrap IS, functionally, just a seek from play_end back
            # to play_start, so there's no need for separate wrap-
            # specific fade logic. Only for REPEAT's raw loop
            # (declick_loop_wrap); an already crossfade-processed
            # Audition/LOOP preview buffer is left alone here entirely,
            # since its wrap point is already seamless and doesn't need
            # (or want) this. fadeout_remaining is set to the ACTUAL
            # remaining distance to play_end, not always the full
            # declick_total, so the fade lands at exactly zero gain
            # precisely at play_end regardless of which callback first
            # detects the condition.
            if (self.declick_loop_wrap and self.play_loop and self.pending_cursor is None
                    and 0 < self.play_end - self.cursor <= self.wrap_declick_total):
                self.pending_cursor = self.play_start
                self.fadeout_remaining = self.play_end - self.cursor
                self.pending_jump_declick_total = self.wrap_declick_total

            write_offset = 0
            if self.pending_cursor is not None:
                # A jump is pending: fade the CURRENTLY PLAYING material
                # out first, then jump and fade the new material in below
                # -- see set_cursor() for why this exists. Deliberately
                # NOT splitting this into a separate helper: it needs to
                # write directly into `outdata` at a running offset and
                # interact with the exact same cursor/bounds state as the
                # normal chunk-production code right below it.
                avail = max(0, len(self.data) - self.cursor)
                nfo = min(frames, self.fadeout_remaining, avail)
                if nfo > 0:
                    chunk = self.data[self.cursor:self.cursor + nfo]
                    ch = outdata.shape[1]
                    if chunk.shape[1] != ch:
                        chunk = np.tile(chunk[:, :1], (1, ch)) if chunk.shape[1] == 1 else chunk[:, :ch]
                    # Exact mirror of the fade-IN curve below: starts at
                    # gain 1.0 (no abruptness at the moment the fade
                    # begins) and eases down to 0 with the same
                    # raised-cosine shape, just run in reverse.
                    t_start = 1.0 - self.fadeout_remaining / self.pending_jump_declick_total
                    t_end = 1.0 - (self.fadeout_remaining - nfo) / self.pending_jump_declick_total
                    t = np.linspace(t_start, t_end, nfo, endpoint=False)
                    gains = np.cos(t * np.pi / 2).astype(np.float32).reshape(-1, 1)
                    outdata[:nfo] = chunk * gains
                    self.cursor += nfo
                    self.fadeout_remaining -= nfo
                    write_offset = nfo
                else:
                    self.fadeout_remaining = 0
                if self.cursor >= len(self.data):
                    # Ran out of source samples to fade through before
                    # fadeout_remaining reached zero on its own (e.g. the
                    # outgoing position was close to the very end of the
                    # file) -- force the jump to complete anyway, since
                    # there's nothing left to fade. Without this, a seek
                    # away from near-the-end-of-file could leave
                    # pending_cursor permanently unresolved: this same
                    # branch's completion check only looked at whether
                    # fadeout_remaining had counted down to zero, not at
                    # whether the SOURCE material to fade through had
                    # simply run out first.
                    self.fadeout_remaining = 0
                if self.fadeout_remaining <= 0:
                    # Fade-out complete (or ran out of source samples to
                    # fade through, e.g. the outgoing position was right
                    # at the end of the file) -- jump now. Whatever's left
                    # of THIS callback's frames continues below using the
                    # new cursor, starting the existing fade-in -- most of
                    # the time this means one callback produces a tiny
                    # sliver of fade-out immediately followed by the start
                    # of the fade-in, rather than needing a second
                    # callback round-trip to begin the incoming side.
                    self.cursor = self.pending_cursor
                    self.pending_cursor = None
                    self._apply_bounds_from_cursor()
                    self.declick_remaining = self.pending_jump_declick_total
                    self.declick_total_active = self.pending_jump_declick_total

            frames_left = frames - write_offset
            if frames_left <= 0:
                return

            remaining = self.play_end - self.cursor
            if remaining <= 0:
                if self.play_loop:
                    self.cursor = self.play_start
                    remaining = self.play_end - self.cursor
                else:
                    outdata[write_offset:] = 0
                    self.playing = False
                    raise _sd.CallbackStop
            n = min(frames_left, remaining)
            chunk = self.data[self.cursor:self.cursor + n]
            ch = outdata.shape[1]
            if chunk.shape[1] != ch:
                chunk = np.tile(chunk[:, :1], (1, ch)) if chunk.shape[1] == 1 else chunk[:, :ch]
            outdata[write_offset:write_offset + n] = chunk
            if self.declick_remaining > 0:
                # only applied here (the jump-origin chunk) -- deliberately
                # NOT applied to the loop-wrap continuation below, since a
                # raw/un-crossfaded loop's seam click is something the user
                # explicitly wants to still hear when previewing it.
                # 1-cos(t*pi/2) starts at 0 with ZERO slope (a gentle onset,
                # avoiding any abruptness right at the jump point) and
                # reaches full level with a steeper finish -- this is the
                # correct raised-cosine fade-IN shape. (Note: sin(t*pi/2),
                # used in an earlier version of this, actually has its
                # STEEPEST slope at t=0, the opposite of what was intended.)
                ramp_n = min(n, self.declick_remaining)
                t_start = 1.0 - self.declick_remaining / self.declick_total_active
                t_end = 1.0 - (self.declick_remaining - ramp_n) / self.declick_total_active
                t = np.linspace(t_start, t_end, ramp_n, endpoint=False)
                gains = (1.0 - np.cos(t * np.pi / 2)).astype(np.float32).reshape(-1, 1)
                outdata[write_offset:write_offset + ramp_n] *= gains
                self.declick_remaining -= ramp_n
            self.cursor += n
            if n < frames_left:
                if self.play_loop:
                    self.cursor = self.play_start
                    n2 = min(frames_left - n, self.play_end - self.play_start)
                    chunk2 = self.data[self.play_start:self.play_start + n2]
                    if chunk2.shape[1] != ch:
                        chunk2 = np.tile(chunk2[:, :1], (1, ch)) if chunk2.shape[1] == 1 else chunk2[:, :ch]
                    outdata[write_offset + n:write_offset + n + n2] = chunk2
                    outdata[write_offset + n + n2:] = 0
                    self.cursor += n2
                else:
                    outdata[write_offset + n:] = 0
                    self.playing = False
                    raise _sd.CallbackStop

    def play(self):
        if not SOUNDDEVICE_AVAILABLE or self.data is None:
            return
        with self.lock:
            self._apply_bounds_from_cursor()
            # smooths the start of playback the same way set_cursor() does --
            # covers both a fresh Play press and a restart triggered by
            # reprocessing (e.g. changing the manual crossfade value while
            # auditioning stops/reloads/replays, which is its own kind of
            # jump and had the same audible-pop problem)
            self.declick_remaining = self.declick_total
            self.declick_total_active = self.declick_total
            self.pending_cursor = None  # defensive: no stale jump from a previous session
            self.fadeout_remaining = 0
        channels = self.data.shape[1]
        # Uses the OS/PortAudio's own default latency and blocksize, not an
        # explicit low-latency + blocksize=256 request. That aggressive
        # configuration (a 5.8ms budget at 44.1kHz) was confirmed, via
        # direct A/B testing, to be the actual cause of an intermittent
        # popping sound during LOOP/REPEAT playback -- reported specifically
        # after Crop, though the mechanism isn't crop-specific, just easiest
        # to reproduce with a short, tightly-looped buffer. Every app-level
        # mechanism that could otherwise explain it was checked and ruled
        # out directly first: underruns (sounddevice's own status flag
        # never fired), spurious recomputation, play_start/play_end/
        # play_loop boundary mismatches, unexpected preview_mode/state
        # resets, the explicit-seek declick machinery below, and the
        # crossfaded content itself (confirmed clean on an actual exported
        # file -- its wrap-point sample jump measured SMALLER than the
        # file's own median sample-to-sample variation, i.e. no
        # discontinuity at all). None of those explained it; removing the
        # aggressive latency/blocksize request did, confirmed with a build
        # that logged which configuration was actually in use each time.
        #
        # This does trade away the specific benefit low latency provided:
        # without it, PortAudio's default latency can leave more audio
        # already queued in the driver at any moment, so a mid-playback
        # jump (an explicit seek/click, see set_cursor()) may have a
        # slightly longer window before the declick ramp's new content
        # actually reaches the speaker. That's a comparatively narrow,
        # one-time-per-seek concern; a periodic pop on every loop cycle
        # during ordinary playback is the more disruptive problem, so this
        # is the right trade for the common case. The fallbacks below exist
        # only as a safety net if this exact system/device combination
        # can't open a stream with default settings at all -- ordinary
        # operation should never need them.
        try:
            self.stream = _sd.OutputStream(samplerate=self.sr, channels=channels,
                                            callback=self._callback, dtype="float32")
        except Exception:
            try:
                self.stream = _sd.OutputStream(samplerate=self.sr, channels=channels,
                                                callback=self._callback, dtype="float32",
                                                latency="low", blocksize=256)
            except Exception:
                self.stream = _sd.OutputStream(samplerate=self.sr, channels=channels,
                                                callback=self._callback, dtype="float32",
                                                latency="low")
        self.stream.start()
        self.playing = True

    def swap_playing_buffer(self, data, sr, cursor=0):
        """Replaces the audio buffer WITHOUT stopping/restarting the actual
        OS audio stream, when one is already running at a matching sample
        rate/channel count. Used specifically for reprocessing WHILE
        already auditioning (e.g. changing the crossfade value mid-play) --
        stopping and recreating the whole PortAudio stream for that is a
        heavier operation that can leave already-queued audio from the OLD
        stream cut off abruptly, which is a SEPARATE click source from the
        new-content-onset click the declick ramp handles. Falls back to a
        plain stop+load (no play -- caller is expected to call play() in
        that case) if a live swap isn't possible.

        `cursor` lets a caller land somewhere other than the very start of
        the new buffer -- needed when swapping back to RAW audio at the
        same logical position a processed preview was at (see
        _exit_preview_mode), rather than always restarting from 0 the way
        swapping in a freshly-reprocessed PREVIEW buffer correctly does.
        Landing anywhere outside [sel_start, sel_end) here would silently
        disable looping the moment set_selection() is called next:
        _apply_bounds_from_cursor() only enables play_loop when cursor is
        already WITHIN the selection at the moment it runs, and this old
        default-0 cursor rarely was, for any selection not starting at
        sample 0 -- confirmed directly (playback stopping dead at the
        selection's end instead of looping, immediately after switching
        from LOOP back to REPEAT mid-playback)."""
        if data.ndim == 1:
            data = data[:, None]
        new_data = np.ascontiguousarray(data.astype(np.float32))
        with self.lock:
            can_hot_swap = (self.playing and self.stream is not None
                             and sr == self.sr and self.data is not None
                             and new_data.shape[1] == self.data.shape[1])
            if can_hot_swap:
                self.data = new_data
                self.sel_start, self.sel_end = 0, len(new_data)
                self.cursor = max(0, min(cursor, len(new_data)))
                self.play_start, self.play_end = 0, len(new_data)
                self.play_loop = self.loop
                self.declick_remaining = self.declick_total
                self.declick_total_active = self.declick_total
                return True
        self.stop()
        self.load(new_data, sr)
        return False

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
            self.pending_cursor = None
            self.fadeout_remaining = 0
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
AUDITION_ON = "#2ecc82"       # distinct green while actively auditioning -- so it doesn't
AUDITION_ON_HOVER = "#4fdb9c"  # read the same as the (blue) Loop toggle or default accent buttons
BORDER = "#37393e"
WAVEFORM_COLOR = "#5b8cff"
# FermaLoop wordmark, embedded directly rather than shipped as a
# separate asset file -- keeps the single-file distribution model
# intact and avoids any asset-path resolution differences between
# running from source and a frozen PyInstaller build. Stored at the
# EXACT final display height (32px), pre-resized with PIL's LANCZOS
# filter (properly antialiased) rather than embedded larger and
# scaled down at runtime -- Tk's PhotoImage.subsample()/zoom() are
# confirmed (Tk's own docs) to be naive every-Nth-pixel decimation
# with no blending at all, not a real resampling filter, which is
# what caused a visibly jagged result when that was tried first.
# Loaded via Tk's own native PNG support (PhotoImage(data=...),
# available since Tk 8.6) rather than Pillow, so it still displays
# correctly even when PIL_AVAILABLE is False --
# see _load_header_logo().
LOGO_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAJEAAAAgCAYAAAASTprzAAAbGklEQVR42u17e5RcVZX3b59z763qqupXQhICxoBIEBpBHgMOKt39"
    "GUA6CZkZqRogJKTz6gkKUQEdn7fvh7IUkAUmCJVHd+fRQatUUJIgPuiOS0ZRkIc0ohlIHKDzTrqr63Uf5+z5o6qaTtNJh29c3+f6"
    "4Kx1V3dVnbvvvuf8zj57//Y+BAC2bQvHcXRHx0PTYMh7mHWTUjoGsA60Pv/GJfP/XOmDd9tbWqPNxnaHgtkdPEdEsJYICPL4ytZW"
    "WtPYw8b2Zgr+f35/AwC1t7dz7LTTokGgH42Fo+fm8zlEIhGQIAxmBs13YXJ8jQ2EhcRkEoAAou+U9xZ2T48kIq71jati1dXn5nJD"
    "PoOHCvn8umx26DaT3X4AcByH34XJOE2DOShfwDvGahtNAJzSCFyiAqVDobDpFgq3Ll18/XdGL7R3Eh5s2xYNDQ1U+dzX18ft7e1M"
    "RMceBwKB3llrx+jt7S0hhOhEzVoAAEvut+0eo6bmD+Ytt9xSPBqAmJnS6bTo65tEaALQCzQ07OdEIq4BetugY2YCQESkj5iWUc+v"
    "THBf3yRqaNjPfX19PJa/doR+qOiW0MdYEJRKpUQ8HtejdEDZGiOVSkkASCQS6n+29zHF0xD7JoEm7y/ps68PhCagqRfacUgfP+BZ"
    "9DZBoBeY3ACuyJm8H5xOYNy5iKdYVv5PJ0iNlDl5PzjdB443gPZNAo2pW2VQVq/btGV99w94/eYf8uquzdeWJ8s4xmDLYyl2rN+Z"
    "mcqAGQYFM4vjsQ62bYujyBQjZB5Tv7F+G6kPAHR1pU5etyF98dquhy5f2/W95rUbvnf2I488Uj1Sl5GONQDM6uKr536P+Z9SzLM7"
    "+WYAaOxh41iTdixg2Paxx8S2WWCU3uOB5HjaeP1H/25UVhQDsrLsAajyylNjDTYRcSKRUKtWpWLRWnWhZnkas44QYdBgsWNgYPcf"
    "EomEW+n7FgSO+C6VSslEIqEcx8GazZunSBU62c8feLmtrS1fmVgiQiqVEhVdu7oeOo0FfVBD1Avowz7E80S0c6S8RCKhkt3dJ5gI"
    "XcCe/x4tOJBEO/xC5veJRMI/Itp8U09au/77SwThhkAHZ0NTrZQGmBlKBcGBwcLezo2ppzzPW9u2+PrHjvZ+x7I+AJAmUvEUV3k+"
    "mjXjEjBOASMEwmEivCgNPOFcQy9WJqxiHUZPpJMgBQeYu4k/pAWaEKABAnUMFAWwU0j8urgP29MJcofBNoa+LR3eJWZEVfuDPLSt"
    "reo36QSpKzvcD0rLuIYYH2JGNRGGmPCc9oN0OkHPgZlAVNrE1nR0P80MYuD9QlANEUFpvZMYh5mY3EB/8uZlC3aOmFC2bduaftqZ"
    "nwdoCYDplmWhfB98z4Mg2hFo/Z2lC69dVR7oYRA8uG7j9bFY7OZ8vsBKB8uWL17w/KrVXQ3RqqittZrJjHpS+NDixdc8n+zc9O3a"
    "6trZmczg7mWLrm9KJjd90IpYdyitZ1qmGRZCgJnheV4BhIeFEre0tib2dHZ2hhVV2ULQEinlCUJIEBF83weY/1z0/TuWL563YaQ1"
    "OemkhmoZ0ulIJHKZ53kwTANghlIaRICUBrTWYNaQUmIom131b4uvv9m2bepFu9juUDCri682wkiTAIIcVmxppe8Mh/g2C5S3gdmb"
    "+FNCYAURTic5YnOtrGAPPgk8rjy0b11Iz4wGUuXznE7+MIXQzhoflxaMIzZpAlgBzHgZGvf85HpaMwzkUUCa1em/Ej3ReF92j3px"
    "W6vxwdnr1TeEKT4nLYRBADTAGiACAh+KFe7dMh+32e0gpx0sauvqL6ifOOF8K2TVaK2hlEJ1rPrU+okTzp9QP/E8SboaANrb2yUR"
    "sHLl+onTTj3zl6Fw1e1SyummaaFYLCKbzbJWCqZpQTOfHovFVq7u7N7Y3t5OqVRKHD58WJRBOC0UrvqHSFXkImmYheTajZfHItHf"
    "SCnjRKI+FLJAVLKKzJgeDlfNYMbJazq651gR60lpGLMJCPu+j1wux/l8zlNKVUUjses89p94YN2Gcz22Ho9GI/+ulDrBDwK4bhH5"
    "fM73PDcA4Yy6mpr1q9dt+pTjOPqkk06SjuNolt7qWHXssmx2yAsCnwv5wg8KufyNRa94tVssXJPP575QLBb+EAQBstmsX1db9+m1"
    "nd1XO46jz5mwQ45rgRzSczu5bs5G3mqGsIoETidR+bl0VUAgDJjCwGxp4dez1/O/pROkKlvIMIDW82cohF8JA1cIA8abj3pTDgmA"
    "BD4gQ1h9VTf/sHEVx0DEb90CRdYvQBHBndWpuq1q8SVhIBzkAW8IB7ycPqg8QGuAA3CoBre0dOlOxyEdT0MYA4cP3cclwS2Gac5g"
    "zchmM48waBcA4cPaW1kjPT29csdO40dVVeGPusUiWPNfNfsPaq1/rVjkAt+bDohrpSETQ5mMV1Nbez2/d8ZfEonE7fdt22aUYzzP"
    "cz2Vz+cKAM8jIW6TUlYppaCCYF8Q+H80SRwCAMHk5vN5TUS1mvVmKY2YVgoqUAcBHiDC1EgkFsnn8xgYOKxMyzpTBXjKMIyQ67oQ"
    "QkAF6r8AwLDM9woiuK7rA5AMcpLJ1Ka2tsTgd9duONsw5CcHBwaLpmWZQeDNX9Y6r3sMX+reTF792jTNC4LA18yYCSCdqXnjqH5J"
    "thpkA/RMkquUwFYrikuUB2gPB5jQA43fM+E1ZngE1BFwBgl8jBkfNsIICwMPtHSwSCfou/EUW+kEeS0dfKsRxV2sgaAIDeA3IPyK"
    "GH9RjAwYIQm8lyUuAqOJLEwwIviXWIC6eIqvRBoqzayHLRLB0C4kkTxfWLhAeYDy9I+Z1LfJNf8ECASWf4Zk+SkZEte6GRTDdeKG"
    "lq7gF+kEbTKWLb7+MwDw4LqNEy3LmhH4AZhx97LWa54c4UQajuMEJ0+f8bloLHapWyxCa+5Vpoq3zZt3YMSYPQvgkdVd3VsNw+zI"
    "Dg0pIeiLqzdu7Fx65ZVvrKiEwMySmS0p5dcsKwTXdXeSoHYoY8uypYlDI7cZrbVg5gnhcJX0fO81gL/MAR6rr7cy2ayaUiwWFkkp"
    "vwpAqCDQAEJSGvB973Fmun2o2vzDNACZvP6wJtwbCoXOcV2XAdTB8qcCGDRAl0+YOFFqzfLggX0/bls8vzuZfNrs7x/ihob9DAC5"
    "3CQzkWguJju6u0Oh8D9ozdBcWlw1U04+ql806UkI50LyWzr43tgJuCR/CJ4gfAsG7t9yHe092n1zNvFHlAfHCOHjMoTvzNnIf0gn"
    "6LctXdxshkoAUi4eMyTaH5lHvzuanKs280kqwE2Bh89V1eN/5Q7gW9sW0WfjKZbpsu87wooFQsIMCnrNloVy2ShRBwA82dIZ9JlV"
    "8utBAZqY2uMp/qFh251hNJ0S0M43QswVYOo6u6fHmPBCQR469JTvOI5au3ZttWa+VQVKa6X7h1D858/Oax1IJp826+tf1X19fdzQ"
    "0ECH3/c+sezCCzc8uHbjByKRyBeJRFWhkJ9HRN8a/YJSSvZ876UiF2d+ekHrnkrUs3v3bokyWcfMkFIiCILXfc2XLm+9dtcIEa8B"
    "cB5cu8mMxaJfzuVyflUkYnlFd/2S1msXjnpczz3J5CdqQjV/NAxjglJKKy4NYqDxH7ls9hbNmoRhbLNtW/T3X6AchzQzU3t7O51y"
    "SlNFTvHtRDo/XUHurC6+yIxgqZvBn9nF1T9Z8qbTXH8Y4r+mobRodgCFc6C2N0E9SvQkgJlz1vNd4Ym4Nb8fd8VT3JTP424zChQH"
    "cPuWBfS10ipn0dgEUfUCJE4vPXf/a9Dvq4dOJ6gfwBfnbOAn/By+b0Zw05Ud3JFO0B9tm0UlXGcGpAkjKOpXXt35p0/bNoteQGxv"
    "L41RhY7Y1kzfmN2lPmaExRWwxGnZXNBiALsCp7k1SK7byCNYReU0NwepVIqnTm0gAFpTaKYVDk8lImjWqwb+uiuTSqWq+vqG/Bkz"
    "JlFTU1PJpL/6KqVSKWtwMLivWCwuj0Sitcx8OYBvAQDrYQedACjPK97w6aWte1KplBWPx30i0iMtEbOGZYVkIZf7+vKl83fdt21b"
    "6OYrr/SICMlk0qivr9eDOf1YEARfllKabtHt117mRgBk9/RIp7lZAUAymTTa2tp2r17X/UJVpKo5n8+zkFIDwI3L5v8WwG+H0dbT"
    "Y/T2lkBcjr4YcIrJZNIUQlzv+x6bpnXclCIRbpZhKPcwnjCqcOqczbz/0WuxLw1o0KjIq+z4VnygdIJum9XFU8wI5uey+EqoBucX"
    "DuHBrQvpayP6qO3tzBiVo3vmTT7KTCfo57M7OR6agF8ELm4GsPSlhhG0KAPSAgVFvv8l52yvtxIQOGX+CCj5ZczEnf43tBKXkwQE"
    "i7nGeAPQN2lSOY4TzYZhcCGfZ5PE98vhcaFEBYx5697kuk1PCSmuANGMjRs31syfPz8jpIDWGpYVMovFws9vXLrwGdu2jUQi4Y09"
    "AQKe54IM83nbtsXUbDaohNX19fU6kUioNZ0PmWWZVCjkn2lra8unUimZaG4eHtT+/n7FzLSmc/NY/JMxYcLFcurUbJBIJFTziPu+"
    "291dH/GtaQH7FwO0REp5ke97figUOmZOkcu89dxOrgsYM7UHaYSxXHmoYYHfxNMQ6QSpuZ1cxyZWkInpQQGPbiV6GMyUJlIVruil"
    "CG4sumi0qvHVII8dajI+E0+xTPeB4ZCuAG9OFydkFJ/QRbwCF9/5yRIaKoPMK/tTv5zVyWsh8K+NPbw8XQEdM0gAfh5KI3gcYGrq"
    "hd4+6p0qxOWeJP92CuldoZg4FYTzxwURept0OWI8k7UmrXXgMd/6YMemAWIQEcbggUDMYICmucUig3lS1hdTAGRKloghpIAk2sLM"
    "1N7be1QkVpoKAuk4jh6LKNRas5QV40UaYALSGIufWt3RPab8FStaXNu2rY7135vJoI+y1g3MfCp7fLISalJ1dS0AIJ/PvYWYPGr6"
    "A4DPOCtciyl+Ab9gjS9tmU+/r2xB8RRb+Ty2VNXgI0ERCNWidc4Gbn2UqKvCA5UpgmxLF99bMxn3DLyKu37aQm5jDxtwSlFbmkjN"
    "Xs83WrW4X3mAVQcUDqKx0eaWpnZoB+Cz+hDYNosXDDgBYUnsNTQAeB5gItJaGEDg6r3KDP0VIHbauZIPG/lSXI4O/Vld6kUQTiUS"
    "U48Jor6+PnKcRMU3mai0BhFkrKb2U4LGJZjhukX4vo9QKBRy3cKEN7cxArMGIA6VeafxM8VSvo00CjGQOu7ejuME9yc7Z0WisTsJ"
    "OCsUCsOyLCilUCjk4bpebmgo06+U6iOikGlaV1T8x6OjvrS4LIk/qwDnPTqPnhvJ9qYTpArr+QIjjI8UDsEnQMsIDGYsB9CV7ivd"
    "v72ZAjBTrgsP0Ot4MhLFcwBTpbykZB0AMJYHBSjlIggKENLCZdFTcLbTSs/ZzMIh0o02G9vn0+uzk3ymYWFPZaw0K5Snc+DnCyhf"
    "sQRjtX195S2QUQkKouI4xpjLMkXF5gxlhg4OZgb2jne5bvGA1uqAZn2QWYq3WhD1tuj4v3Wr8FHJdd0zo9HqLWA+i5mRy2UPHj50"
    "aH0mO9jq+fpioTFjMCLPbVt8/T8L0A9MyxJajwOi8tv6BiYEOTSM5HjOipdGkgPkwQAIggEWApKBLADY7Uey3GED9ZaJSwo+asBv"
    "fg972K/JCQOSAQZBsgZYIA8AL6VLfbY7FFyR5KmiDh/3gfCRFCcAEsK2K1b2+NfsMS1RQ0MDv5keoCwRgUgIrfW/hIT/QhBEhWHk"
    "xk8UBlGEZL5YUo3VCJ35/y2I/Iru3wQBrKE161/5jNZRUSBsZpFKpeRALqgZ1wqN2M7go96KYdOsDv7dWX14pWIVbJuFs5ien9XF"
    "60I1WMwaUvnIQuN2gKky8Y29kNubKZAdfEv1+3HLoT+jFkROYw8b24Eg3gBKg0kz2nWAh60YwiQAN4OVjy2kv1SAG0+x5bpYzgJf"
    "MkxM9gV+DmBPCXOatAKg9aSnpwzVAjhcBvdb2uQGVEL4qeXAJzuuT1Qph2DwG0QEwzCglSdbW1sH3s6EjZew/b8PICIuUnHl+vUT"
    "EfAHAz9ggLPs8zXLl163N5lMmv39/dzQ0MB9fX28e/Vqif5+ddL0Gd5xWjkNAHtcPHsi4QAT7nYcmrvsJDYBZscBA0xbF9KSWev5"
    "R1YVpqkBPLF1Oe2oONaNdilCumw1nyotLBvYiYNk4OYrNvLqx5tp93A6hJm2Ef10diefJyw0ejns3LqQfmbbLJw+8NxOPqXooVsY"
    "uEQIoDiIF2NRvFLyHYlBijgApEkTPUTOAPNT8TTEaB4JYEonoBs7OQzW57IGQHht3O1sUjk6A9OzRIBhGADRxcxMyeTTZiUjP/oa"
    "nW0flvN305iYoap8cYIQwjJNg7RWf1q69Lq9qVRKtrW1+Y7jBOXksJ5ZX0+O42jWPOW4XqQ8/M+0kQ9gY2wKrmpZx4tXt5Hf2AMZ"
    "T2F4fLbeQNseTlBy63LaUcncX5Bkc7tDQdxmK1yFbgCRoICbhIEJJrDJtlmkE6QuSLJZyehvaaWXH/4kJbfMp5+BmZz2UgkH1+Mg"
    "CcT9IlaZUUAQHk4nSDX2QI5Ml8gQQUDGQcSvHsZbsFHqTxzRwUyjSryHS0U1T41riXrL0Zlketx13TuklCDCPCL6ZiqV0un0qyIR"
    "j+uKI2bbNgEQjuMEyXWb7oxEIlPyhfyhXbt2fRFAQJr+LsDEDEgpyRWUN1lrrbVg0ORKrVDJEs1goBcNDQ2cSCS8+zs6ppmGsaxY"
    "LKqqqipZet8eI5M9II6VNxMbsbI4gDYziuSsDVy1tZlWjUymDsVgFPaCJkeh0YcARPwM4Ld08okFA52RifjH3F58f9tiemhWB8+L"
    "nohZzzB+NDvJS7a00QG0AbCZ4im29uUgqjLg6jSCdIJU2ZoMzd7I860IFrmDyAgLDwJM25vetDREgF+EliYWX35//t6ftdFr8RRb"
    "leXw6mGI7c3kN9o9hjTE16HBWoEIOj0uiBynRP4tWXLds6s7Nj1pGOZHrVDo7OS67tsTicRXx4h0GIBOdmxeVFMduy0UCqNQKDzW"
    "2tpa8okE/11USBKgPeLwvtZrd53U0f2GZj7ZsqxT13Q+9LWlrdf+77a2Nn9k/9UdD33MkPSgZYVPzOezLhHJ0vs2B8DRC/E/8Ris"
    "RxfQzpYOtqvqcRcYK6/q5ivAuOfED+DXqy8kH6O2jZZOPlEaiDNwmxXFtMIBHGATX7CZxdOb8dnCQXzUimGuRzjnqk18Z8jCD50E"
    "7QdwxFb7iW0cMg/gUgjcKk1cLkOAX8CNWxdSf6PNxnYaoXfJJ1TSEjVWTSj9iXU8J12SOWxbL7trdzQ0ZXKHtMS5YLBW+veRqNFr"
    "HM+AV/wirenzAP9H4Ae+ZZlfWbv+ofcCvLJYZby8v68v39DQYAwMuJPJlP8qpPh6Lpfz8vm8JEH/PqpC8e0Aif+W/cq8FoPA0Np0"
    "iPTqzu6VsVjszoHDhz3Lspw1nZvnaubfgTBImqMgOscwxKWRaBSDAwO7iOiUQqEQSCmvWr/xBzuKAR5ta736ZcmKgCOpiEJVielN"
    "J+juWV38gVANFmsfs5WP2f0v4S+zN/AL0HgdgAdCHYD3g/AhYWKCkEBQRE5pJLYtoL/uzbH5TBvtmNXB16kifmSGcKrWeKBYxO2z"
    "N/BzzHiFGIMghECYhv04hwy8XxiAMIDiAFZuXUgPlPSBGjPQYpA0xcVg/ftZXUFSa34aAATReSRpkbTEGcqDZ4RhgfXn0wmpjgtE"
    "iURClYu9fvPAmg03VVfXrHQ9F6ZhLnBdd7415L1+0ntPzw4MeQYMMdmyrFqlFCLRKDKZgS+2LZ7/wn333RdasWKFW3Fqj6cQmajU"
    "tZQh4XEd5VL0eGzBzBBERMyQmkuRYl3UvGdwYPDsmpqaBa7rIhwOny+lcT4RlUhRIZHJDPqZwcydKkzflkX8vLq65oJMZnBK7YT6"
    "O4t79jUAWAgoCZJv0SCdQCkaW0hLZm/gvULiC9KCZIUZJDBjZP+ynwEhAeVjh++i9bFWerJC8pX/bruyg1uMENYZFk6BiRNAmEkC"
    "M0cuKdYASUAH8JWH27cupNvLADqyZJbBZADk6r3Kxc+MiFhghMV0AHdof7g8BcoHdACYUVjuoLpt2yKzN54ayd0MTwJBa01HA9Ly"
    "pQtWDeWG4oLEf4IIkWiUqmtqp9XW1p1ZU1t3eqy6urYEEdo9NDi4vG3x/G+mUil5zjnnqIrCzKyZWYtxQ3zytWaPtfb4GMQMMzNr"
    "7WnNHsD+OHXcfqkfXCFKubO+vj5euui6G7L53Dyt+ZdFt7gnlxvKZoeGBrKZoT8NZYa+q7S6cNmi675y47x5hyXzJ3O53E8ItC87"
    "lHEZXCzX7zAzNDM0iZGop1I0ZrPYsoC+HChcygoPa4XMEZs7AxwAWuGVwIMzuBcXjQCQqpCU8RTLxxbRE/lDuFC5uEMr7GJ15Dpj"
    "DegAgzpAWit85NH5dHvFGR+z5poBBszdnlgSFNUC7etXWAHSKl2gEpCI8Io7oK7btsi4u6IXVbaZjo6HpsEK1wE+dDGza8mSJUNj"
    "FclXyk+TyWTEiNS3QKlLNWM6wCGA8lLQa5rE7wqDhZ/edNMNB0cfevxud3d9VFRNgudiQAR7VsyfnxnrOQCQ7O4+wRKRWLFYhJep"
    "3b1iRYs7FjA6OzvDQPhEhMMo+EP5Gxcs2Hc0EK3q7DwxGq4Lq1yOTzvt5DfKeTJi5uGy3VQqVVsooFYI6b/nPRP2V3JpqVRKlov4"
    "GQDWpFITYsqo35stZj+z9Lq9V63lahEt8ydF7PtxKw0Mh9GjqhIBYO5DPI0DnMfAdCaEhMAhIrxs9ePZ9C1UqPBTzhiHBkbKuWwD"
    "R0MS50HjAyDUEaEIxi4d4NmtC+mNY5XZliob1YtmVDT4OXVY6fwZP11cs/+yuzgamhpcSprOAagOgjPE/IL+z309W5yT80fT62/O"
    "+fy9cUPHo+9YBwFGf3+sAwPH02xmYY9zMKHRZmPcInxmGq+o3ubxC/5ndaoX/ynFPKtLHfz4ysGJb7dQn0YW4Le3lz47DvF4TsjI"
    "4ziVYzuVYzxAL9rb29VYReyV+pwRkdyxtqmRffWxTz6UJnW8s2HHI7PSpyzrqE77qH4MMNnlNITTDsZ4Rfw2i8pRnGFGeD84HYfG"
    "2zwAUKn3OUJOJcs/Thu2RHl9qDiUnfHLm2oPglnE08en2zvsmN277W2AiI4XyOLdIXy3/U/buyB6t70Lonfb3yAFNMy6/J9lE/4b"
    "dxIo6Ph7ZNYAAAAASUVORK5CYII="
)

# App/taskbar/titlebar icon, embedded the same way as the wordmark
# above and for the same reasons. This is DIFFERENT from (and in
# addition to) PyInstaller's --icon build flag: --icon only sets the
# .exe file's own icon resource, which is what Explorer/Finder show
# for the file itself -- it does NOT set the icon of the actual
# running window. Windows in particular always uses the running
# window's own icon for the taskbar, falling back to Tk's built-in
# default (the well-known "feather" icon) unless the app explicitly
# sets one at runtime -- see the iconphoto() call in __init__.
#
# THREE separate sizes (16/32/48), not one -- iconphoto() accepts
# multiple images and lets Windows pick the exact match for each
# context (16px titlebar, 32px taskbar, 48px Alt-Tab/large icons)
# rather than scaling a single source at runtime. A single 64px
# image was previously provided for all contexts, and Tk's own
# runtime scaling down to 16px for the titlebar produced a visibly
# jagged result -- the same root cause (naive, non-antialiased
# scaling) already diagnosed and fixed for the header wordmark.
# Loop transport-button artwork, embedded the same way as the
# wordmark/app-icon above. This one is a solid-color shape on a
# transparent background rather than a fixed-color image -- its own
# alpha channel is used directly as a recolorable mask in
# render_icon_image()'s "loop" branch, so it still responds
# correctly to hover/active-state color changes and the existing
# rotate-while-previewing animation, the same way the procedurally-
# drawn icons already did.
LOOP_ICON_MASK_PNG_B64 = (
    "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"
)

APP_ICON_16_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAADTklEQVR42k2UXWiWZRyHf///fd/Px/tsc1+sOTOZfUmQmhVRoc4P"
    "qlEIYjvoJJHwYweGGYQdhdCZsoM80KVQ4oEwsASDQcHcVsw+lmaUUsTEtdZc27u92973fe77ee777qAsf3AdXofXjwDAe09E5AHg"
    "UM/oRia5r1JNN1VT3ZbnOTzRJDOGq6n+8MKxzi/xj0Qg8nRX3v3e5Wh5S2uPzuze1DgpySBSDrm1mF/MsFC1gM9zJnd6emry8NDZ"
    "PSm8J+rq6xOPlZ9WRtt+kjUdMDN+y5NNbv0j9dxcHxEATBer/tuf7riLg7d4wYRkTWlwdjrubKn8lREAvNs7dlLFjQdquWj271wZ"
    "1BYkdOZQLGkAQGNdgDAQKC0ZvH9m1NyeocDq4qlPjr/STUfOTDxLECOKMvvOa20iNQ4Xh6cwNrGAaprB2hxSAO1tCXZtW4U4FHi7"
    "54otVYxg555jKdTBDDFeeqYB5dTh2PlxXP9tCWlG0Pm/ZMDozSI+OH8DRIRXt6+GQwCp+CAbJzcnoceaByI+98UsMsuoTUJEocLO"
    "jhXYtXUlCnGAmkKI9hV1cM5jy1Nt3JAIALxZ5p5blQQ++6ZME7MOy5IA84saO55vxtYNDbi7CwO/49qvJUxM38SGNY1UiAS0plYG"
    "C2ROoKVBYXlTiLJhEEsQ8X8yE4NZoC4J8eeMxp3ZFFGgIIUAg9SUdiE61hb8/pcbsPbBAoJA4fIPSxi4WsLA93MYuDqHupoQFe3R"
    "fn8durav8tYLBIGcYhZiqGIDfP2LcUzAH7OAzgV0zrh0pYRLI3PILKOiCVGo8MaO1fj5VslVjEIcBUOceZxgcvj8uocHYc8LNVj/"
    "UIw4UoijAHEcoBArPPHoMhx5/WEksUD/yAziECApTxAAvPWxPSkTPtAsU9P9YhjUxgSdecwvWQBAfQ0jVIzFikXvp+NmTieBKU+d"
    "Ov7mum7q6vOiUIZqFK5fxtyhdOq3Pc5uXbvgplpBAFBcyP2PYxU3eG2OnainvDozGISi80byXfZ/TB/5qFmhx3m31zmWMWskyoC8"
    "RVV7ZJYhoXNB5vT47fzw2aPtqfee6N40AeDQObMxkGKfy+0ma0wbwyIQmFTSD5PVvUd33/fVvRfwN7U1icZeOtEsAAAAAElFTkSu"
    "QmCC"
)

APP_ICON_32_PNG_B64 = (
    "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"
    "R3hHMYIAAAAASUVORK5CYII="
)

APP_ICON_48_PNG_B64 = (
    "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"
    "cbIhT2g0AAAAAElFTkSuQmCC"
)

APP_ICON_512_PNG_B64 = (
    "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"
    "4uDxcP8DETlxQaM1rav9lB7LAE5kFH61JYev/uwEUCwmMeb6DATUDQQwA7YEzlyWNB54pgEg+vesZWnYFkFXdEwGAuYqBMT9TXO4"
    "26AQAj++by8+9OUt+OUfDmNgyAMzQxBBCoIQVAwcrPwTvi6iYwmEvYcy+MGv94zbEbCeIYAZFFX+XWrV5rvHeIu4hZX9mcCoNRiK"
    "Kg/zY84Xa+lcMUNj9OCxZyvWYy9rxaNpf44k7NhfwKe+52H5QhsrOiXamiU6miWaEwTXDiMi40ZPBPgBkPWAoZzCUEZhcDjA/kHG"
    "/qM+MnkNWyBM/Stt4VxyfSX3LPw+FG6+zBPf04kQiBHt30zln1fzfacxo8XYc1S6pirPqPJlVr9+jLnucc8QVZ55fM/Krnv8Pat2"
    "meXPovz7jv+6Jf8VfZZSGklXYuWiOPrfyP8zZfG9X7UkjVRCouDrcGlgTJ+iqIJgTT2jln5U5Zii/2ca3Vi9pCGOvl5jv+PRwZwn"
    "GabHfkeeYCgpO1+1jkcEYh4tfc0T953Re10SL1P2+aMvVDrFZN8xVtqaUjb2H8niC997Es1JC0vmp7B0YRMWtCfQ1myhKWmjOT1a"
    "3je+BkGEbD7AcC7A8RN5HBnI4+kDGezadwL5goJrj18SnqzdVGsq1d5X7TtWPk/52MM1jq8T3tPSthc+S7YY1MasayDNOoIAhEFc"
    "SgM79np47JlwRm9bYYCgqDA7ZwBaAYFm+CrEA0sCliQknKh+kq7esQ0E1AcEKGYs7nTQnJST6RzGTjcARP82N9lY3JnCU/uGYdlU"
    "ntprIKBuICAEAcCxJQiMgq+w49khPLZ7EByNx1IK2GWFoXRRPVSKESiGHygEiuFYArYt4Tpi9Ds1CgSEhY/aLRaynVkDzDS5BFZf"
    "SgBRWMo14YR/1sU1fq6oRwsKdxF0bQlQ+VpTqW8yEFCfEEAEKAUsneeMnssQwMwuA0TPYOk8F088MwzXGdOfDATUHQTECR9CEFxX"
    "IAFZtlyg4gJCPN6J2pLg2HZJ7BdHFVu55ns61yGAiCnMAqAOAUbA9BzW8Od4TACPDeqLHlxx3UiMriFx6fGq8sPgSdb5TExAFQgo"
    "+czZGxMQtouFHW7UdkwA4MwDQPgMFs1LhgNu1YFgdFCuuWeYmIDiNc2WmICxZ+SSoD6OJhAEQEaxWvHaP5XEh2rNURZBhUnLSa7t"
    "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"
    "tP8fEHK4wwI/v9IAAAAASUVORK5CYII="
)




ZERO_LINE_COLOR = "#9a9da3"  # matches MUTED -- verified 5.43:1 contrast against the panel (WCAG-checked, not eyeballed); a literally darker grey actually reads WORSE here, since it blends toward the panel background instead of standing out from it

TOOLTIP_BG = "#111214"       # near-black, existing tooltip background
TOOLTIP_BORDER = "#2a2c30"
TOOLTIP_WRAPLENGTH = 260  # applies to every tooltip's text, so a long
                          # description always wraps to a readable box
                          # width instead of running off-screen as one
                          # unbounded line
KEYCAP_BG = "#0a0b0c"        # darker than the tooltip itself -- a recessed "simulated key" look
KEYCAP_BORDER = "#2a2c30"
KEYCAP_FG = "#c8cad0"        # light enough to read clearly against the darker keycap
SELECTION_COLOR = "#3a4a6b"
PLAYHEAD_COLOR = "#ff5c5c"
HANDLE_COLOR = "#e6e6e8"


def _hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def format_time(seconds):
    """HH:MM:SS.mmm"""
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def pick_tick_interval(span_sec, canvas_width_px=None, min_label_spacing_px=70, target_ticks=8):
    """Chooses a 'nice' timeline tick spacing (in seconds) for a given
    visible time span. If canvas_width_px is given, the target tick count
    scales with available space (roughly one label per min_label_spacing_px)
    instead of a fixed count -- so a wide window shows proportionally more
    labeled ticks rather than the same handful stretched across it."""
    candidates = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5,
                  1, 2, 5, 10, 15, 30, 60, 120, 300, 600]
    if canvas_width_px:
        target_ticks = max(3, int(canvas_width_px / min_label_spacing_px))
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


def render_rounded_box_with_tail_image(w, h, radius, fill_hex, border_hex, tail_x,
                                        tail_w=16, tail_h=8, supersample=4):
    """Same rounded box as render_rounded_box_image, plus a small upward-
    pointing triangular tail baked into the TOP edge at horizontal
    position tail_x -- a speech-bubble/callout shape, used to visually
    connect the XFADE CURVE/OVERLAP/LOOP ALIGNMENT group to the LOOP
    button above it: the tail points at what the box is commentary on.

    `h` is the MAIN BOX's own height; the returned image is h+tail_h
    tall overall, with the main box occupying the bottom h pixels and
    the tail occupying the top tail_h-pixel strip.

    fill_hex may be None for an OUTLINE-ONLY box (fully transparent
    interior, just the border stroke) -- used for the group container
    specifically, so the margin between this outline and the individual
    boxes it wraps shows the canvas's own background color underneath,
    rather than this shape's own fill creating a second, visually
    mismatched panel-colored layer in that margin (and in the gaps
    between the individual boxes, which sit on top of this one but
    don't cover its full interior either).

    Draw order matters here. When fill_hex IS given: the main rect (with
    its full border, including the top edge) is drawn first, then the
    tail's FILL is drawn on top -- deliberately covering the portion of
    the box's own top border line that falls within the tail's base --
    and only then are the tail's two slanted SIDE edges drawn with the
    border color (not its base). When fill_hex is None, there's no fill
    to cover that segment with, so it's explicitly ERASED (painted fully
    transparent) instead before the tail's two slanted edges are drawn.
    Either way, the goal is the same: the final outline reads as one
    continuous path (box top edge -> up one tail side -> point -> down
    the other tail side -> back to box top edge) rather than a separate
    triangle glued on top with a stray line across its own base."""
    w, h = max(1, int(w)), max(1, int(h))
    tail_h = max(0, int(tail_h))
    total_h = h + tail_h
    sw, sh = w * supersample, total_h * supersample
    img = Image.new("RGBA", (sw, sh), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    box_top = tail_h * supersample
    fill_rgba = (_hex_to_rgb(fill_hex) + (255,)) if fill_hex else None
    border_rgba = _hex_to_rgb(border_hex) + (255,)
    border_w = max(1, supersample)

    draw.rounded_rectangle([0, box_top, sw - 1, sh - 1], radius=radius * supersample,
                            fill=fill_rgba, outline=border_rgba, width=border_w)

    if tail_h > 0:
        tx = tail_x * supersample
        tw = tail_w * supersample
        if fill_rgba is not None:
            draw.polygon([(tx - tw / 2, box_top + border_w), (tx + tw / 2, box_top + border_w), (tx, 0)],
                         fill=fill_rgba)
        else:
            draw.rectangle([tx - tw / 2, box_top - border_w, tx + tw / 2, box_top + border_w],
                            fill=(0, 0, 0, 0))
        draw.line([(tx - tw / 2, box_top), (tx, 0)], fill=border_rgba, width=border_w)
        draw.line([(tx, 0), (tx + tw / 2, box_top)], fill=border_rgba, width=border_w)

    return img.resize((w, total_h), Image.LANCZOS)


def render_dropdown_bg_image(w, h, radius, fill_hex, border_hex, caret_hex, supersample=4):
    """Same rounded box as render_rounded_box_image, plus an anti-aliased
    caret triangle baked in -- the caret used to be drawn separately as a
    raw canvas polygon, which (like the rounded corners before PIL
    rendering was added) isn't anti-aliased and looks jagged/distorted at
    this size."""
    w, h = max(1, int(w)), max(1, int(h))
    sw, sh = w * supersample, h * supersample
    img = Image.new("RGBA", (sw, sh), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([0, 0, sw - 1, sh - 1], radius=radius * supersample,
                            fill=_hex_to_rgb(fill_hex) + (255,),
                            outline=_hex_to_rgb(border_hex) + (255,), width=max(1, supersample))
    cx, cy = sw - 16 * supersample, sh / 2
    s = 5 * supersample
    draw.polygon([(cx - s, cy - 3 * supersample), (cx + s, cy - 3 * supersample), (cx, cy + 4 * supersample)],
                 fill=_hex_to_rgb(caret_hex) + (255,))
    return img.resize((w, h), Image.LANCZOS)


def render_slider_image(w, h, track_radius, thumb_x, bg_hex, track_hex, fill_hex, thumb_hex, supersample=4):
    """Anti-aliased horizontal slider: a rounded track, a filled 'progress'
    portion up to the thumb, and a circular thumb -- all PIL-rendered for
    the same reason the other custom widgets are (native ttk.Scale
    rendering, especially the trough/thumb on Windows, largely ignores
    ttk.Style overrides)."""
    w, h = max(1, int(w)), max(1, int(h))
    sw, sh = w * supersample, h * supersample
    img = Image.new("RGBA", (sw, sh), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    cy = sh / 2
    pad = h / 2 * supersample
    track_h = max(2, int(sh * 0.28))
    draw.rounded_rectangle([pad, cy - track_h / 2, sw - pad, cy + track_h / 2],
                            radius=track_h / 2, fill=_hex_to_rgb(track_hex) + (255,))
    tx = thumb_x * supersample
    if tx > pad:
        draw.rounded_rectangle([pad, cy - track_h / 2, max(pad + 1, tx), cy + track_h / 2],
                                radius=track_h / 2, fill=_hex_to_rgb(fill_hex) + (255,))
    r = sh * 0.42
    draw.ellipse([tx - r, cy - r, tx + r, cy + r], fill=_hex_to_rgb(thumb_hex) + (255,))
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


def render_radio_image(size, selected, bg_hex, accent_hex, muted_hex, supersample=6):
    """Anti-aliased circular radio-button indicator: filled accent dot
    when selected, empty muted-outline ring otherwise."""
    size = max(1, int(size))
    sw = size * supersample
    img = Image.new("RGBA", (sw, sw), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    cx = cy = sw / 2
    r = sw * 0.42
    color = _hex_to_rgb(accent_hex) if selected else _hex_to_rgb(muted_hex)
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color + (255,), width=max(2, int(sw * 0.09)))
    if selected:
        ir = r * 0.5
        draw.ellipse([cx - ir, cy - ir, cx + ir, cy + ir], fill=color + (255,))
    return img.resize((size, size), Image.LANCZOS)


def render_icon_image(name, size, color_hex, supersample=6, rotation_deg=0):
    """Simple vector-style transport/tool icons (play, pause, stop, loop,
    repeat, stretch, crop), drawn via PIL and supersampled for clean anti-
    aliased edges at small sizes. `rotation_deg` only affects "loop" (used
    to animate it while a crossfade preview is actively playing)."""
    sw = size * supersample
    img = Image.new("RGBA", (sw, sw), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    color = _hex_to_rgb(color_hex) + (255,)
    pad = sw * 0.18
    cx, cy = sw / 2, sw / 2

    if name == "play":
        draw.polygon([(pad, pad * 0.5), (pad, sw - pad * 0.5), (sw - pad * 0.6, sw / 2)], fill=color)
    elif name == "pause":
        bw, gap = sw * 0.24, sw * 0.16
        x0, x1 = sw / 2 - gap / 2 - bw, sw / 2 + gap / 2
        draw.rectangle([x0, pad, x0 + bw, sw - pad], fill=color)
        draw.rectangle([x1, pad, x1 + bw, sw - pad], fill=color)
    elif name == "stop":
        draw.rectangle([pad, pad, sw - pad, sw - pad], fill=color)
    elif name == "loop":
        # Custom-designed artwork (matches the app's own icon/wordmark
        # eye motif), replacing the geometric version below -- which is
        # deliberately KEPT, not deleted, as "loop_legacy", in case it's
        # ever wanted again. The source PNG is already a solid-color
        # shape on a transparent background, so its own alpha channel is
        # a ready-made mask: resize (LANCZOS, properly antialiased -- the
        # whole reason this needs care, per the header wordmark and app
        # icon fixes earlier) and rotate (BICUBIC, also antialiased) the
        # MASK at full quality, THEN recolor by compositing a flat fill
        # of whatever `color` was requested through that alpha -- this
        # is what lets the same artwork still respond correctly to hover/
        # active-state color changes and the existing rotate-while-
        # previewing animation, exactly like the procedural icons do.
        mask_img = Image.open(io.BytesIO(base64.b64decode(LOOP_ICON_MASK_PNG_B64))).convert("RGBA")
        mask_resized = mask_img.resize((sw, sw), Image.LANCZOS)
        if rotation_deg:
            mask_resized = mask_resized.rotate(-rotation_deg, resample=Image.BICUBIC, expand=False)
        alpha = mask_resized.split()[3]
        solid_color = Image.new("RGBA", (sw, sw), color)
        img.paste(solid_color, (0, 0), alpha)
    elif name == "loop_legacy":
        # sized/weighted to match the visual density of the filled icons
        # (play/stop/crop) -- the original thinner, smaller circle read as
        # noticeably lighter/smaller next to them. rotation_deg animates it
        # while actively playing the processed/crossfaded preview.
        r = sw * 0.42
        bbox = [cx - r, cy - r, cx + r, cy + r]
        width = max(3, int(sw * 0.19))
        draw.arc(bbox, start=25 + rotation_deg, end=155 + rotation_deg, fill=color, width=width)
        draw.arc(bbox, start=205 + rotation_deg, end=335 + rotation_deg, fill=color, width=width)

        def arrowhead(angle_deg, s):
            a = math.radians(angle_deg)
            tipx, tipy = cx + r * math.cos(a), cy + r * math.sin(a)
            a1, a2 = a + math.radians(135), a - math.radians(135)
            p1 = (tipx + s * math.cos(a1), tipy + s * math.sin(a1))
            p2 = (tipx + s * math.cos(a2), tipy + s * math.sin(a2))
            draw.polygon([(tipx, tipy), p1, p2], fill=color)

        arrowhead(155 + rotation_deg, sw * 0.22)
        arrowhead(335 + rotation_deg, sw * 0.22)
    elif name == "repeat":
        # musical repeat barlines: ||: :||  -- loops the RAW selection with
        # no crossfade processing (you'll hear a click at the seam)
        top, bot = sw * 0.14, sw * 0.86
        thick_w = max(2, int(sw * 0.095))
        thin_w = max(2, int(sw * 0.055))
        dot_r = sw * 0.048
        half_gap = sw * 0.11
        for sign in (-1, 1):
            x = cx + sign * half_gap
            draw.ellipse([x - dot_r, sw * 0.34 - dot_r, x + dot_r, sw * 0.34 + dot_r], fill=color)
            draw.ellipse([x - dot_r, sw * 0.66 - dot_r, x + dot_r, sw * 0.66 + dot_r], fill=color)
            x += sign * dot_r * 2.6
            draw.line([(x, top), (x, bot)], fill=color, width=thin_w)
            x += sign * thin_w * 2.8
            draw.line([(x, top), (x, bot)], fill=color, width=thick_w)
    elif name == "stretch":
        # calipers being pulled apart by a double-headed arrow. Arrowhead
        # base spread widened from the original 0.68x to 1.0x -- at 0.68
        # the base was only modestly wider than the shaft itself, which
        # read as barely-there wings rather than a clearly double-headed
        # arrow at this icon's actual small display size.
        pad2, top, bot, L = sw * 0.08, sw * 0.18, sw * 0.82, sw * 0.16
        w = max(3, int(sw * 0.13))
        draw.line([(pad2 + L, top), (pad2, top), (pad2, bot), (pad2 + L, bot)],
                  fill=color, width=w, joint="curve")
        draw.line([(sw - pad2 - L, top), (sw - pad2, top), (sw - pad2, bot), (sw - pad2 - L, bot)],
                  fill=color, width=w, joint="curve")
        shaft_w = max(3, int(sw * 0.13))
        s = sw * 0.17
        x_left, x_right = pad2 + L + sw * 0.02, sw - pad2 - L - sw * 0.02
        base_left, base_right = x_left + s, x_right - s
        draw.line([(base_left, cy), (base_right, cy)], fill=color, width=shaft_w)
        draw.polygon([(x_left, cy), (base_left, cy - s * 1.0), (base_left, cy + s * 1.0)], fill=color)
        draw.polygon([(x_right, cy), (base_right, cy - s * 1.0), (base_right, cy + s * 1.0)], fill=color)
    elif name == "crop":
        w, L = max(3, int(sw * 0.17)), sw * 0.34
        draw.line([(pad, pad + L), (pad, pad), (pad + L, pad)], fill=color, width=w, joint="curve")
        draw.line([(sw - pad, sw - pad - L), (sw - pad, sw - pad), (sw - pad - L, sw - pad)],
                   fill=color, width=w, joint="curve")
    elif name == "gear":
        # classic settings gear: a polygon alternating between an outer
        # (base-of-tooth) radius and a tip-of-tooth radius around the
        # circle, then a transparent circle "punched" through the center
        # -- correct since the canvas is RGBA, so erasing back to
        # (0,0,0,0) leaves a true hole regardless of what's behind the
        # icon when it's later composited, rather than needing to match
        # a specific background color. (math is imported at module
        # level -- a local "import math" here would shadow it for the
        # WHOLE function due to Python's scoping rules, breaking the
        # nested arrowhead() helper other icons rely on, even on calls
        # that never touch this branch at all.)
        n_teeth = 8
        outer_r = sw * 0.30    # base circle radius, between teeth
        tip_r = sw * 0.40      # tooth tip radius
        tooth_half = math.pi / n_teeth * 0.34
        points = []
        for i in range(n_teeth):
            base_angle = 2 * math.pi * i / n_teeth
            a0 = base_angle - tooth_half
            a1 = base_angle - tooth_half * 0.42
            a2 = base_angle + tooth_half * 0.42
            a3 = base_angle + tooth_half
            points.append((cx + outer_r * math.cos(a0), cy + outer_r * math.sin(a0)))
            points.append((cx + tip_r * math.cos(a1), cy + tip_r * math.sin(a1)))
            points.append((cx + tip_r * math.cos(a2), cy + tip_r * math.sin(a2)))
            points.append((cx + outer_r * math.cos(a3), cy + outer_r * math.sin(a3)))
        draw.polygon(points, fill=color)
        hole_r = sw * 0.135
        draw.ellipse([cx - hole_r, cy - hole_r, cx + hole_r, cy + hole_r], fill=(0, 0, 0, 0))
    elif name == "info":
        # lowercase serif "i" -- drawn geometrically (stem + serif feet +
        # dot) rather than via a font, so it doesn't depend on a specific
        # font file being present on whatever machine the app runs on
        stem_w = sw * 0.16
        stem_top, stem_bot = sw * 0.42, sw * 0.82
        foot_w, foot_h = sw * 0.30, sw * 0.045
        draw.rounded_rectangle([cx - foot_w / 2, stem_top - foot_h / 2, cx + foot_w / 2, stem_top + foot_h / 2],
                                radius=foot_h * 0.4, fill=color)
        draw.rounded_rectangle([cx - foot_w / 2, stem_bot - foot_h / 2, cx + foot_w / 2, stem_bot + foot_h / 2],
                                radius=foot_h * 0.4, fill=color)
        draw.rounded_rectangle([cx - stem_w / 2, stem_top, cx + stem_w / 2, stem_bot],
                                radius=stem_w * 0.3, fill=color)
        dot_r = sw * 0.075
        dot_cy = sw * 0.24
        draw.ellipse([cx - dot_r, dot_cy - dot_r, cx + dot_r, dot_cy + dot_r], fill=color)
    elif name in ("undo", "redo"):
        clockwise = (name == "redo")
        r = sw * 0.27
        width = max(3, int(sw * 0.15))
        bbox = [cx - r, cy - r, cx + r, cy + r]
        if not clockwise:
            start_deg, end_deg = -90, 190
            arrow_at_deg, pointing_deg = 190, 190 + 90
        else:
            start_deg, end_deg = -190, 90
            arrow_at_deg, pointing_deg = -190, -190 - 90
        draw.arc(bbox, start=start_deg, end=end_deg, fill=color, width=width)
        tri_size = sw * 0.30
        tri = Image.new("RGBA", (int(tri_size * 2), int(tri_size * 2)), (0, 0, 0, 0))
        tdraw = ImageDraw.Draw(tri)
        c = tri_size
        tdraw.polygon([(c + tri_size * 0.55, c), (c - tri_size * 0.35, c - tri_size * 0.45),
                       (c - tri_size * 0.35, c + tri_size * 0.45)], fill=color)
        tri = tri.rotate(-pointing_deg, resample=Image.BICUBIC, center=(c, c))
        ax = math.radians(arrow_at_deg)
        tip_x, tip_y = cx + r * math.cos(ax), cy + r * math.sin(ax)
        img.paste(tri, (int(tip_x - c), int(tip_y - c)), tri)
    else:
        raise ValueError(f"unknown icon name: {name}")

    return img.resize((size, size), Image.LANCZOS)


def _rounded_rect_points(w, h, r):
    r = min(r, w / 2, h / 2)
    return [r, 0, w - r, 0, w, 0, w, r, w, h - r, w, h,
            w - r, h, r, h, 0, h, 0, h - r, 0, r, 0, 0]


class RoundedEntry:
    """A ttk.Entry substitute with actual rounded, anti-aliased corners
    (drawn via PIL when available; falls back to a canvas polygon, which
    looks blockier since Tk's canvas doesn't anti-alias, if Pillow isn't
    installed). Resizes responsively via <Configure> unless a fixed
    pixel `width` is given (for small fields like a numeric entry)."""

    def __init__(self, parent, textvariable, bg, field_bg, fg, border, height=32, radius=10, width=None):
        import tkinter as tk
        self.tk = tk
        self.radius = radius
        self.bg, self.field_bg, self.border, self.fg = bg, field_bg, border, fg
        self.fixed_width = width
        self.frame = tk.Frame(parent, bg=bg)
        canvas_kwargs = {"height": height, "bg": bg, "highlightthickness": 0, "takefocus": 0}
        if width is not None:
            canvas_kwargs["width"] = width
        self.canvas = tk.Canvas(self.frame, **canvas_kwargs)
        self.canvas.pack(fill="x" if width is None else None, expand=(width is None))
        self.entry = tk.Entry(self.canvas, textvariable=textvariable, bg=field_bg, fg=fg,
                               insertbackground=fg, relief="flat", highlightthickness=0,
                               bd=0, font=("Segoe UI", 10),
                               disabledbackground=field_bg, disabledforeground=MUTED,
                               readonlybackground=field_bg)
        self._bg_photo = None
        self.canvas.bind("<Configure>", self._redraw)
        if width is not None:
            self.canvas.after(1, self._redraw)  # fixed-size canvases don't fire <Configure> reliably on all platforms

    def _redraw(self, event=None):
        w = self.fixed_width or self.canvas.winfo_width()
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

    def configure(self, **kw):
        self.entry.configure(**kw)


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
        self.canvas = tk.Canvas(self.frame, width=box, height=box, bg=bg, highlightthickness=0, takefocus=0)
        self.canvas.pack(side="left")
        self.label = tk.Label(self.frame, text=text, bg=bg, fg=fg, font=("Segoe UI", 10), justify="left")
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


class RoundedRadio:
    """A radio-button-style selector: a circular indicator (filled when
    selected) plus a text label. `is_selected_fn` (not a single shared
    Variable) decides whether THIS option is the active one -- this
    keeps it usable for boolean-backed choices (e.g. Auto/Manual, which
    is really just one BooleanVar) as well as genuine multi-value
    StringVars (e.g. Equal power/Linear), without needing two different
    widget classes. Call refresh() on every RoundedRadio in a group
    whenever any one of them is clicked, so they all update in sync."""

    def __init__(self, parent, text, is_selected_fn, on_click, bg, fg, accent, muted,
                 command=None, size=16):
        import tkinter as tk
        self.tk = tk
        self.text = text
        self.is_selected_fn = is_selected_fn
        self.on_click = on_click
        self.bg, self.fg, self.accent, self.muted = bg, fg, accent, muted
        self.size = size
        self.command = command
        self._photo = None

        self.frame = tk.Frame(parent, bg=bg)
        self.canvas = tk.Canvas(self.frame, width=size, height=size, bg=bg, highlightthickness=0, takefocus=0)
        self.canvas.pack(side="left")
        self.label = tk.Label(self.frame, text=text, bg=bg, fg=fg, font=("Segoe UI", 10))
        self.label.pack(side="left", padx=(6, 0))
        self.canvas.bind("<Button-1>", self._click)
        self.label.bind("<Button-1>", self._click)
        self.refresh()

    def _click(self, event=None):
        self.on_click()
        if self.command:
            self.command()

    def refresh(self):
        selected = self.is_selected_fn()
        self.canvas.delete("all")
        if PIL_AVAILABLE:
            img = render_radio_image(self.size, selected, self.bg, self.accent, self.muted)
            self._photo = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor="nw", image=self._photo)
        else:
            r = self.size * 0.42
            cx = cy = self.size / 2
            color = self.accent if selected else self.muted
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, outline=color, width=2)
            if selected:
                ir = r * 0.5
                self.canvas.create_oval(cx - ir, cy - ir, cx + ir, cy + ir, fill=color, outline="")
        self.label.configure(fg=self.fg if selected else self.muted)

    def pack(self, **kw):
        self.frame.pack(**kw)


class RoundedDropdown:
    """A ttk.Combobox substitute styled to match the app's dark theme.
    ttk.Combobox's popped-down list is a native platform listbox that
    ttk.Style can't fully restyle on most platforms -- hence the white
    background/hard edges. This draws its own small popup instead."""

    def __init__(self, parent, textvariable, values, bg, field_bg, fg, border, accent,
                 height=32, radius=10, width=140):
        import tkinter as tk
        self.tk = tk
        self.variable = textvariable
        self.values = values
        self.bg, self.field_bg, self.fg, self.border, self.accent = bg, field_bg, fg, border, accent
        self.radius = radius
        self.fixed_width = width
        self.height = height
        self.popup = None

        self.frame = tk.Frame(parent, bg=bg)
        self.canvas = tk.Canvas(self.frame, width=width, height=height, bg=bg, highlightthickness=0, takefocus=0)
        self.canvas.pack()
        self._bg_photo = None
        self.canvas.bind("<Button-1>", self._toggle_popup)
        self.variable.trace_add("write", lambda *a: self._redraw())
        self._redraw()

    def _redraw(self):
        w, h = self.fixed_width, self.height
        self.canvas.delete("all")
        if PIL_AVAILABLE:
            img = render_dropdown_bg_image(w, h, self.radius, self.field_bg, self.border, self.fg)
            self._bg_photo = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor="nw", image=self._bg_photo)
        else:
            pts = _rounded_rect_points(w, h, self.radius)
            self.canvas.create_polygon(pts, smooth=True, fill=self.field_bg, outline=self.border)
            cx, cy = w - 16, h / 2
            self.canvas.create_polygon(cx - 5, cy - 3, cx + 5, cy - 3, cx, cy + 4, fill=self.fg)
        self.canvas.create_text(12, h / 2, anchor="w", text=self.variable.get(),
                                 fill=self.fg, font=("Segoe UI", 10))

    def _toggle_popup(self, event=None):
        if self.popup is not None:
            self._close_popup()
            return
        tk = self.tk
        toplevel = self.canvas.winfo_toplevel()
        # Renders as an ordinary Frame placed WITHIN the main window's own
        # widget tree, not a separate floating Toplevel window. Confirmed
        # (by moving the main window and watching a fully-functional menu
        # appear, detached, behind it) that the previous overrideredirect
        # Toplevel approach was reliably being created and positioned
        # correctly -- it just wasn't staying above the main window in
        # macOS's cross-window stacking order, no matter what combination
        # of -topmost/focus_force/lift/deferred-timing was tried. Since
        # this is now a plain Frame inside the SAME window as everything
        # else, there's no cross-window z-order for macOS's window server
        # to get wrong -- .lift() here is Tk's own internal sibling
        # stacking within one window, a fundamentally more reliable
        # operation than raising one whole window above another.
        toplevel.update_idletasks()
        x = self.canvas.winfo_rootx() - toplevel.winfo_rootx()
        y = self.canvas.winfo_rooty() - toplevel.winfo_rooty() + self.canvas.winfo_height()

        row_h = 30
        self.popup = tk.Frame(toplevel, bg=self.field_bg, highlightthickness=1,
                               highlightbackground=self.border)
        self.popup.place(x=x, y=y, width=self.fixed_width, height=len(self.values) * row_h)
        for val in self.values:
            row = tk.Label(self.popup, text=val, bg=self.field_bg, fg=self.fg, anchor="w",
                            font=("Segoe UI", 10), padx=12, pady=6)
            row.pack(fill="x")
            row.bind("<Enter>", lambda e, r=row: r.configure(bg=self.accent))
            row.bind("<Leave>", lambda e, r=row: r.configure(bg=self.field_bg))
            row.bind("<Button-1>", lambda e, v=val: self._select(v))
        self.popup.lift()

        # Click-outside-to-close: safe to bind directly on the owning
        # toplevel now (not bind_all), since it's cleanly unbound the
        # moment the popup closes either way -- no risk to other,
        # unrelated app-wide bindings.
        self._outside_click_id = toplevel.bind(
            "<Button-1>", self._on_click_outside, add="+")

    def _on_click_outside(self, event):
        if self.popup is None:
            return
        # ignore the click that's opening/closing the dropdown itself --
        # its own <Button-1> handler (_toggle_popup) manages that case
        if event.widget is self.canvas:
            return
        px, py = self.popup.winfo_rootx(), self.popup.winfo_rooty()
        pw, ph = self.popup.winfo_width(), self.popup.winfo_height()
        if px <= event.x_root <= px + pw and py <= event.y_root <= py + ph:
            return  # inside the popup -- its own row bindings handle selection
        self._close_popup()

    def _select(self, val):
        self.variable.set(val)
        self._close_popup()

    def _close_popup(self):
        if self.popup is not None:
            try:
                self.popup.destroy()
            except Exception:
                pass
            self.popup = None
        outside_id = getattr(self, "_outside_click_id", None)
        if outside_id is not None:
            try:
                self.canvas.winfo_toplevel().unbind("<Button-1>", outside_id)
            except Exception:
                pass
            self._outside_click_id = None

    def pack(self, **kw):
        self.frame.pack(**kw)


class RoundedSlider:
    """A ttk.Scale substitute styled to match the app's dark theme.
    ttk.Scale's trough/thumb are largely native-theme-drawn (especially on
    Windows) and don't reliably respect ttk.Style overrides -- the same
    class of problem RoundedDropdown solves for Combobox. Integer-only
    steps (used for the MP3 VBR quality 0-9 scale)."""

    def __init__(self, parent, variable, from_, to, bg, track_hex, fill_hex, thumb_hex,
                 width=140, height=20, command=None):
        import tkinter as tk
        self.tk = tk
        self.variable, self.from_, self.to = variable, from_, to
        self.bg, self.track_hex, self.fill_hex, self.thumb_hex = bg, track_hex, fill_hex, thumb_hex
        self.width, self.height = width, height
        self.command = command

        self.frame = tk.Frame(parent, bg=bg)
        self.canvas = tk.Canvas(self.frame, width=width, height=height, bg=bg, highlightthickness=0, takefocus=0)
        self.canvas.pack()
        self._bg_photo = None
        self.canvas.bind("<Button-1>", self._on_interact)
        self.canvas.bind("<B1-Motion>", self._on_interact)
        self._redraw()

    def _value_to_x(self, value):
        frac = (value - self.from_) / max(1e-9, (self.to - self.from_))
        pad = self.height / 2
        return pad + frac * (self.width - 2 * pad)

    def _x_to_value(self, x):
        pad = self.height / 2
        frac = (x - pad) / max(1e-9, (self.width - 2 * pad))
        frac = max(0.0, min(1.0, frac))
        return round(self.from_ + frac * (self.to - self.from_))

    def _on_interact(self, event):
        value = self._x_to_value(event.x)
        if value != self.variable.get():
            self.variable.set(value)
            self._redraw()
            if self.command:
                self.command(value)
        else:
            self._redraw()

    def _redraw(self):
        w, h = self.width, self.height
        self.canvas.delete("all")
        value = self.variable.get()
        thumb_x = self._value_to_x(value)
        track_r = h * 0.18
        if PIL_AVAILABLE:
            img = render_slider_image(w, h, track_r, thumb_x, self.bg,
                                       self.track_hex, self.fill_hex, self.thumb_hex)
            self._bg_photo = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor="nw", image=self._bg_photo)
        else:
            cy = h / 2
            self.canvas.create_line(h / 2, cy, w - h / 2, cy, fill=self.track_hex, width=max(2, int(h * 0.3)))
            self.canvas.create_line(h / 2, cy, thumb_x, cy, fill=self.fill_hex, width=max(2, int(h * 0.3)))
            r = h * 0.4
            self.canvas.create_oval(thumb_x - r, cy - r, thumb_x + r, cy + r, fill=self.thumb_hex, outline="")

    def pack(self, **kw):
        self.frame.pack(**kw)


class ToolTip:
    """A small delayed hover tooltip for any Tk/ttk widget. Works for
    text-only icon buttons (where there's no visible label to explain
    what they do) as well as ordinary fields/buttons."""

    enabled = True  # class-level: toggling this affects every ToolTip
                     # instance at once, app-wide (see the "Show hover
                     # tooltips" checkbox in Hints & Keyboard Shortcuts)
    _all_instances = []  # so set_enabled(False) can immediately hide any
                          # tooltip that's already showing, not just
                          # prevent future ones from appearing

    def __init__(self, widget, text=None, delay=500):
        import tkinter as tk
        self.tk = tk
        self.widget = widget
        self.text = text
        self.rich = None
        self.section = None
        self.delay = delay
        self.tip = None
        self.after_id = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")
        ToolTip._all_instances.append(self)

    @classmethod
    def set_enabled(cls, value):
        cls.enabled = value
        if not value:
            for inst in cls._all_instances:
                inst._hide()

    def _schedule(self, event=None):
        if not ToolTip.enabled:
            return
        self._cancel()
        self.after_id = self.widget.after(self.delay, self._show)

    def _cancel(self):
        if self.after_id is not None:
            try:
                self.widget.after_cancel(self.after_id)
            except Exception:
                pass
            self.after_id = None

    def set_text(self, text):
        """Updates the tooltip's text in place -- used when a keyboard
        shortcut is remapped, so an already-built tooltip reflects the
        new binding without needing to be recreated."""
        self.text = text
        self.rich = None
        self.section = None

    def set_section(self, title, description, bullets):
        """Section-level tooltip: bold TITLE, a regular-face overview
        description, a separator, then a bulleted list of the section's
        individual sub-features. Used so hovering ANYWHERE within a
        multi-control section (e.g. XFADE OVERLAP's Manual/Auto radios)
        shows one comprehensive tooltip covering the whole area, instead
        of a different fragment depending on which exact control you're
        over -- attach the SAME ToolTip content to every widget in the
        section for that to actually work on hover."""
        self.section = (title, description, list(bullets))
        self.text = None
        self.rich = None

    def set_rich(self, name, key, description):
        """Switches this tooltip to the richer transport-button style:
        bold NAME, a simulated-keycap badge for the shortcut, a
        separator, then the description -- instead of one plain line."""
        self.rich = (name, key, description)
        self.text = None
        self.section = None

    def _show(self):
        if self.tip is not None or (not self.text and not self.rich and not self.section):
            return
        try:
            anchor_x = self.widget.winfo_rootx() + 12
            anchor_y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
            screen_w = self.widget.winfo_screenwidth()
            screen_h = self.widget.winfo_screenheight()
        except Exception:
            return
        tk = self.tk
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        try:
            self.tip.wm_attributes("-topmost", True)
        except Exception:
            pass
        self.tip.wm_geometry(f"+{anchor_x}+{anchor_y}")

        if self.rich:
            name, key, description = self.rich
            frame = tk.Frame(self.tip, bg=TOOLTIP_BG, relief="solid", borderwidth=1,
                              highlightbackground=TOOLTIP_BORDER)
            frame.pack()
            header = tk.Frame(frame, bg=TOOLTIP_BG)
            header.pack(fill="x", padx=9, pady=(7, 5))
            tk.Label(header, text=name.upper(), bg=TOOLTIP_BG, fg=FG,
                     font=("Segoe UI", 9, "bold")).pack(side="left")
            if key:
                self._render_keycap(header, key).pack(side="left", padx=(8, 0))
            tk.Frame(frame, height=1, bg=TOOLTIP_BORDER).pack(fill="x", padx=9)
            tk.Label(frame, text=description, bg=TOOLTIP_BG, fg=MUTED,
                     font=("Segoe UI", 9), wraplength=TOOLTIP_WRAPLENGTH, justify="left").pack(
                     anchor="w", fill="x", padx=9, pady=(5, 7))
        elif self.section:
            title, description, bullets = self.section
            frame = tk.Frame(self.tip, bg=TOOLTIP_BG, relief="solid", borderwidth=1,
                              highlightbackground=TOOLTIP_BORDER)
            frame.pack()
            tk.Label(frame, text=title.upper(), bg=TOOLTIP_BG, fg=FG,
                     font=("Segoe UI", 9, "bold"), wraplength=TOOLTIP_WRAPLENGTH, justify="left").pack(
                     anchor="w", fill="x", padx=9, pady=(7, 2))
            if description:
                tk.Label(frame, text=description, bg=TOOLTIP_BG, fg=MUTED,
                         font=("Segoe UI", 9), wraplength=TOOLTIP_WRAPLENGTH, justify="left").pack(
                         anchor="w", fill="x", padx=9, pady=(0, 6))
            if bullets:
                tk.Frame(frame, height=1, bg=TOOLTIP_BORDER).pack(fill="x", padx=9)
                bullets_frame = tk.Frame(frame, bg=TOOLTIP_BG)
                bullets_frame.pack(fill="x", padx=9, pady=(6, 7))
                for bullet in bullets:
                    # True hanging indent: the bullet marker and the
                    # description text are TWO separate labels side by
                    # side, not one label with "\u2022 {bullet}" as a
                    # single string. A single label's wrapped lines
                    # always return to ITS OWN left edge (i.e. back
                    # under the bullet marker itself), which is exactly
                    # the inconsistent-looking indentation reported --
                    # wrapped continuation lines were landing under the
                    # bullet character instead of under the text that
                    # follows it. With the description in its own label
                    # positioned to the right of a fixed-width marker
                    # column, its wrapped lines naturally align with
                    # their own first line instead.
                    row = tk.Frame(bullets_frame, bg=TOOLTIP_BG)
                    row.pack(fill="x", anchor="w", pady=2)
                    # anchor="n" on the pack() call itself (not just the
                    # anchor="nw" on the Label constructor, which only
                    # controls the text's position WITHIN its own label,
                    # not where the label widget sits within the row) --
                    # without this, pack's default is to vertically
                    # CENTER each side="left" sibling within the row's
                    # full height, which for a wrapped 3-line description
                    # put the bullet marker near the SECOND line instead
                    # of aligned with the first.
                    tk.Label(row, text="\u2022", bg=TOOLTIP_BG, fg=MUTED,
                             font=("Segoe UI", 9), width=2, anchor="nw").pack(side="left", anchor="n")
                    tk.Label(row, text=bullet, bg=TOOLTIP_BG, fg=MUTED,
                             font=("Segoe UI", 9), wraplength=TOOLTIP_WRAPLENGTH - 32,
                             justify="left", anchor="w").pack(side="left", fill="x", anchor="n")

        else:
            label = tk.Label(self.tip, text=self.text, bg=TOOLTIP_BG, fg=FG,
                              font=("Segoe UI", 9), padx=8, pady=4,
                              relief="solid", borderwidth=1,
                              wraplength=TOOLTIP_WRAPLENGTH, justify="left")
            label.pack()

        # Reposition to stay fully on-screen -- this couldn't be done
        # up-front, since the tooltip's actual rendered size isn't known
        # until its content is built. Without this, a tooltip could run
        # off the right or bottom edge of the screen depending on where
        # the main window happens to be sitting when you hover something.
        try:
            self.tip.update_idletasks()
            tip_w, tip_h = self.tip.winfo_width(), self.tip.winfo_height()
            margin = 8
            final_x, final_y = anchor_x, anchor_y
            if final_x + tip_w > screen_w - margin:
                final_x = max(margin, self.widget.winfo_rootx() - tip_w - 4)  # flip to the left of the widget instead
            if final_y + tip_h > screen_h - margin:
                final_y = max(margin, self.widget.winfo_rooty() - tip_h - 6)  # flip above the widget instead
            if (final_x, final_y) != (anchor_x, anchor_y):
                self.tip.wm_geometry(f"+{final_x}+{final_y}")
        except Exception:
            pass

    def _render_keycap(self, parent, key_text):
        """A small rounded, recessed 'keycap' badge for a shortcut --
        darker than the tooltip's own background, with light text, so it
        reads as a distinct simulated key rather than plain text."""
        tk = self.tk
        try:
            import tkinter.font as tkfont
            f = tkfont.Font(family="Segoe UI", size=8, weight="bold")
            text_w = f.measure(key_text)
        except Exception:
            text_w = len(key_text) * 7
        w = max(20, text_w + 14)
        h = 18
        canvas = tk.Canvas(parent, width=w, height=h, bg=TOOLTIP_BG, highlightthickness=0, takefocus=0)
        if PIL_AVAILABLE:
            img = render_rounded_box_image(w, h, 5, KEYCAP_BG, KEYCAP_BORDER)
            photo = ImageTk.PhotoImage(img)
            canvas._keycap_photo = photo  # keep a reference or Tk garbage-collects it
            canvas.create_image(0, 0, anchor="nw", image=photo)
        else:
            canvas.create_rectangle(0, 0, w, h, fill=KEYCAP_BG, outline=KEYCAP_BORDER)
        canvas.create_text(w / 2, h / 2, text=key_text, fill=KEYCAP_FG, font=("Segoe UI", 8, "bold"))
        return canvas

    def _hide(self, event=None):
        self._cancel()
        if self.tip is not None:
            try:
                self.tip.destroy()
            except Exception:
                pass
            self.tip = None


class LoopCrossfadeGUI:
    HANDLE_HIT_PX = 6
    CLICK_SLOP_PX = 3
    ICON_SIZE = 26  # transport icons sized to roughly fill the button height

    def __init__(self):
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
        self.tk, self.filedialog, self.messagebox, self.ttk = tk, filedialog, messagebox, ttk

        self.root = TkinterDnD.Tk() if DND_AVAILABLE else tk.Tk()
        # Hidden immediately, shown only once at the very end of __init__
        # (see the matching deiconify() there) -- without this, the
        # window is visible from the instant it's created, meaning every
        # subsequent step below (building all the widgets, the box-
        # layout measurement passes that repack the CURVE/XFADE/LOOP
        # boxes multiple times, the geometry()/minsize() calls, the
        # temporary worst-case status text swap for height measurement)
        # all happened live, in full view, one after another -- read as
        # rapid, jittery window flashing/resizing at startup on Windows.
        #
        # WINDOWS ONLY, deliberately: on macOS this same withdraw/
        # deiconify sequence coincided with a NEW, worse symptom -- the
        # whole Desktop visibly blanking out and restoring at launch.
        # That's consistent with macOS's window server handling a
        # withdrawn-then-deiconified window differently than Windows
        # does (plausibly some form of Space/compositor transition
        # tied to how and when a window first becomes visible), though
        # this is the best available explanation rather than a directly
        # confirmed root cause -- Tk's macOS behavior in this area isn't
        # thoroughly documented, and this can't be tested without a real
        # Mac. Given that, the safer choice is keeping the confirmed
        # Windows fix while reverting macOS to its known-good prior
        # behavior (window visible from creation), rather than keeping
        # a fix for one platform's cosmetic issue at the cost of a
        # worse one on another.
        if not IS_MACOS:
            self.root.withdraw()
        self.root.title("FermaLoop")
        self.root.configure(bg=BG)
        # Sets the RUNNING window's own icon -- separate from and in
        # addition to PyInstaller's --icon build flag, which only sets
        # the .exe/.app file's icon resource (what Explorer/Finder show
        # for the file itself, before it's running). Windows in
        # particular always uses the running window's own icon for the
        # taskbar, falling back to Tk's built-in default ("feather")
        # icon if nothing is set here, regardless of what --icon was
        # given at build time.
        #
        # Windows and macOS need GENUINELY DIFFERENT calls here, not
        # just different image sizes -- an earlier version of this code
        # learned that the hard way. Windows correctly matches each
        # provided image to its own context (16px titlebar, 32px
        # taskbar, 48px Alt-Tab) when given a list. macOS does NOT: Tk's
        # own iconphoto() documentation is explicit that on macOS only
        # the FIRST image in the list is used at all -- multi-resolution
        # matching is "outside Tk's scope" there. A first attempt at
        # fixing a blurry-while-running macOS Dock icon added a large
        # image to the END of the existing Windows-oriented list
        # ([16, 32, 48, 256]) -- which did nothing at all on macOS,
        # since it was still reading the 16px image and silently
        # ignoring everything after it. The Dock displays a RUNNING
        # app's icon from this same call, not from the .app bundle's
        # .icns file, which only takes over once the app quits -- so
        # without a correctly-FIRST large image here, macOS has no
        # choice but to upscale whatever small image it actually reads,
        # producing a visibly blurry Dock icon specifically while
        # running. 512px is what Tk's own documentation recommends for
        # smooth macOS Dock rendering, hence the platform split below
        # rather than a shared list.
        try:
            windowingsystem = self.root.tk.call("tk", "windowingsystem")
        except Exception:
            windowingsystem = ""
        try:
            if windowingsystem == "aqua":
                self._app_icon_photos = [tk.PhotoImage(data=APP_ICON_512_PNG_B64)]
            else:
                self._app_icon_photos = [
                    tk.PhotoImage(data=APP_ICON_16_PNG_B64),
                    tk.PhotoImage(data=APP_ICON_32_PNG_B64),
                    tk.PhotoImage(data=APP_ICON_48_PNG_B64),
                ]
            self.root.iconphoto(True, *self._app_icon_photos)
        except Exception:
            pass
        self.window_sizes = load_window_sizes()
        # tooltip on/off preference lives in the same JSON as window
        # sizes/positions rather than a dedicated file, since it's a
        # single small boolean and this file is already loaded/saved at
        # startup/shutdown anyway
        ToolTip.enabled = bool(self.window_sizes.get("tooltips_enabled", True))
        self.tooltips_enabled_var = tk.BooleanVar(value=ToolTip.enabled)

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
        self._canvas_resize_after_id = None  # debounce state for the waveform's
                                               # own resize handler, see
                                               # _on_canvas_resize below
        self._was_playing_last_poll = False  # lets _poll_playhead reliably detect
                                              # playback ending ON ITS OWN (e.g. RAW/
                                              # non-looped playback reaching the end of
                                              # the file) -- see _poll_playhead itself
        self.drag_mode = None      # None | "start" | "end" | "new" | "pending"
        self.drag_anchor_x = None
        self.pre_drag_selection = None
        self.undo_stack = []
        self.redo_stack = []
        self._preview_mode = False  # backing field for the preview_mode property below;
                                     # True while the player holds a processed preview, not
                                     # raw audio. The property setter is now a plain
                                     # passthrough (see its own comment for why), so this
                                     # direct assignment isn't strictly required anymore --
                                     # left as-is since it's still correct and harmless.
        self.export_mode = "raw"   # "loop" / "repeat" / "raw" -- the SOLE source of truth
                                    # for what Process & Save actually does, the button
                                    # label, and the filename suffix. Deliberately separate
                                    # from preview_mode/repeat_var: those track LIVE
                                    # playback state and can legitimately change for
                                    # reasons that have nothing to do with the user's
                                    # intended export mode (Crop and PaulXStretch both
                                    # invalidate preview_mode, since they change the
                                    # underlying data out from under any live preview
                                    # buffer -- but neither one means "the user changed
                                    # their mind about exporting as a LOOP"). Set ONLY by
                                    # the two toggle handlers (on_repeat_toggle,
                                    # on_loop_preview) and by load_file/unload_file --
                                    # never by anything that merely edits audio, so it
                                    # can't be reset by a call site that never had a
                                    # reason to touch it in the first place. Confirmed
                                    # this was a real, serious gap, not just cosmetic:
                                    # run_process() used to read preview_mode directly,
                                    # so Crop-after-LOOP would have silently exported
                                    # UNPROCESSED audio, not just mislabeled a filename.
        self._wave_photo = None    # keep a reference so PIL's PhotoImage isn't garbage collected
        self._last_loop_dur = 0.0       # duration/crossfade of the most recently computed
        self._last_loop_xfade_ms = 0.0  # LOOP preview -- lets a RESUMED (not just freshly
                                          # computed) preview still show accurate specifics
                                          # in its status message; see _status_for_now_playing
        self._icon_cache = {}      # keeps icon PhotoImage references alive too
        self._loop_anim_after_id = None
        self._loop_anim_frames = None
        self._loop_anim_index = 0
        self._last_tooltip = None
        self._play_tooltip = None
        self.player = AudioPlayer()
        self.shortcuts = load_shortcuts()

        self.in_path_var = tk.StringVar()
        self.out_path_var = tk.StringVar()
        self._save_as_root = None          # loaded file's path minus extension AND
                                            # mode suffix -- None until a file is loaded
        self._save_as_ext = ""             # current output extension, kept in sync by
                                            # _on_format_changed as well as load()
        self._save_as_current_suffix = ""  # the suffix _update_save_as_suffix last
                                            # actually wrote, for comparison below
        self._save_as_user_customized = False  # latches True the moment the user's
                                                # own edit diverges from the above --
                                                # never auto-updated again until a new
                                                # file loads and resets it
        self._save_as_programmatic_update = False  # suppresses the trace below from
                                                     # mistaking THIS class's own writes
                                                     # (mode suffix or format/extension
                                                     # changes) for a user edit
        self.out_path_var.trace_add("write", self._on_out_path_changed)
        self.format_var = tk.StringVar(value="FLAC (Lossless)")
        self.mp3_quality_var = tk.DoubleVar(value=2)
        self.mp3_quality_label_var = tk.StringVar(value="")
        self.auto_xfade_value_var = tk.StringVar(value="")
        self.xfade_var = tk.StringVar(value="1")
        self.curve_var = tk.StringVar(value="Equal power")
        self.auto_xfade_var = tk.BooleanVar(value=False)   # OFF by default, per spec
        self.snap_var = tk.BooleanVar(value=False)
        self.window_var = tk.StringVar(value="0.25")
        self.repeat_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="")
        self.time_var = tk.StringVar(value="00:00:00.000")
        self.selection_duration_var = tk.StringVar(value="Selection: --")
        self._click_flag = None       # (x_pixel, time_str) or None
        self._click_flag_after_id = None
        self._hover_flag = None       # (x_pixel, time_str) or None -- same shape/
                                       # rendering as _click_flag (both drawn via
                                       # _draw_flag), but a separate lifecycle: shown
                                       # after a brief dwell while passively hovering,
                                       # cleared immediately on Leave or once a
                                       # click/drag begins, rather than the click
                                       # flag's own fixed 1.5s auto-clear timer
        self._hover_flag_after_id = None
        self._live_update_after_id = None
        self._canvas_tooltip = None
        self._shortcuts_dialog = None
        self._shortcuts_dialog_close_fn = None

        for var in (self.xfade_var, self.curve_var, self.auto_xfade_var,
                    self.snap_var, self.window_var):
            var.trace_add("write", self._on_param_changed)
        self.format_var.trace_add("write", self._on_format_changed)

        self._build_widgets()
        self._bind_shortcuts()
        if DND_AVAILABLE:
            self._enable_drag_and_drop()

        self._apply_saved_or_natural_size()
        # NOW show the window, for the first time (Windows only -- see
        # the matching withdraw() note above for why macOS never hides
        # it in the first place). Everything above this point has
        # happened invisibly while withdrawn, so what actually appears
        # on screen is the final, already-settled window in one shot,
        # not a rapid sequence of intermediate ones.
        if not IS_MACOS:
            self.root.deiconify()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._poll_playhead()

    # ---------------- styling ----------------

    def _build_style(self):
        style = self.ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("TLabel", background=BG, foreground=FG, font=("Segoe UI", 10))
        style.configure("Muted.TLabel", background=BG, foreground=MUTED, font=("Segoe UI", 9))
        style.configure("MutedOnPanel.TLabel", background=PANEL, foreground=MUTED, font=("Segoe UI", 9))
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
        style.configure("AuditionOn.TButton", background=AUDITION_ON, foreground="#ffffff",
                         borderwidth=0, padding=10, font=("Segoe UI", 11, "bold"))
        style.map("AuditionOn.TButton", background=[("active", AUDITION_ON_HOVER)])
        style.configure("Icon.TButton", background=PANEL, foreground=FG, borderwidth=0, padding=9)
        style.map("Icon.TButton", background=[("active", BORDER)])
        style.configure("IconFlash.TButton", background=ACCENT, foreground="#ffffff", borderwidth=0, padding=9)
        style.configure("IconToggleOn.TButton", background=ACCENT, foreground="#ffffff", borderwidth=0, padding=9)
        style.map("IconToggleOn.TButton", background=[("active", ACCENT_HOVER)])

    # ---------------- icon buttons / tooltips ----------------

    def _get_icon(self, name, size=None, color=FG, rotation_deg=0):
        size = size or self.ICON_SIZE
        key = (name, size, color, rotation_deg)
        if key not in self._icon_cache:
            if PIL_AVAILABLE:
                img = render_icon_image(name, size, color, rotation_deg=rotation_deg)
                self._icon_cache[key] = ImageTk.PhotoImage(img)
            else:
                self._icon_cache[key] = None
        return self._icon_cache[key]

    def _make_rounded_section(self, parent, fill_color, border_color, radius=12, padding=12):
        """Returns (outer, inner). Build content into `inner` with normal
        pack/grid, then call _finalize_rounded_section(outer) once -- it
        measures the content's natural size and draws a properly-fitted
        anti-aliased rounded-rect background behind it (same 'measure
        first, draw to fit' approach as the other custom widgets)."""
        tk = self.tk
        outer = tk.Frame(parent, bg=BG)
        canvas = tk.Canvas(outer, bg=BG, highlightthickness=0)
        canvas.pack(fill="both", expand=True)
        inner = tk.Frame(canvas, bg=fill_color)
        outer._rc = {"canvas": canvas, "inner": inner, "fill": fill_color,
                      "border": border_color, "radius": radius, "padding": padding}
        return outer, inner

    def _finalize_rounded_section(self, outer):
        rc = outer._rc
        canvas, inner, padding = rc["canvas"], rc["inner"], rc["padding"]
        inner.update_idletasks()
        w = inner.winfo_reqwidth() + padding * 2
        h = inner.winfo_reqheight() + padding * 2
        canvas.configure(width=w, height=h)
        if PIL_AVAILABLE:
            img = render_rounded_box_image(w, h, rc["radius"], rc["fill"], rc["border"])
            photo = ImageTk.PhotoImage(img)
            canvas._bg_photo = photo  # keep a reference or Tk garbage-collects it
            canvas.create_image(0, 0, anchor="nw", image=photo)
        canvas.create_window(padding, padding, anchor="nw", window=inner)

    def _finalize_responsive_section(self, outer, forced_height=None):
        """Like _finalize_rounded_section, but the box stretches
        horizontally to fill whatever space its parent gives it (matching
        how Input/Save-as and Process & Save already behave) instead of
        staying a fixed size -- redraws the rounded background on every
        resize. `forced_height` lets a row of these share one common
        height instead of each sizing independently to its own content.

        Height is stored in rc (mutable) rather than captured by value in
        the redraw() closure, so _set_box_height() can change it later
        (e.g. switching between a shared row height and the box's own
        natural height as the layout switches between side-by-side and
        stacked) without re-binding <Configure> or recreating the content
        window each time -- doing that on every layout-mode switch would
        stack up duplicate bindings."""
        rc = outer._rc
        canvas, inner, padding = rc["canvas"], rc["inner"], rc["padding"]
        inner.update_idletasks()
        natural_w = inner.winfo_reqwidth() + padding * 2
        natural_h = inner.winfo_reqheight() + padding * 2
        rc["natural_w"], rc["natural_h"] = natural_w, natural_h
        rc["current_height"] = forced_height if forced_height is not None else natural_h
        canvas.configure(height=rc["current_height"])
        canvas.create_window(padding, padding, anchor="nw", window=inner, tags="content")

        def redraw(event=None):
            w = max(rc["natural_w"], canvas.winfo_width())
            h = rc["current_height"]
            if PIL_AVAILABLE:
                img = render_rounded_box_image(w, h, rc["radius"], rc["fill"], rc["border"])
                photo = ImageTk.PhotoImage(img)
                canvas._bg_photo = photo  # keep a reference or Tk garbage-collects it
                canvas.delete("bg")
                canvas.create_image(0, 0, anchor="nw", image=photo, tags="bg")
                canvas.tag_lower("bg")  # keep it behind the actual content

        rc["redraw"] = redraw
        canvas.bind("<Configure>", redraw)
        redraw()
        return natural_w, natural_h

    def _set_box_height(self, outer, new_height):
        """Changes a previously-finalized responsive box's height without
        re-binding <Configure> or recreating its content window."""
        rc = outer._rc
        rc["current_height"] = new_height
        rc["canvas"].configure(height=new_height)
        rc["redraw"]()

    def _on_auto_detect_clicked(self):
        self.auto_xfade_var.set(True)
        self._refresh_xfade_box()
        self._on_param_changed()

    def _on_manual_clicked(self):
        self.auto_xfade_var.set(False)
        self._refresh_xfade_box()
        self._on_param_changed()

    def _build_xfade_box(self, parent, header_label=None):
        """Manual above Auto (Manual is the more commonly used option, and
        defaults to selected). Both share one grid so the field/reference-
        value column lines up regardless of which row is active. The
        field-purpose labels ("Crossfade:") live in tooltips instead of
        on-screen text, per the "hover-over help instead of cluttering
        the interface with text" direction.

        Builds into its OWN dedicated content frame (packed into `parent`)
        rather than gridding directly into `parent` -- `parent` already
        has the section header's label/divider managed by pack(), and Tk
        forbids mixing pack and grid as siblings under the same parent."""
        tk = self.tk
        content = tk.Frame(parent, bg=PANEL)
        content.pack(fill="x")

        manual_radio = RoundedRadio(content, "Manual", lambda: not self.auto_xfade_var.get(),
                                     self._on_manual_clicked, PANEL, FG, ACCENT, MUTED)
        manual_radio.frame.grid(row=0, column=0, sticky="w", pady=(0, 2))

        self.xfade_entry = RoundedEntry(content, self.xfade_var, PANEL, FIELD_BG, FG, BORDER,
                                         height=26, radius=7, width=60)
        self.xfade_entry.frame.grid(row=0, column=1, sticky="w", padx=(10, 0), pady=(0, 2))
        self._defocus_on_return(self.xfade_entry.entry)

        auto_radio = RoundedRadio(content, "Auto", lambda: self.auto_xfade_var.get(),
                                   self._on_auto_detect_clicked, PANEL, FG, ACCENT, MUTED)
        auto_radio.frame.grid(row=1, column=0, sticky="w")

        xfade_section_desc = "Sets how much of the selection's head and tail blend together at the loop seam"
        xfade_section_bullets = ["Manual: crossfade duration in seconds",
                                  "Auto: automatically picks the crossfade length that best "
                                  "matches the head and tail of the selection, instead of a fixed value"]
        widgets_for_tip = [manual_radio.frame, self.xfade_entry.frame, auto_radio.frame]
        if header_label is not None:
            widgets_for_tip.append(header_label)
        for widget in widgets_for_tip:
            tip = ToolTip(widget)
            tip.set_section("XFADE OVERLAP", xfade_section_desc, xfade_section_bullets)

        self.auto_value_label = tk.Label(content, textvariable=self.auto_xfade_value_var,
                                          bg=PANEL, fg=MUTED, font=("Segoe UI", 9))

        self._xfade_radios = [manual_radio, auto_radio]
        self._refresh_xfade_box()

    def _refresh_xfade_box(self):
        auto_on = self.auto_xfade_var.get()
        for r in self._xfade_radios:
            r.refresh()
        self.xfade_entry.configure(state="disabled" if auto_on else "normal")
        if auto_on:
            self.auto_value_label.grid(row=1, column=1, sticky="w", padx=(10, 0))
        else:
            self.auto_value_label.grid_remove()
        self._update_auto_crossfade_preview()

    def _defocus_on_return(self, entry_widget):
        """Numeric entry fields (crossfade, search window, etc.) don't lose
        keyboard focus on their own after Return -- Tkinter Entry widgets
        just don't do that by default. Without this, typing a value and
        pressing Enter left the field focused, so a SUBSEQUENT press of
        Space (meant as the global Play/Pause shortcut) typed a literal
        space character into the field instead of toggling playback.

        Defocuses to the entry's OWN owning toplevel (winfo_toplevel()),
        not always self.root -- this same method is shared by entries
        that live in other dialogs (e.g. PaulXStretch's factor/window
        fields), where sending focus to the main window instead of back
        to the dialog itself was the actual cause of two related, but
        platform-different, bugs: on Windows, the modal dialog's own
        grab_set() silently blocked focus from ever leaving for another
        toplevel at all, so Return appeared to do nothing; on macOS the
        focus_set() call succeeded, but the dialog (still transient/
        grabbed) stayed visually on top while keyboard focus had
        actually moved to the window underneath it -- an actual
        Enter-behaves-differently-per-platform bug, not a lookalike.
        For every entry that already lived in the main window, its own
        winfo_toplevel() IS self.root, so this is a strict superset of
        the old behavior, not a change to it, for every existing caller
        except the two that were actually broken.

        Returns "break" so this Return doesn't ALSO immediately trigger
        a toplevel-level "Return activates the default button" binding
        (see open_stretch_dialog) in the same keypress -- defocusing and
        then activating the default action are meant to be two separate
        presses, not one."""
        def _defocus(e=None):
            entry_widget.winfo_toplevel().focus_set()
            return "break"
        entry_widget.bind("<Return>", _defocus)
        entry_widget.bind("<KP_Enter>", _defocus)

    def _transport_tooltip(self, action_name):
        """Looks up the display name/description from TRANSPORT_HINTS and
        the CURRENT shortcut (respecting remaps) for action_name, and
        returns (display_name, key_display, description) -- the three
        pieces ToolTip.set_rich() renders as a bold name, a simulated
        keycap badge, and the description underneath."""
        display_name, description = TRANSPORT_HINTS[action_name]
        key = self.shortcuts.get(action_name, DEFAULT_SHORTCUTS.get(action_name, ""))
        key_display = format_key_for_display(key) if key else ""
        return (display_name, key_display, description)

    def _make_icon_button(self, parent, icon_name, tooltip_text, command, size=None, style="Icon.TButton"):
        """Icon-only button with a hover tooltip -- falls back to a short
        text label if Pillow isn't installed, so the button never ends up
        blank. takefocus=0 is important here: a ttk.Button that currently
        holds keyboard focus (which it gets automatically on click) has
        its OWN built-in binding that invokes it on <space> -- separate
        from and in addition to the app's global <space> shortcut. Without
        this, clicking Play then pressing Space called on_play_pause()
        TWICE for one keypress (once from the button's own focus-triggered
        invoke, once from the global shortcut), which looked like a brief
        stop-and-restart or an unexpected jump.

        tooltip_text may be a plain string (simple one-line hint) or a
        (name, key, description) tuple, in which case the tooltip renders
        in the richer transport-button style via ToolTip.set_rich()."""
        is_rich = isinstance(tooltip_text, tuple)
        fallback_text = tooltip_text[0] if is_rich else tooltip_text
        photo = self._get_icon(icon_name, size)
        if photo is not None:
            btn = self.ttk.Button(parent, image=photo, style=style, command=command, takefocus=0)
        else:
            btn = self.ttk.Button(parent, text=fallback_text, style=style, command=command, takefocus=0)
        if is_rich:
            tip = ToolTip(btn)
            tip.set_rich(*tooltip_text)
            self._last_tooltip = tip
        else:
            self._last_tooltip = ToolTip(btn, tooltip_text)
        return btn

    def _set_play_pause_icon(self, playing):
        self.btn_play.configure(image=self._get_icon("pause" if playing else "play"))
        if self._play_tooltip is not None:
            self._play_tooltip.text = "Pause" if playing else "Play"

    # ---------------- Loop (animated crossfade-preview) / Repeat icons ----------------

    def _get_loop_animation_frames(self, n_frames=12):
        key = ("_loop_anim_frames", self.ICON_SIZE, n_frames)
        if key not in self._icon_cache:
            frames = [self._get_icon("loop", self.ICON_SIZE, ACCENT, rotation_deg=i * (360 / n_frames))
                      for i in range(n_frames)]
            self._icon_cache[key] = frames
        return self._icon_cache[key]

    def _refresh_loop_icon(self):
        """Three distinct visual states for the Loop button, not two:
        grey/static (LOOP isn't the current export mode), accent/static
        (LOOP IS the export mode, but nothing is actively producing that
        audio right now -- e.g. right after Crop, or simply armed but
        not yet started), and accent/spinning (LOOP is both the export
        mode AND genuinely playing right now). Confirmed directly: after
        Crop, the icon should stay accent-colored (mode still selected)
        but stop spinning (audio genuinely isn't playing anymore),
        rather than reverting all the way to plain grey the way it did
        when this was driven by preview_mode alone -- preview_mode
        tracks whether the player is HOLDING a live buffer, export_mode
        tracks what the user actually chose, and those aren't always
        the same thing anymore (see export_mode's own comment)."""
        is_spinning = self.preview_mode and self.player.playing
        if is_spinning:
            if self._loop_anim_after_id is None:
                self._loop_anim_frames = self._get_loop_animation_frames()
                self._loop_anim_index = 0
                self._animate_loop_icon()
            return
        if self._loop_anim_after_id is not None:
            try:
                self.root.after_cancel(self._loop_anim_after_id)
            except Exception:
                pass
            self._loop_anim_after_id = None
        color = ACCENT if self.export_mode == "loop" else FG
        icon = self._get_icon("loop", self.ICON_SIZE, color)
        if icon is not None:
            self.btn_loop.configure(image=icon)

    def _animate_loop_icon(self):
        if not (self.preview_mode and self.player.playing) or self._loop_anim_frames is None:
            self._loop_anim_after_id = None
            return
        self.btn_loop.configure(image=self._loop_anim_frames[self._loop_anim_index])
        self._loop_anim_index = (self._loop_anim_index + 1) % len(self._loop_anim_frames)
        self._loop_anim_after_id = self.root.after(90, self._animate_loop_icon)

    def _refresh_repeat_icon(self):
        """Repeat's displayed color reflects export_mode now, not
        preview_mode/repeat_var directly -- accent when REPEAT is the
        chosen export mode, grey otherwise (including while LOOP is
        chosen instead, the same "suppressed in favor of LOOP" behavior
        as before, just keyed off the more durable value)."""
        color = ACCENT if self.export_mode == "repeat" else FG
        icon = self._get_icon("repeat", self.ICON_SIZE, color)
        if icon is not None:
            self.btn_repeat.configure(image=icon)

    # ---------------- widget layout ----------------

    def _build_widgets(self):
        tk, ttk = self.tk, self.ttk
        outer = ttk.Frame(self.root, padding=16)
        outer.pack(fill="both", expand=True)
        self._outer_frame = outer

        header_row = ttk.Frame(outer); header_row.pack(fill="x", pady=(0, 10))
        # Falls back to the plain text label if the embedded logo can't
        # be loaded for any reason (corrupted data, an unusually old Tk
        # build without PNG PhotoImage support, etc.) -- the header
        # should never end up blank.
        self._logo_photo = None
        try:
            self._logo_photo = tk.PhotoImage(data=LOGO_PNG_B64)
        except Exception:
            self._logo_photo = None
        if self._logo_photo is not None:
            tk.Label(header_row, image=self._logo_photo, bg=BG, bd=0,
                     highlightthickness=0).pack(side="left")
        else:
            ttk.Label(header_row, text="FermaLoop", style="Heading.TLabel").pack(side="left")
        btn_gear = self._make_icon_button(header_row, "gear", "Preferences and Help",
                                           self.open_shortcuts_dialog, size=20)
        btn_gear.pack(side="right")

        # file row (rounded entries)
        row = ttk.Frame(outer); row.pack(fill="x", pady=3)
        ttk.Label(row, text="Input", width=7).pack(side="left")
        in_entry = RoundedEntry(row, self.in_path_var, BG, FIELD_BG, FG, BORDER)
        in_entry.pack(side="left", fill="x", expand=True, padx=6)
        ToolTip(in_entry.frame, "Path to the audio file to load")
        self._defocus_on_return(in_entry.entry)
        btn_browse_in = ttk.Button(row, text="Browse", command=self.choose_input, takefocus=0)
        btn_browse_in.pack(side="left")
        ToolTip(btn_browse_in, "Choose an audio file from disk")

        row = ttk.Frame(outer); row.pack(fill="x", pady=3)
        ttk.Label(row, text="Save as", width=7).pack(side="left")
        out_entry = RoundedEntry(row, self.out_path_var, BG, FIELD_BG, FG, BORDER)
        out_entry.pack(side="left", fill="x", expand=True, padx=6)
        ToolTip(out_entry.frame, "Where the processed loop will be saved")
        self._defocus_on_return(out_entry.entry)
        btn_browse_out = ttk.Button(row, text="Browse", command=self.choose_output, takefocus=0)
        btn_browse_out.pack(side="left")
        ToolTip(btn_browse_out, "Choose where to save the processed file")

        row = ttk.Frame(outer); row.pack(fill="x", pady=3)
        ttk.Label(row, text="Format", width=7).pack(side="left")
        format_dropdown = RoundedDropdown(row, self.format_var, FORMAT_OPTIONS,
                                           BG, FIELD_BG, FG, BORDER, ACCENT, height=28, radius=8, width=180)
        format_dropdown.pack(side="left", padx=(6, 10))
        ToolTip(format_dropdown.frame, "FLAC and MP4 (Apple Lossless) are both lossless; "
                                        "MP3 uses variable bitrate at the quality set below")

        self.mp3_quality_row = ttk.Frame(row)
        mp3_scale = RoundedSlider(self.mp3_quality_row, self.mp3_quality_var, 0, 9,
                                   BG, FIELD_BG, ACCENT, FG, width=140, height=20,
                                   command=self._on_mp3_quality_change)
        mp3_scale.pack(side="left")
        ttk.Label(self.mp3_quality_row, textvariable=self.mp3_quality_label_var,
                  style="Muted.TLabel").pack(side="left", padx=(8, 0))
        ToolTip(self.mp3_quality_row, "MP3 VBR quality: left = smaller file/lower quality, "
                                       "right = larger file/higher quality")
        self._on_mp3_quality_change()
        self._on_format_changed()

        # timeline ruler (shared coordinate space with the waveform below)
        self.timeline_canvas = tk.Canvas(outer, height=22, bg=BG, highlightthickness=0)
        self.timeline_canvas.pack(fill="x", pady=(6, 0))
        self.timeline_canvas.bind("<Configure>", lambda e: self._redraw())

        # waveform canvas
        canvas_frame = ttk.Frame(outer, style="Panel.TFrame")
        canvas_frame.pack(fill="x", pady=(0, 4))
        self.canvas = tk.Canvas(canvas_frame, height=self.canvas_height, bg=PANEL,
                                 highlightthickness=0, takefocus=0)
        self.canvas.pack(fill="x", expand=True, padx=4, pady=4)
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self.canvas.bind("<Motion>", self._on_canvas_motion)
        self.canvas.bind("<Leave>", self._on_canvas_leave)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)     # Windows / macOS
        self.canvas.bind("<Button-4>", self._on_mousewheel)       # Linux scroll up
        self.canvas.bind("<Button-5>", self._on_mousewheel)       # Linux scroll down
        self._draw_placeholder()

        # timer + selection duration readouts
        info_row = ttk.Frame(outer); info_row.pack(fill="x", pady=(2, 6))
        ttk.Label(info_row, textvariable=self.time_var, style="Muted.TLabel",
                  font=("Segoe UI", 22, "bold")).pack(side="left")
        ttk.Label(info_row, textvariable=self.selection_duration_var, style="Muted.TLabel").pack(side="right")

        # transport
        transport = ttk.Frame(outer); transport.pack(fill="x", pady=(4, 4))
        transport_icons = ttk.Frame(transport)
        transport_icons.pack(expand=True)  # centers the icon group horizontally

        self.btn_play = self._make_icon_button(transport_icons, "play",
            self._transport_tooltip("play_pause"), self.on_play_pause, size=self.ICON_SIZE)
        self.btn_play.pack(side="left", padx=(0, 4))
        self._play_tooltip = self._last_tooltip

        self.btn_stop = self._make_icon_button(transport_icons, "stop",
            self._transport_tooltip("stop"), self.on_stop, size=self.ICON_SIZE)
        self.btn_stop.pack(side="left", padx=4)
        self._stop_tooltip = self._last_tooltip

        self.btn_repeat = self._make_icon_button(transport_icons, "repeat",
            self._transport_tooltip("loop_toggle"), self.on_repeat_toggle, size=self.ICON_SIZE)
        self.btn_repeat.pack(side="left", padx=4)
        self._repeat_tooltip = self._last_tooltip

        self.btn_loop = self._make_icon_button(transport_icons, "loop",
            self._transport_tooltip("audition"), self.on_loop_preview, size=self.ICON_SIZE)
        self.btn_loop.pack(side="left", padx=(16, 4))
        self._loop_tooltip = self._last_tooltip

        self.btn_crop = self._make_icon_button(transport_icons, "crop",
            self._transport_tooltip("crop"), self.on_crop, size=self.ICON_SIZE)
        self.btn_crop.pack(side="left", padx=4)
        self._crop_tooltip = self._last_tooltip

        btn_stretch = self._make_icon_button(transport_icons, "stretch",
            self._transport_tooltip("stretch"), self.open_stretch_dialog, size=self.ICON_SIZE)
        btn_stretch.pack(side="left", padx=4)
        self._stretch_tooltip = self._last_tooltip

        # Divider removed -- the ~37px combined gap it created (14px pad
        # + the line + 14px pad + cols_row's own 4px top pad) read as too
        # much dead space between the transport controls and the boxes
        # below. A single, more modest gap replaces it -- present, but
        # tightened, not eliminated outright.
        #
        # ---- three distinct task boxes: CURVE, XFADE, LOOP -- LOOP last,
        # since it's likely the least-used of the three. Each is its own
        # small rounded box with a header, rather than one shared
        # container trying to hold four different decisions at once. ----
        #
        # All three are wrapped in an outer group container -- a subtle
        # border around the trio with a small tail pointing up at the
        # LOOP button, a speech-bubble/callout shape that visually ties
        # these settings specifically to LOOP (they affect it, and only
        # it -- REPEAT and raw playback ignore them entirely). This is
        # PURE STATIC RENDERING added directly into the existing main
        # window -- no new Toplevel, no popup, no focus/grab handling of
        # any kind -- which is deliberately what keeps this low-risk
        # across platforms: it sidesteps the entire class of macOS-
        # specific window-server issues (the Format dropdown, the
        # PaulXStretch focus bug) that came from a SEPARATE window being
        # involved. This is just one more rounded box drawn the same way
        # every other one in this app already is, at one more level of
        # nesting.
        GROUP_TAIL_H = 9
        GROUP_MARGIN = 8  # inset between the outer border and the boxes it wraps
        loop_group_canvas = tk.Canvas(outer, bg=BG, highlightthickness=0)
        loop_group_canvas.pack(fill="x", pady=(4, 8))
        cols_row = ttk.Frame(loop_group_canvas)
        cols_row_window = loop_group_canvas.create_window(
            GROUP_MARGIN, GROUP_TAIL_H + GROUP_MARGIN, anchor="nw", window=cols_row)

        def _redraw_loop_group(event=None):
            # unlike pack(fill="x"), a canvas create_window doesn't
            # automatically stretch its embedded widget -- its width has
            # to be set explicitly on every resize for cols_row (and via
            # it, the three boxes' own fill=both/expand=True packing) to
            # actually track the available space, rather than staying
            # frozen at whatever width it happened to have when created.
            canvas_w = loop_group_canvas.winfo_width()
            if canvas_w < 10:
                return  # not yet realized; the initial call right after bind() below covers this
            content_w = max(1, canvas_w - GROUP_MARGIN * 2)
            loop_group_canvas.itemconfigure(cols_row_window, width=content_w)
            # height must be measured AFTER the width change above is
            # actually applied, not before -- otherwise this reads a
            # stale height from the previous width, the class of
            # stale-measurement bug already hit more than once elsewhere
            # in this file.
            loop_group_canvas.update_idletasks()
            content_h = cols_row.winfo_reqheight()
            box_h = content_h + GROUP_MARGIN * 2

            # Tail x-position: LOOP's own horizontal center, translated
            # into this canvas's coordinate space via a root-coordinate
            # difference -- the same proven technique already used for
            # the Format dropdown's positioning, rather than anything
            # relying on pack/grid-relative offsets that can drift once
            # multiple independently-centered containers are involved.
            tail_x = (self.btn_loop.winfo_rootx() - loop_group_canvas.winfo_rootx()
                      + self.btn_loop.winfo_width() / 2)
            tail_x = max(GROUP_MARGIN + 10, min(canvas_w - GROUP_MARGIN - 10, tail_x))

            img = render_rounded_box_with_tail_image(
                # radius is inner_radius(12) + GROUP_MARGIN(8) = 20, not an
                # arbitrary bump -- that's the radius that makes this
                # curve CONCENTRIC with the child boxes' own 12px corners,
                # offset outward by exactly the margin between them, which
                # is what actually reads as "hugging" them rather than
                # just being a same-ish-sized curve independently applied
                # to a bigger rectangle. Previously 10 -- smaller than the
                # children's own 12, when it should be visibly larger.
                canvas_w, box_h, radius=20, fill_hex=None, border_hex=BORDER,
                tail_x=tail_x, tail_w=16, tail_h=GROUP_TAIL_H,
            )
            photo = ImageTk.PhotoImage(img)
            loop_group_canvas._bg_photo = photo  # keep a reference or Tk garbage-collects it
            loop_group_canvas.delete("bg")
            loop_group_canvas.create_image(0, 0, anchor="nw", image=photo, tags="bg")
            loop_group_canvas.tag_lower("bg")  # keep it behind the actual content
            loop_group_canvas.configure(height=box_h + GROUP_TAIL_H)
            loop_group_canvas.coords(cols_row_window, GROUP_MARGIN, GROUP_TAIL_H + GROUP_MARGIN)

        loop_group_canvas.bind("<Configure>", _redraw_loop_group)
        _redraw_loop_group()  # harmless no-op now (canvas isn't sized yet),
                               # but matches the belt-and-suspenders pattern
                               # every other responsive box in this app uses
        self._loop_group_canvas = loop_group_canvas
        self._loop_group_cols_row = cols_row
        self._redraw_loop_group = _redraw_loop_group
        self._loop_group_margin, self._loop_group_tail_h = GROUP_MARGIN, GROUP_TAIL_H

        def _section_header(parent, title):
            label = ttk.Label(parent, text=title, background=PANEL, foreground=FG,
                               font=("Segoe UI", 10, "bold"))
            label.pack(anchor="w")
            tk.Frame(parent, height=1, bg=BORDER).pack(fill="x", pady=(3, 6))
            return label

        # Build all three boxes' CONTENT first (without packing/finalizing
        # any of them yet) so their natural heights can be measured
        # together below, and all three finalized with ONE shared height.

        # CURVE
        curve_outer, curve_inner = self._make_rounded_section(cols_row, PANEL, BORDER, radius=12, padding=8)
        curve_header = _section_header(curve_inner, "XFADE CURVE")
        self._curve_radios = []

        def _on_curve_click(value):
            self.curve_var.set(value)
            for r in self._curve_radios:
                r.refresh()
            self._on_param_changed()

        curve_section_tip = ("Shapes how the crossfade blends the two ends together",
                              ["Equal power: smoother, constant perceived loudness through the fade",
                               "Linear: simpler ramp, can dip slightly in the middle"])
        for value in ("Equal power", "Linear"):
            radio = RoundedRadio(curve_inner, value, (lambda v=value: self.curve_var.get() == v),
                                  (lambda v=value: _on_curve_click(v)), PANEL, FG, ACCENT, MUTED)
            radio.pack(anchor="w", pady=1)
            self._curve_radios.append(radio)
            tip = ToolTip(radio.frame)
            tip.set_section("XFADE CURVE", curve_section_tip[0], curve_section_tip[1])
        header_tip = ToolTip(curve_header)
        header_tip.set_section("XFADE CURVE", curve_section_tip[0], curve_section_tip[1])

        # XFADE
        xfade_outer, xfade_inner = self._make_rounded_section(cols_row, PANEL, BORDER, radius=12, padding=8)
        xfade_header = _section_header(xfade_inner, "XFADE OVERLAP")
        self._build_xfade_box(xfade_inner, header_label=xfade_header)

        # LOOP (least-used, so it goes last)
        loop_outer, loop_inner = self._make_rounded_section(cols_row, PANEL, BORDER, radius=12, padding=8)
        loop_header = _section_header(loop_inner, "LOOP ALIGNMENT")
        snap_row = tk.Frame(loop_inner, bg=PANEL)
        snap_row.pack(anchor="w")
        snap_cb = RoundedCheckbutton(snap_row, "Snap to\ntransients",
                            self.snap_var, PANEL, FG, FIELD_BG, ACCENT, BORDER,
                            command=self._toggle_window_entry)
        snap_cb.pack(side="left")
        self.window_entry = RoundedEntry(snap_row, self.window_var, PANEL, FIELD_BG, FG, BORDER,
                                          height=26, radius=7, width=60)
        self.window_entry.pack(side="left", padx=(14, 0))
        self.window_entry.configure(state="normal" if self.snap_var.get() else "disabled")
        self._defocus_on_return(self.window_entry.entry)

        loop_section_desc = "Trims the selection to align with the strongest nearby transient at each end"
        loop_section_bullets = ["Snap to transients: trims to the strongest nearby attack at each end, "
                                 "so the loop starts/ends on the beat instead of an arbitrary sample -- "
                                 "works alongside either Auto or Manual crossfade",
                                 "Search range: how far from each end to search for a transient, in "
                                 "seconds -- auto-populated, override by typing a new value"]
        for widget in (snap_cb.frame, self.window_entry.frame, loop_header):
            tip = ToolTip(widget)
            tip.set_section("LOOP ALIGNMENT", loop_section_desc, loop_section_bullets)

        # Each "finalize" call creates the box's own content window +
        # background and binds its own resize handling.
        self._box_pairs = [(curve_outer, curve_inner), (xfade_outer, xfade_inner), (loop_outer, loop_inner)]
        self._box_side_by_side_paddings = [(0, 8), (8, 8), (8, 0)]
        self._cols_row = cols_row

        natural_widths = []
        natural_heights = []
        for box_outer, box_inner in self._box_pairs:
            nw, nh = self._finalize_responsive_section(box_outer)
            natural_widths.append(nw)
            natural_heights.append(nh)
        self._boxes_shared_height = max(natural_heights)

        # Packed side-by-side once, directly, with no runtime switching.
        # A stacked (vertical) fallback used to exist here, letting the
        # window's true minimum width be as small as a single box's
        # natural width -- deliberately restored once already this
        # session after an earlier removal turned out to be based on an
        # unverified, wrong assumption (see the transcript). This
        # removal is different: an explicit, informed choice -- the
        # collapsibility genuinely isn't wanted, on either platform, and
        # a fresh install with no saved window size yet was defaulting
        # to this narrow/stacked/tall arrangement on macOS specifically
        # (Windows only avoided it because a saved size from prior
        # testing already existed). All three boxes now always render
        # side by side; the true minimum width is simply what that needs.
        for (box_outer, _), pad in zip(self._box_pairs, self._box_side_by_side_paddings):
            box_outer.pack(side="left", fill="both", expand=True, padx=pad)
            self._set_box_height(box_outer, self._boxes_shared_height)

        btn_process = ttk.Button(outer, style="Accent.TButton",
                   command=self.run_process, takefocus=0)
        self.process_btn_var = tk.StringVar(value="Save Unprocessed")
        btn_process.configure(textvariable=self.process_btn_var)
        self._process_btn_tooltip = ToolTip(btn_process, "")
        self._refresh_process_button_label()  # export_mode is only ever set via
                                                # _set_export_mode, which this button
                                                # doesn't exist yet to be refreshed BY
                                                # at __init__ time -- this explicit call
                                                # is what gives it its real starting
                                                # label/tooltip instead of the StringVar's
                                                # hardcoded default and an empty tooltip
        btn_process.pack(fill="x", pady=(8, 8))

        # wraplength is deliberately NOT a fixed value here: Tk labels
        # with a hardcoded wraplength can reserve that much width up
        # front regardless of current text content, which was silently
        # keeping the window's true minimum width wide (700px) no matter
        # what the CURVE/XFADE/LOOP row's own stacking logic did -- an
        # entirely separate, unrelated source of the same "can't shrink"
        # symptom. Binding it to the actual available width instead
        # means it only ever wraps to fit what's really there.
        status_label = ttk.Label(outer, textvariable=self.status_var, style="Muted.TLabel",
                                  wraplength=400, justify="left")
        status_label.pack(anchor="w", fill="x")
        outer.bind("<Configure>", lambda e: status_label.configure(wraplength=max(200, e.width - 32)), add="+")

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
            notes_label = ttk.Label(outer, text=" / ".join(notes), style="Muted.TLabel",
                                     foreground="#e2a33d", wraplength=400, justify="left")
            notes_label.pack(anchor="w", pady=(8, 0))
            outer.bind("<Configure>", lambda e: notes_label.configure(wraplength=max(200, e.width - 32)), add="+")

    def _toggle_window_entry(self):
        self.window_entry.configure(state="normal" if self.snap_var.get() else "disabled")

    def _on_tooltips_toggle(self):
        ToolTip.set_enabled(self.tooltips_enabled_var.get())

    def _on_mp3_quality_change(self, value_str=None):
        q = int(round(float(self.mp3_quality_var.get())))
        self.mp3_quality_var.set(q)  # snap the slider to integer steps
        self.mp3_quality_label_var.set(f"V{q} ({MP3_QUALITY_INFO.get(q, '')})")

    def _on_format_changed(self, *args):
        fmt = self.format_var.get()
        ext = FORMAT_EXT.get(fmt, ".flac")
        current = self.out_path_var.get()
        if current:
            base, _ = os.path.splitext(current)
            # _save_as_ext is updated here too (not just out_path_var
            # itself), and _save_as_programmatic_update suppresses
            # _on_out_path_changed's user-edit detection around this
            # write -- otherwise a plain format change would itself get
            # latched as "the user manually customized the filename,"
            # permanently disabling the mode-suffix auto-update over
            # nothing more than an extension swap.
            self._save_as_ext = ext
            self._save_as_programmatic_update = True
            try:
                self.out_path_var.set(base + ext)
            finally:
                self._save_as_programmatic_update = False
        if fmt == "MP3 (VBR)":
            self.mp3_quality_row.pack(side="left")
        else:
            self.mp3_quality_row.pack_forget()

    # ---------------- file loading ----------------

    def _initial_browse_dir(self):
        """Best starting folder for the Browse dialogs: prefer the
        currently-loaded file's folder, then the current Save-as folder,
        so Browse doesn't dump you in the generic default (Documents on
        Windows) once you're already working with a file somewhere else."""
        for path in (self.loaded_path, self.out_path_var.get(), self.in_path_var.get()):
            if path:
                d = os.path.dirname(path)
                if d and os.path.isdir(d):
                    return d
        return None

    def choose_input(self):
        initial_dir = self._initial_browse_dir()
        kwargs = {"initialdir": initial_dir} if initial_dir else {}
        path = self.filedialog.askopenfilename(
            title="Choose audio file",
            filetypes=[("Audio files", "*.wav *.aif *.aiff *.mp3 *.mp4 *.m4a *.flac"), ("All files", "*.*")],
            **kwargs,
        )
        if path:
            self.load_file(path)

    def choose_output(self):
        ext = FORMAT_EXT.get(self.format_var.get(), ".flac")
        initial_dir = self._initial_browse_dir()
        kwargs = {"initialdir": initial_dir} if initial_dir else {}
        path = self.filedialog.asksaveasfilename(
            title="Save processed file as", defaultextension=ext,
            filetypes=[("FLAC", "*.flac"), ("MP4 (Apple Lossless)", "*.mp4"), ("MP3", "*.mp3"),
                       ("WAV", "*.wav"), ("AIFF", "*.aiff"), ("M4A", "*.m4a")],
            **kwargs,
        )
        if path:
            # Tk's file dialogs (built on Tcl, which uses forward slashes
            # internally on every platform) can hand back forward-slash
            # paths even on Windows -- normpath() converts to the native
            # separator (backslash on Windows) without affecting POSIX.
            path = os.path.normpath(path)
            self.out_path_var.set(path)
            # keep the Format selector in sync if the user picked a
            # different extension directly in the save dialog
            chosen_ext = os.path.splitext(path)[1].lower()
            for label, e in FORMAT_EXT.items():
                if e == chosen_ext and self.format_var.get() != label:
                    self.format_var.set(label)
                    break

    def _enable_drag_and_drop(self):
        for widget in (self.root, self.canvas):
            widget.drop_target_register(DND_FILES)
            widget.dnd_bind("<<Drop>>", self._on_drop)

    def _on_drop(self, event):
        paths = _parse_dnd_paths(event.data)
        if paths:
            self.load_file(paths[0])

    def load_file(self, path):
        path = os.path.normpath(path)  # normalize separators (see choose_output for why)
        ext = os.path.splitext(path)[1].lower()
        if ext not in SUPPORTED_EXTS:
            self.messagebox.showerror("FermaLoop", f"Unsupported file type: {ext}")
            return
        try:
            self.status_var.set("Loading...")
            self.root.update_idletasks()
            data, sr, sampwidth = decode_to_pcm(path)
        except Exception as e:
            self.status_var.set("Failed to load file.")
            self.messagebox.showerror("FermaLoop", str(e))
            return

        self.data, self.sr, self.sampwidth = data, sr, sampwidth
        self.loaded_path = path
        self.cropped = False
        self.sel_start, self.sel_end = 0, len(data)
        self.zoom_start, self.zoom_end = 0, len(data)
        self.preview_mode = False
        self.repeat_var.set(False)  # a fresh file shouldn't inherit REPEAT from
                                      # whatever was loaded before it -- same stale-
                                      # state category as the LOOP-entry fix above,
                                      # just for a different transition into "neither
                                      # mode active"
        self._set_export_mode("raw")  # a fresh file also shouldn't inherit the
                                       # PREVIOUS file's intended export mode
        self._refresh_loop_and_repeat_icons()
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.player.load(data, sr)

        self.in_path_var.set(path)
        # auto-fill Save As to the same directory as the loaded file, using
        # the currently-selected output FORMAT's extension (not necessarily
        # the input file's own extension, since only FLAC/MP4/MP3 are
        # offered as save targets). The mode suffix itself (RAW/REPEAT/
        # LOOP) is added by _update_save_as_suffix, not hardcoded here --
        # it reads current mode state, which is now guaranteed "neither"
        # at this point via the resets just above, so this always starts
        # a freshly-loaded file at " RAW".
        root_name, orig_ext = os.path.splitext(path)
        out_ext = FORMAT_EXT.get(self.format_var.get(), orig_ext)
        self._save_as_root = os.path.normpath(root_name)
        self._save_as_ext = out_ext
        self._save_as_user_customized = False
        self._update_save_as_suffix()

        self._click_flag = None
        self._cancel_hover_flag()  # defensive consistency with the click flag reset
                                    # just above -- a fresh load shouldn't carry over
                                    # any stale hover state from before
        self._redraw()
        self._update_selection_duration_label()
        self._update_auto_crossfade_preview()
        dur = len(data) / sr
        self.status_var.set(f"Loaded {os.path.basename(path)} ({dur:.2f}s). Select a region, then LOOP or REPEAT to preview, or Save directly.")

    # ---------------- undo / redo ----------------

    def _snapshot(self, label="change"):
        return {"data": self.data, "sel_start": self.sel_start, "sel_end": self.sel_end,
                "cropped": self.cropped, "zoom_start": self.zoom_start, "zoom_end": self.zoom_end,
                "label": label}

    def _restore(self, snap):
        self.data = snap["data"]
        self.sel_start, self.sel_end = snap["sel_start"], snap["sel_end"]
        self.cropped = snap["cropped"]
        self.zoom_start, self.zoom_end = snap["zoom_start"], snap["zoom_end"]
        self.preview_mode = False
        self._refresh_loop_and_repeat_icons()
        self.player.load(self.data, self.sr)
        self.player.set_selection(self.sel_start, self.sel_end)
        self._redraw()
        self._update_selection_duration_label()
        self._update_auto_crossfade_preview()

    def push_undo(self, label="change"):
        """`label` names the operation this snapshot's later restoration
        would undo (e.g. "crop", "stretch") -- used to build a specific
        undo/redo status message instead of a generic one. Call this
        BEFORE making the change, same as always; the label describes
        the change about to happen, not the state being captured."""
        self.undo_stack.append(self._snapshot(label))
        self.redo_stack.clear()

    def undo(self):
        if not self.undo_stack or self.data is None:
            return
        entry = self.undo_stack.pop()
        label = entry.get("label", "change")
        self.redo_stack.append(self._snapshot(label))
        self._restore(entry)
        self.status_var.set(f"Undid {label}.")

    def redo(self):
        if not self.redo_stack or self.data is None:
            return
        entry = self.redo_stack.pop()
        label = entry.get("label", "change")
        self.undo_stack.append(self._snapshot(label))
        self._restore(entry)
        self.status_var.set(f"Redid {label}.")

    # ---------------- waveform canvas ----------------

    def _draw_placeholder(self):
        self.canvas.delete("all")
        w = self.canvas.winfo_width() or self.canvas_width
        h = self.canvas.winfo_height() or self.canvas_height
        self.canvas.create_text(w // 2, h // 2, text="Drag & drop audio file, select a region to loop",
                                 fill=MUTED, font=("Segoe UI", 10))
        self.timeline_canvas.delete("all")

    def _on_canvas_resize(self, event):
        self.canvas_width, self.canvas_height = event.width, event.height
        # Debounced: this used to call _redraw() unconditionally, on
        # EVERY <Configure> tick during an active resize drag -- which
        # can fire many times per second -- meaning the full waveform
        # (render_waveform_image, processing the entire visible audio
        # segment) got expensively re-rendered back to back, with no
        # gaps, for as long as the drag continued. Because Python's GIL
        # means CPU-bound work on THIS (GUI) thread directly delays the
        # SEPARATE audio callback thread from running, and that callback
        # has an extremely tight budget (blocksize=256 at 44.1kHz is
        # roughly 5.8ms per invocation -- see play()), a resize drag
        # alone was enough to starve it into genuine buffer underruns.
        # Reported directly: aggressive clicking/popping starting the
        # moment a resize began, regardless of LOOP/REPEAT state --
        # which itself points at resize-triggered CPU contention rather
        # than anything about the audio data or processing mode.
        # Waiting for the resize to actually settle before doing the
        # expensive render, instead of on every intermediate tick, is
        # the standard fix for this class of problem -- the canvas
        # widget's own on-screen bounds still track the drag smoothly
        # via Tk's normal geometry handling either way; only the
        # waveform IMAGE's content lags by this delay.
        if self._canvas_resize_after_id is not None:
            try:
                self.root.after_cancel(self._canvas_resize_after_id)
            except Exception:
                pass
        self._canvas_resize_after_id = self.root.after(120, self._finish_canvas_resize)

    def _finish_canvas_resize(self):
        self._canvas_resize_after_id = None
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
        interval = pick_tick_interval(span_sec, canvas_width_px=w)
        minor_interval = interval / 5.0

        # minor (unlabeled) ticks first so major ticks draw on top of them
        t = (int(vs / self.sr / minor_interval)) * minor_interval
        while t <= ve / self.sr + minor_interval:
            # skip positions that coincide with a major tick (within float tolerance)
            if abs((t / interval) - round(t / interval)) > 1e-6:
                x = self._sample_to_x(t * self.sr, w)
                if -5 <= x <= w + 5:
                    self.timeline_canvas.create_line(x, h - 3, x, h, fill=BORDER)
            t += minor_interval

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

        # Hover takes precedence over the click flag if both are somehow
        # active at once (e.g. clicked, then immediately hovered nearby
        # while the click flag's own 1.5s timer hadn't expired yet) --
        # hover reflects the CURRENT cursor position, which is more
        # relevant than a static, aging click location; drawing both
        # would just overlap.
        if self._hover_flag is not None:
            fx, ftext = self._hover_flag
            self._draw_flag(fx, ftext, w)
        elif self._click_flag is not None:
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

    CLOSE_BTN_SIZE = 18
    CLOSE_BTN_MARGIN = 8

    def _close_button_bbox(self, canvas_width):
        x2 = canvas_width - self.CLOSE_BTN_MARGIN
        x1 = x2 - self.CLOSE_BTN_SIZE
        y1 = self.CLOSE_BTN_MARGIN
        y2 = y1 + self.CLOSE_BTN_SIZE
        return x1, y1, x2, y2

    def _draw_close_button(self, w):
        x1, y1, x2, y2 = self._close_button_bbox(w)
        self.canvas.create_oval(x1, y1, x2, y2, fill=PANEL, outline=BORDER, tags="close_btn")
        pad = self.CLOSE_BTN_SIZE * 0.28
        self.canvas.create_line(x1 + pad, y1 + pad, x2 - pad, y2 - pad, fill=MUTED, width=2, tags="close_btn")
        self.canvas.create_line(x2 - pad, y1 + pad, x1 + pad, y2 - pad, fill=MUTED, width=2, tags="close_btn")
        self.canvas.tag_bind("close_btn", "<Enter>", self._on_close_btn_enter)
        self.canvas.tag_bind("close_btn", "<Leave>", lambda e: self._hide_canvas_tooltip())

    def _on_close_btn_enter(self, event):
        self._show_canvas_tooltip("Unload the current audio file",
                                   self.canvas.winfo_rootx() + event.x,
                                   self.canvas.winfo_rooty() + event.y)

    def _show_canvas_tooltip(self, text, x_root, y_root):
        self._hide_canvas_tooltip()
        self._canvas_tooltip = self.tk.Toplevel(self.canvas)
        self._canvas_tooltip.wm_overrideredirect(True)
        try:
            self._canvas_tooltip.wm_attributes("-topmost", True)
        except Exception:
            pass
        self._canvas_tooltip.wm_geometry(f"+{x_root + 12}+{y_root + 12}")
        self.tk.Label(self._canvas_tooltip, text=text, bg="#111214", fg="#e6e6e8",
                      font=("Segoe UI", 9), padx=8, pady=4, relief="solid", borderwidth=1).pack()

    def _hide_canvas_tooltip(self):
        if getattr(self, "_canvas_tooltip", None) is not None:
            try:
                self._canvas_tooltip.destroy()
            except Exception:
                pass
            self._canvas_tooltip = None

    def unload_file(self):
        self._hide_canvas_tooltip()
        self._exit_preview_mode()
        self.player.stop()
        self.data = None
        self.loaded_path = None
        self.sel_start = self.sel_end = 0
        self.zoom_start = self.zoom_end = 0
        self.cropped = False
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.repeat_var.set(False)
        self._set_export_mode("raw")  # nothing loaded, so no meaningful export mode
        self.in_path_var.set("")
        self.out_path_var.set("")
        self._save_as_root = None  # so the NEXT load's fresh " RAW" default isn't
                                    # mistaken for a leftover custom filename
        self._save_as_user_customized = False
        self.time_var.set("00:00:00.000")
        self._update_selection_duration_label()
        self._update_auto_crossfade_preview()
        self._redraw()
        self.status_var.set("Unloaded. Drag & drop an audio file, or click Browse.")

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

        # subtle zero-crossing reference line (amplitude 0 = vertical center)
        self.canvas.create_line(0, h / 2, w, h / 2, fill=ZERO_LINE_COLOR, width=1)

        sx = self._sample_to_x(self.sel_start, w)
        ex = self._sample_to_x(self.sel_end, w)
        # PIL-rendered RGBA image with real alpha, NOT stipple -- Tk's
        # stipple fills are explicitly documented as unsupported outside
        # X11 ("stipples are not well supported on platforms that do not
        # use X11 as their drawing API" -- the Tk manual itself). On
        # macOS's Aqua, this rendered as a fully OPAQUE fill instead of a
        # semi-transparent highlight, completely hiding the waveform
        # underneath the selected region. A real RGBA image with alpha
        # composites correctly on every platform, since Tk's PhotoImage/
        # Canvas image rendering handles alpha natively rather than
        # through this legacy, X11-specific mechanism.
        sel_w = max(1, int(ex - sx))
        if PIL_AVAILABLE and sel_w > 0:
            sel_img = Image.new("RGBA", (sel_w, max(1, int(h))), _hex_to_rgb(SELECTION_COLOR) + (110,))
            self._sel_photo = ImageTk.PhotoImage(sel_img)  # keep a reference or Tk garbage-collects it
            self.canvas.create_image(sx, 0, anchor="nw", image=self._sel_photo)
        else:
            self.canvas.create_rectangle(sx, 0, ex, h, fill=SELECTION_COLOR, outline="", stipple="gray50")
        self.canvas.create_line(sx, 0, sx, h, fill=HANDLE_COLOR, width=2, tags="handle_start")
        self.canvas.create_line(ex, 0, ex, h, fill=HANDLE_COLOR, width=2, tags="handle_end")

        cursor = self._display_cursor_sample()
        cx = self._sample_to_x(cursor, w)
        self.canvas.create_line(cx, 0, cx, h, fill=PLAYHEAD_COLOR, width=1, tags="playhead")

        self._draw_close_button(w)

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

    def _on_canvas_motion(self, event):
        """Shows the time under the cursor after a brief dwell -- distinct
        from _show_click_flag (which fires immediately on an actual click
        and auto-clears after a fixed 1.5s): this is for PASSIVE hovering,
        debounced so it only appears once the mouse has been reasonably
        still, rather than flickering on every pixel of movement while
        just passing through on the way somewhere else. Suppressed
        entirely while actively dragging a selection/edge (drag_mode is
        not None): <Motion> fires alongside <B1-Motion> for the same
        movement, and a drag already has its own visual feedback (the
        selection itself, plus the click flag once released) -- showing
        a second, hover-driven flag at the same time would clutter the
        timeline rather than help."""
        if self.data is None or self.drag_mode is not None:
            self._cancel_hover_flag()
            return
        w = self.canvas.winfo_width()
        samp = max(0, min(self._x_to_sample(event.x, w), len(self.data)))
        if self._hover_flag_after_id is not None:
            try:
                self.root.after_cancel(self._hover_flag_after_id)
            except Exception:
                pass
        self._hover_flag_after_id = self.root.after(
            200, lambda x=event.x, s=samp: self._show_hover_flag(x, s))

    def _show_hover_flag(self, x_pixel, sample):
        self._hover_flag_after_id = None
        self._hover_flag = (x_pixel, format_time(sample / self.sr))
        self._redraw_timeline()

    def _on_canvas_leave(self, event):
        self._cancel_hover_flag()

    def _cancel_hover_flag(self):
        if self._hover_flag_after_id is not None:
            try:
                self.root.after_cancel(self._hover_flag_after_id)
            except Exception:
                pass
            self._hover_flag_after_id = None
        if self._hover_flag is not None:
            self._hover_flag = None
            self._redraw_timeline()

    def _on_canvas_press(self, event):
        if self.data is None:
            return
        w = self.canvas.winfo_width()
        x1, y1, x2, y2 = self._close_button_bbox(w)
        if x1 <= event.x <= x2 and y1 <= event.y <= y2:
            self.unload_file()
            return
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

    def _snap_selection_edge(self, sample, other_edge_sample=None):
        """Selection edges always snap to the nearest zero crossing -- for
        this app specifically (crossfaded loop points), there's no real
        case where a non-zero-crossing edge is preferable, so this isn't
        an optional toggle. The search radius scales with the current
        zoom level, with a wide-enough capture radius (~10 pixels' worth
        of samples) to feel like a genuine magnetic snap rather than raw
        mouse precision -- tight at extreme zoom, wider when zoomed out,
        capped at a quarter-second so it can't search unreasonably far
        when fully zoomed out.

        When other_edge_sample is given (the OTHER end of the selection,
        already placed), prefers a zero-crossing whose slope direction
        matches the waveform's direction there -- so the loop seam
        continues in the same apparent direction of motion rather than
        reversing right at the wrap point. This is only ever a preference
        on top of an already amplitude-safe candidate; see
        _nearest_zero_crossing_directional for how the fallback works."""
        if self.data is None:
            return sample
        vs, ve = self._visible_range()
        w = max(1, self.canvas.winfo_width() or self.canvas_width)
        per_pixel = max(1, (ve - vs) / w)
        window = max(1, min(int(per_pixel * 10), int(self.sr * 0.25)))
        preferred_direction = None
        if other_edge_sample is not None:
            preferred_direction = self._local_slope_direction(self.data, self.sr, other_edge_sample)
        return self._nearest_zero_crossing_directional(self.data, self.sr, sample, window,
                                                         preferred_direction=preferred_direction,
                                                         hard_cap_seconds=window / self.sr)

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
            self.sel_start = self._snap_selection_edge(min(samp, self.sel_end), other_edge_sample=self.sel_end)
        elif self.drag_mode == "end":
            self.sel_end = self._snap_selection_edge(max(samp, self.sel_start), other_edge_sample=self.sel_start)
        elif self.drag_mode == "new":
            # the anchor (wherever the drag started) has nothing to match
            # yet, so it just takes the nearest amplitude-safe point; the
            # end you're actively dragging then prefers matching ITS
            # direction, once it's been placed
            anchor_samp = self._x_to_sample(self.drag_anchor_x, w)
            snapped_anchor = self._snap_selection_edge(anchor_samp)
            snapped_moving = self._snap_selection_edge(samp, other_edge_sample=snapped_anchor)
            self.sel_start, self.sel_end = min(snapped_anchor, snapped_moving), max(snapped_anchor, snapped_moving)

        if self.drag_mode in ("start", "end", "new"):
            self._update_selection_duration_label()
            self._update_auto_crossfade_preview()
            self._redraw()

    def _raw_to_preview_cursor(self, raw_sample):
        """Inverse of _display_cursor_sample: maps a click position in
        self.data-space into the current preview buffer's own sample
        space. Only meaningful while self.preview_mode is True."""
        if self.player.data is None:
            return 0
        span = max(1, self.sel_end - self.sel_start)
        frac = (raw_sample - self.sel_start) / span
        frac = max(0.0, min(1.0, frac))
        total = self.player.data.shape[0]
        return int(frac * total)

    def _local_slope_direction(self, data, sr, sample, probe_ms=1.0):
        """Classifies the local waveform trend at `sample` as +1 (rising)
        or -1 (falling), using the mono mix -- direction needs one
        definitive answer, unlike the amplitude-closeness check elsewhere
        which deliberately uses the per-channel max to catch out-of-phase
        content. Returns None if the mono mix is too flat locally to call
        a direction (near-silence, or content that happens to sit exactly
        out-of-phase so the mix reads as flat) -- callers must treat None
        as "no preference," not guess a direction from noise."""
        if data is None or len(data) == 0:
            return None
        n = len(data)
        probe = max(4, int(sr * probe_ms / 1000))
        lo = max(0, sample - probe)
        hi = min(n, sample + probe)
        if hi - lo < 2:
            return None
        seg = data[lo:hi]
        mono = seg.mean(axis=1) if seg.ndim > 1 else seg
        diff = float(mono[-1] - mono[0])
        if abs(diff) < 1e-4:
            return None
        return 1 if diff > 0 else -1

    def _zero_crossing_candidates(self, data, sr, sample, window):
        """Returns [(index, direction), ...] for every TRUE sign-change
        zero crossing within [sample-window, sample+window] (mono mix),
        direction +1 for rising (neg->pos), -1 for falling (pos->neg)."""
        n = len(data)
        lo = max(0, sample - window)
        hi = min(n, sample + window)
        if hi - lo < 2:
            return []
        seg = data[lo:hi]
        mono = seg.mean(axis=1) if seg.ndim > 1 else seg
        signs = np.sign(mono)
        candidates = []
        for i in range(len(mono) - 1):
            if signs[i] == 0 or signs[i] == signs[i + 1] or signs[i + 1] == 0:
                continue
            direction = 1 if mono[i + 1] > mono[i] else -1
            candidates.append((lo + i + 1, direction))
        return candidates

    def _nearest_zero_crossing_directional(self, data, sr, sample, max_window_samples,
                                            preferred_direction=None, hard_cap_seconds=2.0):
        """When preferred_direction is given, ALWAYS returns a true zero-
        crossing of that direction if one can be found at all -- widening
        the search progressively rather than settling for a nearby
        mismatched-direction point. This is deliberately strict: matching-
        direction crossings occur roughly once per full waveform cycle
        (vs. once per half-cycle for either direction), so for any real,
        actively-oscillating audio a match is virtually always close by --
        an earlier version gave up and fell back to a mismatched direction
        whenever the match seemed "too far," which in practice meant
        selections could end up with mismatched slopes far more often
        than intended. Only falls back to the plain amplitude-nearest
        point (regardless of direction) for genuinely degenerate audio --
        silence or DC -- where no real crossing exists even within a
        generous widened search."""
        baseline = self._nearest_zero_crossing(data, sr, sample, max_window_samples=max_window_samples)
        if preferred_direction is None or data is None or len(data) == 0:
            return baseline
        hard_cap = min(len(data), max(1, int(sr * hard_cap_seconds)))
        window = max(1, int(max_window_samples))
        while True:
            matching = [c for c in self._zero_crossing_candidates(data, sr, sample, window)
                        if c[1] == preferred_direction]
            if matching:
                best_idx, _ = min(matching, key=lambda c: abs(c[0] - sample))
                return best_idx
            if window >= hard_cap:
                return baseline  # genuinely degenerate case -- accept the mismatch
            window = min(window * 2, hard_cap)

    def _nearest_zero_crossing(self, data, sr, sample, window_ms=15, max_window_samples=None):
        """Finds the sample index within a small window around `sample`
        where the waveform amplitude is closest to zero. Repositioning
        playback to click PRECISELY where the user clicked, mid-waveform-
        cycle, is exactly what produces an audible tick even with a fade
        applied -- fading from silence into a signal that was ALREADY at
        significant amplitude right at the splice point still means the
        envelope itself has a discontinuity at that instant. Starting from
        a near-zero-amplitude point removes the discontinuity at the
        source rather than just smoothing over it. A few samples of
        position accuracy is an easy trade for a genuinely clean splice.

        Uses the MAX absolute value across channels, not the mono mix --
        averaging channels together can hide a real discontinuity: for
        out-of-phase stereo content (e.g. some reverb/widening effects),
        the mix can read as exactly zero at every point in the whole
        signal even while each individual channel sits at full amplitude,
        which would mean the search never moves the click at all.

        max_window_samples caps the search radius -- important at extreme
        zoom, where the default ms-based window can be LARGER than the
        entire visible viewport, snapping the cursor to a position outside
        what's currently shown. The playhead then appears to not move at
        all, when it actually jumped somewhere off-screen."""
        if data is None or len(data) == 0:
            return sample
        window = max(1, int(sr * window_ms / 1000))
        if max_window_samples is not None:
            window = max(1, min(window, int(max_window_samples)))
        n = len(data)
        lo = max(0, sample - window)
        hi = min(n, sample + window)
        if hi <= lo:
            return sample
        segment = data[lo:hi]
        magnitude = np.abs(segment).max(axis=1) if segment.ndim > 1 else np.abs(segment)
        idx = int(np.argmin(magnitude))
        return lo + idx

    def _on_canvas_release(self, event):
        if self.data is None:
            self.drag_mode = None
            return
        if self.drag_mode == "pending":
            # a plain click (no meaningful drag): move the playhead there
            w = self.canvas.winfo_width()
            samp = self._x_to_sample(event.x, w)
            samp = max(0, min(samp, len(self.data)))
            # cap the zero-crossing search to a fraction of a pixel's worth
            # of samples (matching _snap_selection_edge's formula) so at
            # extreme zoom the snap can't jump the cursor somewhere
            # off-screen (which looked like clicking did nothing, when it
            # had actually moved -- just nowhere visible)
            vs, ve = self._visible_range()
            w_px = max(1, self.canvas.winfo_width() or self.canvas_width)
            per_pixel = max(1, (ve - vs) / w_px)
            zc_cap = max(1, min(int(per_pixel * 10), int(self.sr * 0.25)))
            # this zc_cap is ALSO passed below as hard_cap_seconds, not just
            # as the starting search window -- _nearest_zero_crossing_
            # directional's own progressive widening otherwise falls back
            # to its default 2-second hard_cap_seconds regardless of this
            # cap, which let a directional-match search wander far past
            # what "capped at a quarter-second" actually promises. Most
            # noticeable clicking near the very start/end of a file: fewer
            # candidate crossings there (often quiet or fading), and the
            # search can only extend in ONE direction at the boundary, so
            # it widened all the way out (confirmed: ~530ms) chasing a
            # direction-matched crossing instead of landing near the click.
            # while actively playing, prefer a landing point that continues
            # in the SAME direction the waveform was already moving right
            # before the click -- reduces the tick further than amplitude-
            # closeness alone: fading from silence into a point moving the
            # OPPOSITE way still reverses the apparent motion at the splice
            outgoing_direction = None
            if self.player.playing:
                # use the PLAYER's own current buffer/cursor, not always
                # self.data -- during audition, player.cursor is in the
                # short preview buffer's own coordinate space, not raw
                # self.data space, so pairing self.data with player.cursor
                # here would silently compute a meaningless direction
                outgoing_direction = self._local_slope_direction(
                    self.player.data, self.player.sr, self.player.cursor)
            if self.preview_mode and self.sel_start <= samp < self.sel_end:
                # clicking WITHIN the loop region while auditioning: stay in
                # preview mode (don't fall back to raw/unprocessed audio) --
                # this is what lets you scrub right up to the loop-back
                # point and hear the actual crossfaded wrap
                preview_cursor = self._raw_to_preview_cursor(samp)
                preview_cursor = self._nearest_zero_crossing_directional(
                    self.player.data, self.player.sr, preview_cursor, zc_cap,
                    preferred_direction=outgoing_direction, hard_cap_seconds=zc_cap / self.player.sr)
                self.player.set_cursor(preview_cursor)
            else:
                # clicking outside the loop region: always operate on raw
                # audio, so you can freely check surrounding context
                self._exit_preview_mode()
                snapped = self._nearest_zero_crossing_directional(
                    self.data, self.sr, samp, zc_cap, preferred_direction=outgoing_direction,
                    hard_cap_seconds=zc_cap / self.sr)
                self.player.set_cursor(snapped)
            self._show_click_flag(event.x, samp)
            self._redraw()
        elif self.drag_mode in ("start", "end", "new") and self.pre_drag_selection is not None:
            if (self.sel_start, self.sel_end) != self.pre_drag_selection:
                # push the PRE-drag selection so undo restores exactly where the drag began
                old_start, old_end = self.pre_drag_selection
                self.undo_stack.append({
                    "data": self.data, "sel_start": old_start, "sel_end": old_end,
                    "cropped": self.cropped, "zoom_start": self.zoom_start, "zoom_end": self.zoom_end,
                    "label": "selection change",
                })
                self.redo_stack.clear()
                if self.preview_mode:
                    # was auditioning: re-process the NEW selection and keep
                    # looping. on_loop_preview reads self.sel_start/sel_end
                    # directly and calls player.load()/play() itself -- do
                    # NOT also call player.set_selection() here, since by
                    # this point player.data is the short PREVIEW buffer,
                    # not self.data, and these raw (large) indices would
                    # get clamped against it and corrupt the player's
                    # bounds right after the correct reprocessing.
                    self.on_loop_preview(silent=True)
                elif self.sel_end > self.sel_start:
                    self.player.set_selection(self.sel_start, self.sel_end)
                    self._update_auto_crossfade_preview()
            self._update_selection_duration_label()
        self.drag_mode = None
        self.pre_drag_selection = None

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

    def _pan(self, direction):
        if self.data is None:
            return
        n = len(self.data)
        vs, ve = self.zoom_start, self.zoom_end
        span = ve - vs
        step = max(1, int(span * 0.2))
        delta = step if direction < 0 else -step
        new_vs = vs + delta
        new_ve = ve + delta
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
        shift_held = bool(event.state & 0x0001)
        if shift_held:
            self._pan(direction)
        else:
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

    def _refresh_loop_and_repeat_icons(self):
        """Keeps both transport icons in sync with current state --
        Loop's now has three visual states (see _refresh_loop_icon),
        Repeat's reflects export_mode (see _refresh_repeat_icon)."""
        self._refresh_loop_icon()
        self._refresh_repeat_icon()

    @property
    def preview_mode(self):
        return self._preview_mode

    @preview_mode.setter
    def preview_mode(self, value):
        # Deliberately does NOT touch export_mode, the button label, or
        # the filename suffix -- those track the user's INTENDED export
        # choice (see export_mode's own comment in __init__), not
        # whether the player happens to be holding a live preview buffer
        # RIGHT NOW. Crop and PaulXStretch both flip preview_mode False
        # as a side effect of invalidating that buffer; neither means
        # the user changed their mind about exporting as a LOOP.
        self._preview_mode = value

    def _set_export_mode(self, mode):
        """The ONLY way export_mode should ever change -- called
        exclusively from the two toggle handlers (on_repeat_toggle,
        on_loop_preview) and from load_file/unload_file. Refreshes the
        button label, Save As suffix, AND both transport icons together,
        since all three now derive from this single value.

        The icon refresh matters more than it looks: _exit_preview_mode
        (called just before this, from both toggle handlers) does its
        OWN icon refresh internally, but at that point export_mode is
        still whatever it was BEFORE this call -- so that refresh reads
        a stale value and leaves the Loop icon showing the OLD mode's
        color. This call runs after export_mode is actually updated, so
        it corrects that stale read. Confirmed directly: toggling LOOP
        off, or switching from LOOP to REPEAT mid-playback, left the
        Loop icon showing blue (the old "loop" reading) even though the
        Save button had already correctly updated -- the two disagreed
        because only the button/suffix were refreshed here before, not
        the icons."""
        self.export_mode = mode
        self._refresh_process_button_label()
        self._update_save_as_suffix()
        self._refresh_loop_and_repeat_icons()

    def _refresh_process_button_label(self):
        """Keeps the Process & Save button's own label AND hover tooltip
        in sync with export_mode -- mirrors run_process()'s own mode
        logic exactly, so both always state what pressing the button
        would actually do. Called from _set_export_mode, the sole place
        export_mode changes, rather than from a property setter that
        would also fire for reasons unrelated to the user's export
        choice."""
        if not hasattr(self, "process_btn_var"):
            return  # not built yet -- _set_export_mode isn't called
                     # this early, but this guard stays as a safety net
        if self.export_mode == "loop":
            self.process_btn_var.set("Save Crossfaded")
            self._process_btn_tooltip.text = ("Crossfade the current selection "
                                                "and save it to the 'Save as' path")
        elif self.export_mode == "repeat":
            self.process_btn_var.set("Save Declicked")
            # Was previously a static tooltip always reading "Crossfade the
            # current selection...", regardless of mode -- correct for
            # LOOP, but wrong for REPEAT, which declicks rather than
            # crossfades. Reported directly.
            self._process_btn_tooltip.text = ("Declick selection edges "
                                                "and save it to the 'Save as' path")
        else:
            # All three labels drop "Process &" uniformly. "Save" alone
            # is accurate for all three: something is always being
            # saved, and REPEAT/LOOP's own transform is already named
            # by the word that follows it.
            self.process_btn_var.set("Save Unprocessed")
            self._process_btn_tooltip.text = ("Save the current selection as-is "
                                                "to the 'Save as' path")

    def _mode_suffix(self):
        """The Save As filename suffix for export_mode -- deliberately
        mirrors _refresh_process_button_label's own source of truth so
        the two never disagree about what "the current mode" is."""
        if self.export_mode == "loop":
            return " LOOP"
        elif self.export_mode == "repeat":
            return " REPEAT"
        else:
            return " RAW"

    def _update_save_as_suffix(self):
        """Keeps the Save As filename's mode suffix in sync with LOOP /
        REPEAT / neither -- but ONLY while the field still holds exactly
        what this method itself last wrote there. The moment the user
        types anything else into it, _on_out_path_changed below latches
        _save_as_user_customized True, and this permanently becomes a
        no-op for the rest of this loaded file's session -- a manual
        filename is never silently overwritten. Runs automatically on
        every relevant state change: hooked into the preview_mode
        property's setter and repeat_var's own trace_add, the same two
        centralized points _refresh_process_button_label already uses,
        so there's no separate list of call sites to keep in sync."""
        if self._save_as_root is None or self._save_as_user_customized:
            return
        new_suffix = self._mode_suffix()
        new_value = self._save_as_root + new_suffix + self._save_as_ext
        self._save_as_programmatic_update = True
        try:
            self.out_path_var.set(new_value)
        finally:
            self._save_as_programmatic_update = False
        self._save_as_current_suffix = new_suffix

    def _on_out_path_changed(self, *args):
        """Detects a genuine USER edit to Save As -- as opposed to this
        same class's own programmatic updates, from either
        _update_save_as_suffix above or _on_format_changed's extension
        swap, both of which set _save_as_programmatic_update around
        their own writes specifically so this can tell the difference --
        and latches _save_as_user_customized so the suffix is never
        auto-updated again for this file."""
        if self._save_as_programmatic_update or self._save_as_root is None:
            return
        expected = self._save_as_root + self._save_as_current_suffix + self._save_as_ext
        if self.out_path_var.get() != expected:
            self._save_as_user_customized = True

    def _exit_preview_mode(self):
        """Swap the player back to raw (un-processed) audio -- used whenever
        something invalidates a processed preview that's currently loaded."""
        if not self.preview_mode:
            return
        if self.player.playing:
            # hot-swap instead of stop+load -- avoids tearing down the
            # actual audio stream (a separate click source from the
            # position-jump click the declick ramp handles) when this
            # happens WHILE actively playing, e.g. clicking outside the
            # loop region to check surrounding context mid-audition.
            # cursor=self.sel_start lands directly on the right position
            # as part of the swap itself (see swap_playing_buffer's own
            # docstring for why) -- so unlike the not-playing branch
            # below, this does NOT need a separate set_cursor() call
            # afterward. Calling set_cursor() with a target EQUAL to
            # where the cursor already sits would still trigger a full
            # fade-out-then-fade-in cycle for a position that isn't
            # actually moving -- a new, pointless audible dip that
            # wasn't there before this fix.
            self.player.swap_playing_buffer(self.data, self.sr, cursor=self.sel_start)
            self.player.set_loop(self.repeat_var.get(), declick_wrap=True)
            self.player.set_selection(self.sel_start, self.sel_end)
        else:
            self.player.stop()
            self.player.load(self.data, self.sr)
            self.player.set_selection(self.sel_start, self.sel_end)
            self.player.set_cursor(self.sel_start)  # load() resets cursor to 0; restore it to
                                                      # the loop start so a following Play resumes
                                                      # the selection, not the start of the file
        self.preview_mode = False
        self._refresh_loop_and_repeat_icons()

    def _flash_button(self, btn, duration_ms=180):
        """Brief blue flash for press feedback, settling back to the
        normal grey icon-button look."""
        btn.configure(style="IconFlash.TButton")
        self.root.after(duration_ms, lambda: btn.configure(style="Icon.TButton"))

    def _status_for_now_playing(self):
        """The status message for "audio is now playing", for whichever
        mode is currently active -- single source of truth, meant to be
        called from every place that STARTS or RESUMES playback, so none
        of them can drift out of sync or simply forget to set one. That
        was exactly the reported bug: resuming via Space after Stop left
        the OLD "Stopped." message on screen indefinitely, since nothing
        on that resume path had ever touched the status bar at all --
        along with several sibling cases in the same shape (Pause showing
        nothing, live-switching REPEAT while playing never updating it).

        Deliberately does NOT say "click X again to stop" for REPEAT/LOOP
        -- that isn't what pressing them again actually does anymore (see
        export_mode's own comment in __init__): they only ever SWITCH the
        mode live, while playback continues; Space/Stop are the only
        things that actually stop audio. Reported directly as misleading
        -- clicking LOOP again while playing doesn't stop anything, it
        disables LOOP and playback continues, unlooped, past the
        selection end."""
        if self.export_mode == "loop" and self.preview_mode:
            return (f"LOOPING with crossfaded edges ({self._last_loop_dur:.2f}s, crossfade "
                    f"{self._last_loop_xfade_ms:.0f} ms). Click within the selection to scrub, "
                    f"or click LOOP again to switch off.")
        elif self.export_mode == "repeat":
            return "REPEATING with declicked edges. Click REPEAT again to switch off."
        else:
            return "Playing."

    def on_play_pause(self):
        if self.data is None:
            return
        if not SOUNDDEVICE_AVAILABLE:
            self.messagebox.showinfo("FermaLoop", "Install the 'sounddevice' package to enable playback:\npip install sounddevice")
            return
        self._flash_button(self.btn_play)
        if self.player.playing:
            # Pause must ONLY pause, wherever the play head currently is --
            # it must never touch preview_mode/selection/loop state. This
            # used to call _exit_preview_mode() unconditionally before this
            # check, which (while Loop was active) stopped the stream,
            # reset the cursor to the selection start, and turned Loop off
            # -- i.e. pressing Pause looked like it restarted playback
            # instead of pausing it.
            self.player.pause()
            self._set_play_pause_icon(False)
            self._refresh_loop_and_repeat_icons()  # LOOP's icon can now be spinning
                                                     # (accent+animated) while playing,
                                                     # so pausing needs to drop it back
                                                     # to accent+static -- nothing else
                                                     # re-checks player.playing on its own
            self.status_var.set("Paused.")  # previously never set at all -- reported
                                             # directly ("tapping PAUSE... doesn't
                                             # display a PAUSED system message")
        else:
            if self.preview_mode:
                # already has a valid loop preview buffer loaded, just
                # paused -- resume it directly, reusing the player's own
                # (already-correct) internal bounds rather than
                # overwriting them with raw file-space selection indices
                self.player.play()
            elif self.export_mode == "loop":
                # LOOP is the armed mode (e.g. via a fresh L press, or
                # simply left over from before), but nothing's actually
                # been computed yet -- compute it now and start playing,
                # the exact same computation on_loop_preview itself runs.
                self._compute_and_play_loop_preview()
                return  # _compute_and_play_loop_preview sets its own
                         # play/pause icon state, status message, and
                         # handles its own failure cases; nothing further
                         # to do here
            else:
                self.player.set_selection(self.sel_start, self.sel_end)
                self.player.set_loop(self.export_mode == "repeat", declick_wrap=True)
                self.player.play()
            self._set_play_pause_icon(True)
            self._refresh_loop_and_repeat_icons()  # symmetric with the pause branch
                                                     # above -- LOOP's icon needs to
                                                     # start spinning again on resume,
                                                     # the same way it correctly drops
                                                     # to static on pause. Missing here
                                                     # before: resuming a paused LOOP
                                                     # preview left the icon stuck
                                                     # static even though audio (and
                                                     # player.playing) had resumed.
            self.status_var.set(self._status_for_now_playing())  # previously never set on
                                                                   # either the resume-from-
                                                                   # pause path or the fresh-
                                                                   # start path -- reported
                                                                   # directly: resuming after
                                                                   # Stop left "Stopped." on
                                                                   # screen indefinitely

    def on_stop(self):
        self._flash_button(self.btn_stop)
        self.player.stop()
        self._set_play_pause_icon(False)
        self._refresh_loop_and_repeat_icons()  # same reasoning as the pause branch
                                                 # above -- Stop also changes
                                                 # player.playing, which the icon's
                                                 # spinning state now depends on
        self.status_var.set("Stopped.")  # previously never touched the status bar at
                                          # all, so it just kept showing whatever
                                          # playback message was there before -- e.g.
                                          # LOOP's own "click LOOP again to stop"
                                          # message, still displayed after having
                                          # already stopped via THIS button instead
        self._redraw()

    def on_rewind(self):
        self.player.rewind()
        self._redraw()

    def on_repeat_toggle(self):
        new_value = not self.repeat_var.get()
        if new_value and self.preview_mode:
            # REPEAT and LOOP are mutually exclusive -- turning REPEAT on
            # while LOOP/Audition is active needs to turn LOOP off and
            # switch back to the raw selection, the same way pressing
            # Loop while REPEAT is active already correctly does the
            # reverse (on_loop_preview builds a whole new playback setup
            # from scratch, which naturally overwrites REPEAT's state;
            # this path never did the equivalent, so REPEAT silently
            # never actually took effect while LOOP was running).
            # repeat_var is set BEFORE exiting preview mode specifically
            # because _exit_preview_mode's own hot-swap-while-playing
            # path reads the CURRENT repeat_var value to decide the loop
            # state it restores -- it needs to already see the new value.
            self.repeat_var.set(new_value)
            self._exit_preview_mode()
        else:
            self.repeat_var.set(new_value)
        # Always applied explicitly, not left to _exit_preview_mode alone:
        # that method only calls set_loop on its hot-swap-while-playing
        # branch, not its stop+load branch (player wasn't playing) or the
        # plain non-preview toggle path below -- this covers all of them
        # with one idempotent call.
        self.player.set_loop(self.repeat_var.get(), declick_wrap=True)
        self._set_export_mode("repeat" if new_value else "raw")  # this already
                                                                    # refreshes BOTH
                                                                    # transport icons
                                                                    # internally now
        # No auto-play from a stopped state anymore -- R/L now only ever
        # affect ALREADY-playing audio (live-switch, above), never start
        # playback on their own. That auto-play was added a few rounds
        # back specifically to match L's own behavior; it's removed here
        # symmetrically now that L no longer does that either (see
        # on_loop_preview) -- "enabling a state and choosing when to
        # begin playback" was the explicit direction: R/L arm export_mode
        # for Space or Save to act on, rather than forcing playback on
        # every press, which could otherwise interrupt someone who just
        # wants to pick a mode and Save immediately.
        #
        # This never updated the status bar at all, in any of its four
        # cases -- reported directly: the message stayed on whatever
        # LOOP had last set (with its own now-fixed stale wording),
        # regardless of switching to REPEAT, or turning either off.
        # Uses the same centralized _status_for_now_playing helper as
        # Play/Space now does, rather than its own hardcoded copy of the
        # "REPEATING with declicked edges..." string -- keeps the wording
        # from being able to drift between the two call sites again, and
        # fixes the same "...to stop" inaccuracy this string used to have
        # (pressing REPEAT again while playing doesn't stop anything, it
        # switches to raw playback while audio keeps going).
        if new_value:
            if self.player.playing:
                self.status_var.set(self._status_for_now_playing())
            else:
                self.status_var.set("REPEAT armed. Press Space to preview, or Save to export.")
        else:
            if self.player.playing:
                self.status_var.set("Switched to raw playback.")
            else:
                self.status_var.set("REPEAT turned off.")

    def _read_process_params(self, silent=False):
        """Validates and returns (xfade_seconds_or_None, curve, snap, window)
        from the current UI fields, or None if invalid. `silent=True` skips
        the error dialog -- used for live re-audition while the person is
        still mid-typing a number (e.g. "0." before they finish "0.3")."""
        xfade_seconds = None
        if not self.auto_xfade_var.get():
            try:
                xfade_seconds = float(self.xfade_var.get())
                if xfade_seconds <= 0:
                    raise ValueError
            except ValueError:
                if not silent:
                    self.messagebox.showerror("FermaLoop", "Crossfade duration must be a positive number of seconds.")
                return None
        try:
            transient_window = float(self.window_var.get()) if self.snap_var.get() else 0.25
        except ValueError:
            if not silent:
                self.messagebox.showerror("FermaLoop", "Transient search window must be a number of seconds.")
            return None
        curve = "equal_power" if self.curve_var.get() == "Equal power" else "linear"
        return xfade_seconds, curve, self.snap_var.get(), transient_window

    def _on_param_changed(self, *args):
        """Live-update hook, debounced so typing a number doesn't reprocess
        on every keystroke. If we're currently auditioning, re-process and
        keep looping automatically instead of requiring Stop + Audition
        again. Otherwise, if Auto-detect is on, just compute and display
        what crossfade length it would currently pick, so toggling the
        checkbox (or changing the selection/curve/snap settings) gives
        immediate feedback without requiring playback."""
        if self._live_update_after_id is not None:
            self.root.after_cancel(self._live_update_after_id)
        if self.preview_mode:
            self._live_update_after_id = self.root.after(250, lambda: self.on_loop_preview(silent=True))
        else:
            self._live_update_after_id = self.root.after(250, self._update_auto_crossfade_preview)

    def _update_auto_crossfade_preview(self):
        """Computes (without playing or saving) what Auto-detect would
        currently pick for the selection, and shows it both in the status
        bar and as a live value directly under the Auto-detect checkbox."""
        if self.data is None or self.preview_mode or not self.auto_xfade_var.get() or self.sel_end <= self.sel_start:
            self.auto_xfade_value_var.set("")
            return
        try:
            segment = self.data[self.sel_start:self.sel_end]
            curve = "equal_power" if self.curve_var.get() == "Equal power" else "linear"
            if self.snap_var.get():
                window = float(self.window_var.get())
                segment, _, _ = snap_to_transients(segment, self.sr, window)
            xfade = auto_select_xfade(segment, self.sr, curve=curve)
            self.auto_xfade_value_var.set(f"\u2248 {xfade * 1000:.0f} ms")
            self.status_var.set(
                f"Auto-detected crossfade for the current selection: {xfade * 1000:.0f} ms. "
                f"Click LOOP to preview it, or Save to use it."
            )
        except Exception:
            self.auto_xfade_value_var.set("")  # non-fatal -- this is just a live status hint

    def on_loop_preview(self, silent=False):
        """Toggle: a direct (non-silent) press while already auditioning
        turns it OFF and returns to raw audio. Otherwise processes the
        CURRENT selection (crop not required) and plays it looped, without
        touching self.data or writing any file -- so you can hear whether
        the crossfade settings are right before committing.
        `silent=True` is used for automatic live re-audition (selection or
        parameter changes while already auditioning) -- it never toggles
        off, skips dialogs, and quietly does nothing if the current state
        isn't ready to process.

        A direct press does NOT force playback to start on its own
        anymore -- see export_mode's own comment in __init__. If
        something is ALREADY playing, this still live-switches it
        immediately (unchanged, confirmed still wanted); if nothing is
        playing, this only arms export_mode="loop" for Space or Save to
        act on later, rather than forcing audio to start."""
        if self.data is None:
            return

        if self.export_mode == "loop" and not silent:
            # Explicit L press while LOOP is the current mode: turn it
            # off. Gated on export_mode, not preview_mode -- LOOP can be
            # the chosen mode (export_mode=="loop") WITHOUT preview_mode
            # ever having become True, specifically via the "armed but
            # not playing" branch below, which sets export_mode alone
            # without loading/computing anything. Gating this on
            # preview_mode meant a second press while merely armed never
            # recognized LOOP as active at all, and fell through to the
            # "arm" branch again instead -- re-arming the SAME mode,
            # which looked exactly like nothing had happened. Confirmed
            # directly: click LOOP once (arms, icon goes static blue),
            # click again, and it stayed on instead of turning off.
            #
            # _exit_preview_mode already does exactly the right thing
            # for both cases -- hot-swaps to keep playing (now raw) if
            # genuinely playing, or just resets internal state without
            # starting anything if not (including a no-op if
            # preview_mode was never True to begin with, e.g. the
            # armed-but-never-played case this fix specifically covers)
            # -- so this no longer forces a stop afterward the way it
            # used to. L/R only ever change WHAT's playing now;
            # Space/Stop are the only things that actually start or
            # stop it.
            self._exit_preview_mode()
            self._set_export_mode("raw")
            if self.player.playing:
                self.status_var.set("Switched to raw playback.")
            else:
                self._set_play_pause_icon(False)
                self.status_var.set("LOOP turned off.")
            self._update_auto_crossfade_preview()  # otherwise the live value under
                                                     # Auto-detect is left showing
                                                     # whatever it last was mid-audition
                                                     # until some OTHER change happens
                                                     # to trigger a refresh
            self._redraw()
            return

        # A direct press while NOTHING is currently playing just arms
        # export_mode="loop" without computing or starting anything --
        # "enabling a state and choosing when to begin playback" was the
        # explicit direction, rather than forcing playback on every
        # press (which could otherwise interrupt someone who just wants
        # to pick a mode and Save immediately). Checked before the
        # sounddevice/playback-readiness checks below since arming
        # doesn't actually need sound at all -- only an actual attempt
        # to PLAY does. silent=True (live re-audition) always proceeds
        # past this -- it only ever fires while ALREADY previewing.
        if not silent and not self.player.playing:
            if self.sel_end <= self.sel_start:
                self.messagebox.showerror("FermaLoop", "Select a region on the waveform first.")
                return
            self._set_export_mode("loop")
            self.repeat_var.set(False)
            self._refresh_loop_and_repeat_icons()
            self.status_var.set("LOOP armed. Press Space to preview, or Save to export.")
            return

        self._compute_and_play_loop_preview(silent=silent)

    def _compute_and_play_loop_preview(self, silent=False):
        """Computes the crossfaded preview for the current selection and
        starts (or live-hot-swaps into) playing it looped. Split out from
        on_loop_preview so on_play_pause can trigger this directly when
        Space is pressed with export_mode=="loop" but nothing's been
        computed yet -- calling on_loop_preview() itself from there
        would hit ITS OWN "arm without playing" branch instead, since at
        that exact moment self.player.playing is still False (Space
        hasn't started anything yet)."""
        if not SOUNDDEVICE_AVAILABLE:
            if not silent:
                self.messagebox.showinfo("FermaLoop", "Install the 'sounddevice' package to enable playback:\npip install sounddevice")
            return
        if self.sel_end <= self.sel_start:
            if not silent:
                self.messagebox.showerror("FermaLoop", "Select a region on the waveform first.")
            return
        params = self._read_process_params(silent=silent)
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

            was_already_auditioning = self.preview_mode and self.player.playing
            if was_already_auditioning and self.player.swap_playing_buffer(preview, self.sr):
                pass  # hot-swapped without touching the running audio stream
            else:
                self.player.stop()
                self.player.load(preview, self.sr)
                self.player.set_loop(True, declick_wrap=False)
                self.player.play()
            self.preview_mode = True
            # REPEAT and LOOP are mutually exclusive, and this is now the
            # active mode -- repeat_var was otherwise ONLY ever managed by
            # on_repeat_toggle, so if REPEAT had been used earlier in the
            # session, entering LOOP left it stuck at True with nothing to
            # ever reset it. on_repeat_toggle computes its own toggle
            # direction from repeat_var's CURRENT value (not from which
            # mode is visibly active), so that stale True made its very
            # next press read as "turn REPEAT off" instead of "turn REPEAT
            # on" -- confirmed directly via logging: entry showed
            # repeat_var=True while LOOP was playing, and that single
            # press exited with repeat_var=False, never touching
            # preview_mode at all. This is the actual root cause; that
            # logic in on_repeat_toggle was correct all along.
            self.repeat_var.set(False)
            self._set_export_mode("loop")  # explicit user action (a fresh L press,
                                            # since silent=True re-audition only runs
                                            # while ALREADY in preview mode, when
                                            # export_mode is already "loop") -- LOOP
                                            # is now the user's intended export choice
            self._refresh_loop_and_repeat_icons()
            self._set_play_pause_icon(True)

            # _update_auto_crossfade_preview (the usual source of this
            # live value, shown directly under the Auto-detect checkbox)
            # deliberately blanks it while preview_mode is active, and
            # _on_param_changed routes live updates here instead of there
            # once auditioning -- so without this, the value visibly
            # disappeared the moment LOOP was enabled, even though it's
            # computed right here as used_xfade and was already being
            # shown in the status message below.
            if self.auto_xfade_var.get():
                self.auto_xfade_value_var.set(f"\u2248 {used_xfade * 1000:.0f} ms")

            dur = preview.shape[0] / self.sr
            self._last_loop_dur = dur              # so a later RESUME (Space after
            self._last_loop_xfade_ms = used_xfade * 1000  # Pause/Stop) can still show
                                                             # accurate specifics via
                                                             # _status_for_now_playing,
                                                             # without a stale "just
                                                             # computed" claim
            self.status_var.set(
                f"LOOPING with crossfaded edges ({dur:.2f}s, crossfade {used_xfade*1000:.0f} ms, "
                f"computed in {elapsed*1000:.0f} ms). Click within the selection to scrub, adjust "
                f"settings to update live, or click LOOP again to switch off."
            )
        except Exception as e:
            self.status_var.set("Audition failed.")
            if not silent:
                self.messagebox.showerror("FermaLoop", str(e))

    def on_crop(self):
        if self.data is None:
            return
        s, e = self.sel_start, self.sel_end
        if e <= s:
            self.messagebox.showerror("FermaLoop", "Select a region on the waveform first.")
            return
        self.push_undo("crop")
        self.player.stop()
        self.data = self.data[s:e]
        self.sel_start, self.sel_end = 0, len(self.data)
        self.zoom_start, self.zoom_end = 0, len(self.data)
        self.cropped = True
        self.preview_mode = False
        self._refresh_loop_and_repeat_icons()
        self.player.load(self.data, self.sr)
        self._redraw()
        self._update_selection_duration_label()
        self._update_auto_crossfade_preview()
        dur = len(self.data) / self.sr
        self.status_var.set(f"Cropped to {dur:.2f}s. (Cmd/Ctrl+Z to undo.)")

    def open_stretch_dialog(self):
        if self.data is None:
            return
        if self.sel_end <= self.sel_start:
            self.messagebox.showerror("FermaLoop", "Select a region on the waveform first.")
            return

        # stop any active playback/audition before editing audio underneath
        # it -- avoids the transport controls (and the button that started
        # this) ending up in a stale state once the dialog closes
        self._exit_preview_mode()
        self.player.stop()
        self._set_play_pause_icon(False)
        self.status_var.set("Preparing to stretch...")  # without this, the status bar
                                                          # could still read a stale
                                                          # "now playing" message while
                                                          # this (modal) dialog is open --
                                                          # and _poll_playhead's own
                                                          # natural-end detection looks
                                                          # for exactly that wording to
                                                          # decide whether IT should react
        self._redraw()

        tk, ttk = self.tk, self.ttk
        dlg = tk.Toplevel(self.root)
        dlg.title("PaulXStretch")
        dlg.configure(bg=BG)
        dlg.transient(self.root)
        dlg.grab_set()  # modal: prevents interacting with transport controls while this is open

        def close_dialog():
            try:
                self.window_sizes["stretch"] = {
                    "width": dlg.winfo_width(),
                    "height": dlg.winfo_height(),
                }
                save_window_sizes(self.window_sizes)
            except Exception:
                pass
            dlg.destroy()

        dlg.protocol("WM_DELETE_WINDOW", close_dialog)

        sel_dur = (self.sel_end - self.sel_start) / self.sr
        ttk.Label(dlg, text=f"Stretching {format_time(sel_dur)} of selected audio.",
                  background=BG, foreground=FG).pack(anchor="w", padx=14, pady=(14, 4))
        ttk.Label(dlg, text="Extreme time-stretch via phase randomization -- great for "
                             "ambient textures/drones, destroys rhythm and transients.",
                  background=BG, foreground=MUTED, wraplength=350, justify="left",
                  font=("Segoe UI", 9)).pack(anchor="w", padx=14, pady=(0, 10))

        row = ttk.Frame(dlg); row.pack(fill="x", padx=14, pady=(4, 0))
        ttk.Label(row, text="Stretch factor:", background=BG, foreground=FG).pack(side="left")
        factor_var = tk.StringVar(value="8.0")
        factor_entry = RoundedEntry(row, factor_var, BG, FIELD_BG, FG, BORDER, height=28, radius=8, width=80)
        factor_entry.pack(side="right")
        ttk.Label(dlg, text="Typical range: 2-50 (higher takes longer and produces much longer output)",
                  background=BG, foreground=MUTED, font=("Segoe UI", 8)).pack(anchor="w", padx=14, pady=(2, 8))

        row = ttk.Frame(dlg); row.pack(fill="x", padx=14, pady=(4, 0))
        ttk.Label(row, text="Window size:", background=BG, foreground=FG).pack(side="left")
        window_var = tk.StringVar(value="0.25")
        stretch_window_entry = RoundedEntry(row, window_var, BG, FIELD_BG, FG, BORDER, height=28, radius=8, width=80)
        stretch_window_entry.pack(side="right")
        ttk.Label(dlg, text="Typical range: 0.05-2.0 seconds (0.1-0.25 is a good starting point)",
                  background=BG, foreground=MUTED, font=("Segoe UI", 8)).pack(anchor="w", padx=14, pady=(2, 0))
        ttk.Label(dlg, text="Smaller reduces amplitude pulsing but can sound grainier/less full; "
                             "larger sounds fuller but may pulse more. Try both -- it's genuinely a "
                             "per-track tradeoff, not a single right answer.",
                  background=BG, foreground=MUTED, font=("Segoe UI", 8),
                  wraplength=350, justify="left").pack(anchor="w", padx=14, pady=(0, 4))

        result_label = ttk.Label(dlg, text="", background=BG, foreground=MUTED,
                                  wraplength=350, justify="left")
        result_label.pack(anchor="w", padx=14, pady=(8, 0))

        def apply_stretch():
            try:
                factor = float(factor_var.get())
                window_seconds = float(window_var.get())
                if factor <= 0 or window_seconds <= 0:
                    raise ValueError
            except ValueError:
                self.messagebox.showerror("PaulXStretch", "Stretch factor and window size must be positive numbers.")
                return

            est_out_sec = sel_dur * factor
            if est_out_sec > 300 and not self.messagebox.askyesno(
                    "PaulXStretch",
                    f"This will produce about {format_time(est_out_sec)} of audio "
                    f"({est_out_sec/60:.1f} minutes). Continue?"):
                return

            try:
                result_label.configure(text="Stretching...")
                dlg.update_idletasks()
                s, e = self.sel_start, self.sel_end
                segment = self.data[s:e]
                t0 = time.time()
                stretched = paulstretch(segment, self.sr, factor, window_seconds)
                elapsed = time.time() - t0

                self.push_undo("stretch")
                self.player.stop()
                self.data = np.concatenate([self.data[:s], stretched, self.data[e:]], axis=0)
                self.sel_start = s
                self.sel_end = s + stretched.shape[0]
                self.zoom_start, self.zoom_end = 0, len(self.data)
                self.preview_mode = False
                self._refresh_loop_and_repeat_icons()
                self._set_play_pause_icon(False)
                self.time_var.set("00:00:00.000")
                self.player.load(self.data, self.sr)
                self._redraw()
                self._update_selection_duration_label()
                self._update_auto_crossfade_preview()
                self.zoom_to_selection()

                out_dur = stretched.shape[0] / self.sr
                self.status_var.set(
                    f"Stretched {sel_dur:.2f}s to {out_dur:.2f}s ({factor:.1f}x) in {elapsed:.1f}s. "
                    f"(Cmd/Ctrl+Z to undo.) LOOP or Crop to build the loop."
                )
                close_dialog()
            except Exception as ex:
                result_label.configure(text=f"Failed: {ex}")

        btn_row = ttk.Frame(dlg); btn_row.pack(fill="x", padx=14, pady=14)
        # Centered, not right-justified: an inner frame holding both
        # buttons, packed into btn_row with no side/fill specified, lands
        # centered horizontally by pack's own default cross-axis
        # behavior (its default anchor is "center") rather than stuck to
        # one edge.
        btn_group = ttk.Frame(btn_row)
        btn_group.pack()
        ttk.Button(btn_group, text="Stretch", style="Accent.TButton", command=apply_stretch).pack(side="left")
        ttk.Button(btn_group, text="Cancel", command=close_dialog).pack(side="left", padx=(6, 0))

        # A single Return/Enter immediately runs Stretch using whatever
        # values are currently in the fields -- if the person hasn't
        # touched them, that's the shown defaults, which is exactly the
        # "assume current settings are wanted" behavior asked for. This
        # used to require a first Enter to defocus the entry (via
        # _defocus_on_return) before a SECOND Enter, now at the dialog
        # level, actually ran Stretch -- that existed because the entry's
        # own binding returned "break", stopping the same keypress from
        # also reaching dlg's binding below. Binding the entries directly
        # to apply_stretch here removes the need for that relay entirely,
        # without touching focus_force() below (the actual fix for the
        # separate Windows issue where the dialog never got real OS-level
        # keyboard focus in the first place -- unrelated to this).
        factor_entry.entry.bind("<Return>", lambda e: apply_stretch())
        factor_entry.entry.bind("<KP_Enter>", lambda e: apply_stretch())
        stretch_window_entry.entry.bind("<Return>", lambda e: apply_stretch())
        stretch_window_entry.entry.bind("<KP_Enter>", lambda e: apply_stretch())
        dlg.bind("<Return>", lambda e: apply_stretch())
        dlg.bind("<KP_Enter>", lambda e: apply_stretch())
        # Cmd+. on macOS / Ctrl+. on Windows cancels, matching each
        # platform's own long-standing "period to cancel/stop" convention
        # (e.g. Cmd+. has cancelled dialogs and long-running operations in
        # Mac software for decades).
        cancel_key = "<Command-period>" if IS_MACOS else "<Control-period>"
        dlg.bind(cancel_key, lambda e: close_dialog())

        # size to fit all content on first paint, respecting a larger saved size
        dlg.update_idletasks()
        required_w, required_h = dlg.winfo_reqwidth(), dlg.winfo_reqheight()
        saved = self.window_sizes.get("stretch")
        w, h = resolve_window_size(required_w, required_h, saved)
        # center over the main window -- this dialog previously only ever
        # set a SIZE, never a position at all, leaving placement entirely
        # to the OS default (which is what was landing it at top-left)
        root_x, root_y = self.root.winfo_rootx(), self.root.winfo_rooty()
        root_w, root_h = self.root.winfo_width(), self.root.winfo_height()
        x = root_x + max(0, (root_w - w) // 2)
        y = root_y + max(0, (root_h - h) // 2)
        dlg.geometry(f"{w}x{h}+{x}+{y}")
        dlg.after(20, lambda: dlg.geometry(f"{w}x{h}+{x}+{y}"))
        dlg.minsize(required_w, required_h)

        # Explicitly claim OS-level keyboard focus for this dialog --
        # grab_set() restricts input to WITHIN this dialog's own
        # hierarchy, but doesn't by itself guarantee the dialog actually
        # becomes the active/focused window. Reported on Windows
        # specifically: the dialog appeared visually on top, but its own
        # titlebar looked inactive while the main window's stayed active
        # -- meaning Return/Enter was still being delivered to the main
        # window, not this dialog or its entry fields, so accepting the
        # default Stretch factor/window size via the keyboard silently
        # did nothing. Focusing the first entry field directly both
        # fixes that and lets someone start typing immediately without
        # needing to click first. The deferred re-assertion is a safety
        # net for the same class of "window server needs a beat to
        # settle" timing issue already seen with the Format dropdown.
        factor_entry.entry.focus_force()
        dlg.after(50, lambda: factor_entry.entry.focus_force() if dlg.winfo_exists() else None)

    def _poll_playhead(self):
        is_playing = self.data is not None and self.player.playing
        if is_playing:
            self._redraw()
            display_samp = self._display_cursor_sample()
            self.time_var.set(format_time(display_samp / self.sr))
        if self._was_playing_last_poll and not is_playing:
            # Playback stopped ON ITS OWN since the last poll -- most
            # notably, RAW (non-looped) playback reaching the natural
            # end of the file via the audio callback's own CallbackStop,
            # rather than through Pause, Stop, or either toggle handler.
            # REPEAT and LOOP both loop indefinitely and so never hit
            # this on their own; this is specifically the RAW case.
            # Nothing else in the app catches this transition at all --
            # none of the status-message/icon fixes elsewhere apply,
            # since none of THEIR call sites ever run here. Before this,
            # the play/pause icon and status message would silently
            # freeze on "now playing" indefinitely once audio actually
            # finished on its own. The OLD version of this check lived
            # entirely inside the `if is_playing:` block above, gated on
            # player.playing already being True at the TOP of this same
            # tick -- which only catches the narrow case where playback
            # ends in the handful of milliseconds between reading that
            # flag and reaching this line. In practice playback almost
            # always finishes BETWEEN two 50ms poll ticks, not during
            # one, so that version essentially never fired; tracking the
            # previous tick's state explicitly is what actually makes
            # this reliable.
            #
            # Only overwrites the status message if it STILL looks like
            # one of the "now playing" messages -- i.e. nothing has
            # ALREADY reacted to this same transition with its own, more
            # specific message (Stop's "Stopped.", Pause's "Paused.",
            # either toggle's "Switched to..."/"...turned off."). Player.
            # playing also flips to False when the user explicitly
            # stops/pauses, not just on a natural end -- and those
            # explicit actions run synchronously, fully finishing
            # (including setting their own status message) before this
            # scheduled poll tick gets a chance to run at all. Checking
            # the message's own CURRENT content, rather than a separate
            # flag every explicit stop-causing action would need to
            # remember to set, is what keeps this from stomping a
            # message an explicit action already set correctly for the
            # exact same transition -- and stays correct automatically
            # for any future stop-causing action too, without needing
            # its own update here.
            current = self.status_var.get()
            if current.startswith(("LOOPING", "REPEATING", "Playing.")):
                self._set_play_pause_icon(False)
                self._refresh_loop_and_repeat_icons()
                self.status_var.set("Finished playing.")
        self._was_playing_last_poll = is_playing
        self.root.after(50, self._poll_playhead)

    # ---------------- keyboard shortcuts ----------------

    def _action_map(self):
        return {
            "play_pause": self.on_play_pause,
            "stop": self.on_stop,
            "rewind": self.on_rewind,
            "loop_toggle": self.on_repeat_toggle,
            "crop": self.on_crop,
            "audition": self.on_loop_preview,
            "undo": self.undo,
            "redo": self.redo,
            "zoom_in": lambda: self._zoom_step(1),
            "zoom_out": lambda: self._zoom_step(-1),
            "zoom_fit": self.zoom_to_fit,
            "zoom_selection": self.zoom_to_selection,
            "stretch": self.open_stretch_dialog,
        }

    def _bind_shortcuts(self):
        for name, fn in self._action_map().items():
            self._bind_one(self.shortcuts.get(name, DEFAULT_SHORTCUTS[name]), fn)

    def _text_entry_focused(self):
        """True if the currently focused widget is a real tk.Entry (the
        widget every RoundedEntry wraps internally). Global shortcuts are
        bound directly to the root window with no built-in focus
        awareness, so without this check, typing/copying/cutting text in
        ANY entry field also fires whatever global shortcut happens to
        share that key -- e.g. Ctrl+C (Copy) also matched the "c" (Crop)
        shortcut, and Ctrl+X (Cut) also matched "x" (Stretch), because Tk
        only checks modifiers a binding pattern explicitly names; a plain
        "c"/"x" pattern matches regardless of whether Ctrl is ALSO held."""
        try:
            return isinstance(self.root.focus_get(), self.tk.Entry)
        except Exception:
            return False

    def _bind_one(self, key, fn):
        def guarded(event=None):
            if self._text_entry_focused():
                return  # let the entry handle typing/copy/cut/paste normally
            fn()
        seq = f"<{key}>" if len(key) > 1 else f"<KeyPress-{key}>"
        try:
            # bind_all rather than binding just the root window: more
            # robust against focus ambiguity across platforms (reported
            # as sporadic on macOS) -- bind_all intercepts the keypress
            # regardless of which specific widget currently has focus,
            # rather than depending on the root window itself being what
            # the OS currently considers focused.
            self.root.bind_all(seq, guarded)
        except self.tk.TclError:
            pass  # an invalid/unsupported key sequence shouldn't crash the app

    def _rebind_all(self):
        for name, fn in self._action_map().items():
            key = self.shortcuts.get(name, DEFAULT_SHORTCUTS[name])
            self._bind_one(key, fn)
        self._refresh_transport_tooltips()

    def _refresh_transport_tooltips(self):
        """Updates every transport button's hover hint to reflect the
        current shortcut bindings -- called after a remap so a tooltip
        never shows a stale key."""
        tooltip_attrs = {
            "_play_tooltip": "play_pause",
            "_stop_tooltip": "stop",
            "_repeat_tooltip": "loop_toggle",
            "_loop_tooltip": "audition",
            "_crop_tooltip": "crop",
            "_stretch_tooltip": "stretch",
        }
        for attr, action_name in tooltip_attrs.items():
            tooltip = getattr(self, attr, None)
            if tooltip is not None:
                tooltip.set_rich(*self._transport_tooltip(action_name))

    @staticmethod
    def _event_to_key_string(event):
        """Builds a Tkinter-bindable key string (e.g. 'Control-z', or on
        macOS 'Command-z') from a KeyPress event, including modifiers --
        needed so remapping Undo/Redo (or any shortcut) to a modifier
        combo actually works, not just the bare key.

        On macOS, Tk's Aqua port maps the Command key to what's
        internally still called "Mod1" -- the SAME state bit (0x0008)
        other platforms use for Alt. Command is by far the more
        commonly intended modifier for application shortcuts on Mac
        (Option/Alt is rarely used that way), so that bit is labeled
        "Command" there instead of "Alt". This is based on well-
        documented Tk-on-Aqua behavior; I can't personally verify it on
        a real Mac from here, so if a remapped Command-based shortcut
        doesn't take, that assumption is the first thing worth checking."""
        parts = []
        state = event.state
        if state & 0x0004:
            parts.append("Control")
        if state & 0x0001:
            parts.append("Shift")
        if IS_MACOS:
            if state & 0x0008:
                parts.append("Command")
        else:
            if state & 0x0008 or state & 0x20000:
                parts.append("Alt")
        keysym = event.keysym
        if keysym in ("Control_L", "Control_R", "Shift_L", "Shift_R", "Alt_L", "Alt_R",
                      "Meta_L", "Meta_R", "Super_L", "Super_R"):
            return None  # a bare modifier key press isn't a usable shortcut on its own
        parts.append(keysym)
        return "-".join(parts)

    def open_shortcuts_dialog(self):
        # toggle: clicking the info icon again while the dialog is already
        # open closes it (same as clicking its own close button), rather
        # than just re-focusing an existing window every time
        if getattr(self, "_shortcuts_dialog", None) is not None:
            try:
                if self._shortcuts_dialog.winfo_exists():
                    close_fn = getattr(self, "_shortcuts_dialog_close_fn", None)
                    if close_fn is not None:
                        close_fn()
                    else:
                        self._shortcuts_dialog.destroy()
                        self._shortcuts_dialog = None
                    return
            except Exception:
                pass
            self._shortcuts_dialog = None

        tk, ttk = self.tk, self.ttk
        import webbrowser
        dlg = tk.Toplevel(self.root)
        dlg.withdraw()  # hidden until fully built AND correctly positioned below --
                         # without this, the Toplevel is visible immediately at
                         # whatever default position the window manager chooses,
                         # then visibly jumps to its real position once geometry()
                         # is finally called at the end of this function. Reported
                         # directly on macOS as a "ghost" flash of the window at
                         # the wrong spot (down and left of where it settles)
                         # before snapping to the correct one (right of, and
                         # top-aligned with, the main window) a moment later.
        self._shortcuts_dialog = dlg
        dlg.title("Preferences and Help")
        dlg.configure(bg=BG)
        # Deliberately NOT calling dlg.transient(self.root) here (unlike
        # the PaulXStretch dialog, which correctly uses it alongside
        # grab_set() since THAT dialog is intentionally modal). Tk's own
        # docs describe transient as "always appears in front of its
        # parent" -- on macOS specifically, this was being enforced
        # aggressively enough to actively steal focus back from the main
        # window on a loop, breaking hover-based tooltips on ALL main-
        # window widgets for as long as this (non-modal, no grab_set())
        # dialog stayed open. This dialog was always meant to let the
        # user freely interact with both windows at once -- transient()
        # was directly undermining that, not supporting it.

        content = tk.Frame(dlg, bg=BG)
        content.pack(fill="both", expand=True)

        tooltip_toggle = RoundedCheckbutton(content, "Show hover tooltips", self.tooltips_enabled_var,
                                             BG, FG, FIELD_BG, ACCENT, BORDER,
                                             command=self._on_tooltips_toggle)
        tooltip_toggle.pack(anchor="w", padx=12, pady=(12, 10))

        tk.Frame(content, height=1, bg=BORDER).pack(fill="x", padx=12, pady=(0, 8))

        ttk.Label(content, text="HINTS", background=BG, foreground=FG,
                  font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=12, pady=(0, 4))
        for hint in ("Click and drag to select; drag white edge bars to adjust",
                     "Click in waveform to move playhead",
                     "Optional: Stretch selection with PaulXStretch",
                     "Set desired LOOP XFade Curve/Overlap & Alignment options",
                     "Enable REPEAT or LOOP to audition and determine saved file processing"):
            ttk.Label(content, text=f"\u2022 {hint}", background=BG, foreground=MUTED,
                      font=("Segoe UI", 9)).pack(anchor="w", padx=22, pady=1)

        tk.Frame(content, height=1, bg=BORDER).pack(fill="x", padx=12, pady=(10, 8))

        ttk.Label(content, text="KEYBOARD SHORTCUTS", background=BG, foreground=FG,
                  font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=12, pady=(0, 4))

        shortcuts_frame = tk.Frame(content, bg=BG)
        shortcuts_frame.pack(fill="x", padx=12, pady=(0, 10))

        # 2px trimmed off the top and bottom of the base "TButton" style's
        # own padding (explicitly 8px uniform, set in _build_style --
        # not a guessed/unknown OS theme default), tightening each
        # shortcut-key button's own height without touching horizontal
        # padding or any other button elsewhere in the app that still
        # uses the base TButton/Accent.TButton styles.
        ttk.Style().configure("ShortcutKey.TButton", padding=(8, 6, 8, 6))

        rows = {}
        for i, (name, label) in enumerate(SHORTCUT_LABELS.items()):
            ttk.Label(shortcuts_frame, text=label, background=BG, foreground=FG).grid(
                row=i, column=0, sticky="w", pady=1)
            raw_key = self.shortcuts.get(name, DEFAULT_SHORTCUTS[name])
            btn = ttk.Button(shortcuts_frame, text=format_key_for_display(raw_key), width=14,
                              style="ShortcutKey.TButton")
            btn.grid(row=i, column=1, padx=(10, 0), pady=1)
            rows[name] = btn

        def start_listening(name, btn):
            btn.configure(text="Press a key...")

            def capture(event):
                key = self._event_to_key_string(event)
                if key is None:
                    return  # ignore bare modifier presses, keep listening
                self.shortcuts[name] = key
                btn.configure(text=format_key_for_display(key))
                dlg.unbind("<KeyPress>")
                self._rebind_all()

            dlg.bind("<KeyPress>", capture)

        for name, btn in rows.items():
            btn.configure(command=lambda n=name, b=btn: start_listening(n, b))

        # (shortcuts_frame's own bottom padding already provides margin)

        tk.Frame(content, height=1, bg=BORDER).pack(fill="x", padx=12, pady=(8, 6))
        footer_row = tk.Frame(content, bg=BG)
        footer_row.pack(anchor="w", fill="x", padx=12, pady=(0, 10))
        ttk.Label(footer_row, text=f"FermaLoop v{APP_VERSION}", background=BG, foreground=MUTED,
                  font=("Segoe UI", 8)).pack(side="left")
        # separator kept as its own plain (non-underlined, non-clickable)
        # label -- previously the leading spaces + bullet were part of
        # the SAME label as the URL, so the underline/accent-color
        # styling made the whole thing (spacing and bullet included)
        # visually read as one giant clickable link, when only the URL
        # itself actually is one
        ttk.Label(footer_row, text="  \u2022  ", background=BG, foreground=MUTED,
                  font=("Segoe UI", 8)).pack(side="left")
        link_label = tk.Label(footer_row, text=APP_URL, bg=BG, fg=ACCENT,
                               font=("Segoe UI", 8, "underline"), cursor="hand2")
        link_label.pack(side="left")
        link_label.bind("<Button-1>", lambda e: webbrowser.open(APP_URL))

        dlg.update_idletasks()
        required_w, required_h = dlg.winfo_reqwidth(), dlg.winfo_reqheight()
        saved = self.window_sizes.get("shortcuts", {})
        w, h = resolve_window_size(required_w, required_h, saved)
        dlg.minsize(required_w, required_h)

        # ---- position: appears to the right of the main window, computed
        # once when opened. Deliberately does NOT try to follow the main
        # window if it's moved afterward -- that turned out to be a much
        # harder problem than it looked (likely Windows animating the
        # move rather than snapping instantly), and wasn't worth the
        # complexity or risk it kept introducing. It's still fully
        # draggable by hand once open, same as any other window. ----
        # y uses winfo_y(), NOT winfo_rooty(): rooty gives the top of the
        # window's CONTENT area (below the titlebar), which is why this
        # dialog was consistently appearing offset downward by roughly
        # the titlebar's height on both macOS and Windows instead of
        # flush with the actual top of the main window. winfo_y() gives
        # the position of the window frame itself, titlebar included.
        x = self.root.winfo_rootx() + self.root.winfo_width() + 10
        y = self.root.winfo_y()
        dlg.geometry(f"{w}x{h}+{x}+{y}")
        dlg.deiconify()  # reveal now that content is built AND correctly
                          # positioned -- see the withdraw() call above
        dlg.after(20, lambda: dlg.geometry(f"{w}x{h}+{x}+{y}"))  # defensive re-apply against WM timing quirks

        def on_close():
            save_shortcuts(self.shortcuts)
            try:
                self.window_sizes["shortcuts"] = {
                    "width": dlg.winfo_width(), "height": dlg.winfo_height(),
                }
                save_window_sizes(self.window_sizes)
            except Exception:
                pass
            self._shortcuts_dialog = None
            self._shortcuts_dialog_close_fn = None
            dlg.destroy()

        self._shortcuts_dialog_close_fn = on_close
        dlg.protocol("WM_DELETE_WINDOW", on_close)

    # ---------------- process & save ----------------

    def run_process(self):
        if self.data is None:
            self.messagebox.showerror("FermaLoop", "Load an audio file first.")
            return
        out_path = self.out_path_var.get()
        if not out_path:
            self.messagebox.showerror("FermaLoop", "Choose a Save As location first.")
            return
        if self.sel_end <= self.sel_start:
            self.messagebox.showerror("FermaLoop", "Select a region on the waveform first.")
            return
        if os.path.exists(out_path):
            if not self.messagebox.askyesno(
                    "FermaLoop",
                    f"'{os.path.basename(out_path)}' already exists. Overwrite it?"):
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

            # Export mode follows export_mode -- the user's last EXPLICIT
            # toggle choice, not whatever preview_mode/repeat_var happen
            # to read right now. This used to check self.preview_mode
            # directly, which meant Crop or PaulXStretch -- both of
            # which flip preview_mode False as a side effect of
            # invalidating the live preview buffer, with NO intent to
            # change the export choice -- would silently fall through to
            # exporting UNPROCESSED audio after a confirmed, tested LOOP
            # selection. Confirmed as a real (not just theoretical) gap:
            # reported directly, with the exact Open->Select->LOOP->Crop
            # sequence that triggers it. export_mode is set ONLY by the
            # two toggle handlers and by load_file/unload_file, so it
            # can't be reset by anything that merely edits audio.
            if self.export_mode == "loop":
                result, used_xfade, start_trim, end_trim = _run_pipeline(
                    segment, self.sr, xfade_seconds, curve, snap, window, self.auto_xfade_var.get(),
                )
                mode_label = "Crossfaded"
            elif self.export_mode == "repeat":
                result = declick_edges(segment, self.sr)
                used_xfade, start_trim, end_trim = 0.0, 0, 0
                mode_label = "Declicked"
            else:
                result = segment
                used_xfade, start_trim, end_trim = 0.0, 0, 0
                mode_label = "Unprocessed"

            encode_from_pcm(result, self.sr, self.sampwidth, out_path,
                             mp3_quality=int(round(self.mp3_quality_var.get())))
            elapsed = time.time() - t0
            duration = result.shape[0] / self.sr
            msg = f"Done in {elapsed:.2f}s -- {duration:.2f}s ({mode_label}) saved to:\n{out_path}"
            if mode_label == "Crossfaded":
                msg += f"\nCrossfade used: {used_xfade * 1000:.0f} ms"
                if snap:
                    msg += (f"\nTrimmed to transients: {start_trim / self.sr * 1000:.0f} ms from start, "
                            f"{end_trim / self.sr * 1000:.0f} ms from end")
            self.status_var.set(msg)
        except Exception as e:
            self.status_var.set("Failed. See error dialog.")
            self.messagebox.showerror("FermaLoop", str(e))

    # ---------------- window sizing ----------------

    def _apply_saved_or_natural_size(self):
        """Sizes the window to fit everything on first paint (no manual
        resize needed), while respecting a larger size -- and now also a
        saved position -- the user may have deliberately set last time."""
        self.root.update_idletasks()

        # Measure the box row's natural (side-by-side, the only
        # arrangement that exists now) size directly -- no more force-
        # stacked-to-measure-then-restore-side-by-side dance. Each box's
        # canvas still needs its width/height explicitly set before
        # measuring: a bare tk.Canvas without an explicit -width doesn't
        # compute winfo_reqwidth() from its content, it just reports
        # whatever it was last actually rendered at.
        for box_outer, _ in self._box_pairs:
            box_outer._rc["canvas"].configure(
                width=box_outer._rc["natural_w"], height=box_outer._rc["natural_h"])

        # The loop-group canvas (wraps XFADE CURVE/OVERLAP/LOOP ALIGNMENT
        # with the border+tail) has the same bare-canvas issue, sized to
        # whatever cols_row needs at this same natural, side-by-side width.
        self._loop_group_cols_row.update_idletasks()
        _group_content_w = self._loop_group_cols_row.winfo_reqwidth()
        _group_content_h = self._loop_group_cols_row.winfo_reqheight()
        _margin, _tail_h = self._loop_group_margin, self._loop_group_tail_h
        self._loop_group_canvas.configure(
            width=_group_content_w + _margin * 2,
            height=_group_content_h + _margin * 2 + _tail_h,
        )

        self.root.update()  # full update -- reqwidth needs a real event
                             # pass to settle and reflect this configure change
        content_w = self.root.winfo_reqwidth()
        natural_h = self.root.winfo_reqheight()  # baseline BEFORE swapping in worst-case text

        # Temporarily substitute a realistic WORST-CASE status message
        # before measuring height, then restore the real one right after.
        # The status label's text changes constantly at runtime (load/
        # crop/undo/process messages), and the longest of these -- the
        # "Process & Save" success message -- can wrap to 3-4 lines via
        # explicit newlines (it includes the full output file path,
        # which can be long depending on the user's folder structure).
        # Sizing the window from whatever short text happens to be
        # showing at startup (typically empty, or the initial "Unloaded"
        # hint) meant the window was never actually tall enough for that
        # message once it appeared -- it would get clipped at the bottom
        # edge, with no indication anything was cut off unless you
        # already knew to manually resize the window taller and look.
        _example_path = "/Users/example/Music/Theatrical Show 2026/Act 2 Scene 3 Ambient Drone LOOP.flac"
        _worst_case_status = (
            "Done in 12.34s -- 45.67s loop saved to:\n"
            f"{_example_path}\n"
            "Crossfade used: 42 ms\n"
            "Trimmed to transients: 15 ms from start, 23 ms from end"
        )
        _real_status = self.status_var.get()
        self.status_var.set(_worst_case_status)
        self.root.update_idletasks()
        worst_case_h = self.root.winfo_reqheight()
        self.status_var.set(_real_status)
        self.root.update_idletasks()

        # Cap how much EXTRA height the worst-case measurement is allowed
        # to add over the natural baseline, rather than trusting it
        # unconditionally. The same worst-case string's wrapped line
        # count isn't reliably consistent across platforms/fonts --
        # "Segoe UI" (Windows) and whatever Tk substitutes for it on
        # macOS can wrap the same text to meaningfully different numbers
        # of lines. Uncapped, this measurement added a modest, reasonable
        # amount of extra height on macOS but ballooned the window to a
        # genuinely huge size on Windows for the exact same source
        # string. A fixed cap keeps this a sensible safety margin
        # everywhere instead of an unpredictable one.
        MAX_EXTRA_STATUS_HEIGHT = 80
        content_h = min(worst_case_h, natural_h + MAX_EXTRA_STATUS_HEIGHT)

        saved = self.window_sizes.get("main")
        w, h = resolve_window_size(content_w, content_h, saved)
        if saved and "x" in saved and "y" in saved:
            try:
                x, y = int(saved["x"]), int(saved["y"])
                self.root.geometry(f"{w}x{h}+{x}+{y}")
            except (TypeError, ValueError):
                self.root.geometry(f"{w}x{h}")
        else:
            self.root.geometry(f"{w}x{h}")

        self._redraw_loop_group()  # the one real, visible render pass -- draws
                                    # the group border/tail at its final size
        self.root.minsize(content_w, content_h)  # content_w/content_h ARE the
                                                  # natural side-by-side size
                                                  # measured above, so this is
                                                  # already the true minimum --
                                                  # no collapsibility below it,
                                                  # by design

    def _on_close(self):
        try:
            self.window_sizes["main"] = {
                "width": self.root.winfo_width(),
                "height": self.root.winfo_height(),
                "x": self.root.winfo_x(),
                "y": self.root.winfo_y(),
            }
            self.window_sizes["tooltips_enabled"] = ToolTip.enabled
            save_window_sizes(self.window_sizes)
        except Exception:
            pass
        self.root.destroy()

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
    parser.add_argument("--mp3-quality", type=int, default=2, choices=range(10),
                         help="LAME VBR quality for .mp3 output: 0=best/largest, 9=worst/smallest (default: 2)")
    args = parser.parse_args()

    if args.xfade is None and not args.auto_xfade:
        args.auto_xfade = True  # sensible default if the user specified neither

    info = process_file(
        args.input, args.output,
        xfade_seconds=args.xfade, curve=args.curve,
        snap_transients=args.snap_transients, transient_window=args.transient_window,
        auto_xfade=args.auto_xfade, mp3_quality=args.mp3_quality,
    )
    dur = info["n_samples"] / info["samplerate"]
    print(f"Wrote {args.output}: {dur:.3f}s, crossfade {info['xfade_seconds']*1000:.0f} ms")
    if args.snap_transients:
        print(f"  trimmed {info['start_trim_samples']} samples from start, "
              f"{info['end_trim_samples']} from end")


if __name__ == "__main__":
    main()
