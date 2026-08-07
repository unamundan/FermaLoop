#!/usr/bin/env python3
"""
FermaLoop
=========
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
import math
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

DEFAULT_SHORTCUTS = {
    "play_pause": "space",
    "stop": "s",
    "rewind": "Home",
    "loop_toggle": "l",
    "crop": "c",
    "audition": "a",
    "undo": "Control-z",
    "redo": "Shift-Control-z",
    "zoom_in": "equal",
    "zoom_out": "minus",
    "zoom_fit": "0",
    "stretch": "x",
}

SHORTCUT_LABELS = {
    "play_pause": "Play / Pause",
    "stop": "Stop",
    "rewind": "Rewind",
    "loop_toggle": "Repeat (loop raw selection)",
    "crop": "Crop to Selection",
    "audition": "Loop (crossfade preview)",
    "undo": "Undo",
    "redo": "Redo",
    "zoom_in": "Zoom In",
    "zoom_out": "Zoom Out",
    "zoom_fit": "Zoom to Fit",
    "stretch": "PaulXStretch...",
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
    w = max(required_w, saved.get("width", required_w))
    h = max(required_h, saved.get("height", required_h))
    return w, h


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
        self.declick_remaining = 0    # samples of that ramp still left to apply

    def load(self, data, sr):
        self.stop()
        if data.ndim == 1:
            data = data[:, None]
        self.data = np.ascontiguousarray(data.astype(np.float32))
        self.sr = sr
        self.declick_total = max(1, int(sr * 0.02))  # 20ms fade-in, applied after any jump
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
            self.cursor = max(0, min(sample, len(self.data)))
            if self.playing:
                self._apply_bounds_from_cursor()
                # clicking to a new position while playing creates a sudden
                # amplitude discontinuity at the jump (the waveform doesn't
                # connect smoothly to wherever it was before) -- that's what
                # causes the audible pop. A very short fade-in smooths it out.
                self.declick_remaining = self.declick_total

    def set_loop(self, value):
        self.loop = bool(value)
        with self.lock:
            if self.data is not None and self.playing:
                self._apply_bounds_from_cursor()

    def _callback(self, outdata, frames, time_info, status):
        with self.lock:
            if self.data is None:
                outdata[:] = 0
                raise _sd.CallbackStop
            remaining = self.play_end - self.cursor
            if remaining <= 0:
                if self.play_loop:
                    self.cursor = self.play_start
                    remaining = self.play_end - self.cursor
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
            if self.declick_remaining > 0:
                # only applied here (the jump-origin chunk) -- deliberately
                # NOT applied to the loop-wrap continuation below, since a
                # raw/un-crossfaded loop's seam click is something the user
                # explicitly wants to still hear when previewing it
                ramp_n = min(n, self.declick_remaining)
                start_gain = 1.0 - self.declick_remaining / self.declick_total
                end_gain = 1.0 - (self.declick_remaining - ramp_n) / self.declick_total
                gains = np.linspace(start_gain, end_gain, ramp_n, endpoint=False).reshape(-1, 1)
                outdata[:ramp_n] *= gains
                self.declick_remaining -= ramp_n
            self.cursor += n
            if n < frames:
                if self.play_loop:
                    self.cursor = self.play_start
                    n2 = min(frames - n, self.play_end - self.play_start)
                    chunk2 = self.data[self.play_start:self.play_start + n2]
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
            self._apply_bounds_from_cursor()
            # smooths the start of playback the same way set_cursor() does --
            # covers both a fresh Play press and a restart triggered by
            # reprocessing (e.g. changing the manual crossfade value while
            # auditioning stops/reloads/replays, which is its own kind of
            # jump and had the same audible-pop problem)
            self.declick_remaining = self.declick_total
        channels = self.data.shape[1]
        self.stream = _sd.OutputStream(samplerate=self.sr, channels=channels,
                                        callback=self._callback, dtype="float32")
        self.stream.start()
        self.playing = True

    def swap_playing_buffer(self, data, sr):
        """Replaces the audio buffer WITHOUT stopping/restarting the actual
        OS audio stream, when one is already running at a matching sample
        rate/channel count. Used specifically for reprocessing WHILE
        already auditioning (e.g. changing the crossfade value mid-play) --
        stopping and recreating the whole PortAudio stream for that is a
        heavier operation that can leave already-queued audio from the OLD
        stream cut off abruptly, which is a SEPARATE click source from the
        new-content-onset click the declick ramp handles. Falls back to a
        plain stop+load (no play -- caller is expected to call play() in
        that case) if a live swap isn't possible."""
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
                self.cursor = 0
                self.play_start, self.play_end = 0, len(new_data)
                self.play_loop = self.loop
                self.declick_remaining = self.declick_total
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
        # calipers being pulled apart by a double-headed arrow
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
        draw.polygon([(x_left, cy), (base_left, cy - s * 0.68), (base_left, cy + s * 0.68)], fill=color)
        draw.polygon([(x_right, cy), (base_right, cy - s * 0.68), (base_right, cy + s * 0.68)], fill=color)
    elif name == "crop":
        w, L = max(3, int(sw * 0.17)), sw * 0.34
        draw.line([(pad, pad + L), (pad, pad), (pad + L, pad)], fill=color, width=w, joint="curve")
        draw.line([(sw - pad, sw - pad - L), (sw - pad, sw - pad), (sw - pad - L, sw - pad)],
                   fill=color, width=w, joint="curve")
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
        canvas_kwargs = {"height": height, "bg": bg, "highlightthickness": 0}
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
        self.canvas = tk.Canvas(self.frame, width=width, height=height, bg=bg, highlightthickness=0)
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
        x = self.canvas.winfo_rootx()
        y = self.canvas.winfo_rooty() + self.canvas.winfo_height()
        self.popup = tk.Toplevel(self.canvas)
        self.popup.wm_overrideredirect(True)
        row_h = 30
        self.popup.wm_geometry(f"{self.fixed_width}x{len(self.values) * row_h}+{x}+{y}")
        frame = tk.Frame(self.popup, bg=self.field_bg, highlightthickness=1, highlightbackground=self.border)
        frame.pack(fill="both", expand=True)
        for val in self.values:
            row = tk.Label(frame, text=val, bg=self.field_bg, fg=self.fg, anchor="w",
                            font=("Segoe UI", 10), padx=12, pady=6)
            row.pack(fill="x")
            row.bind("<Enter>", lambda e, r=row: r.configure(bg=self.accent))
            row.bind("<Leave>", lambda e, r=row: r.configure(bg=self.field_bg))
            row.bind("<Button-1>", lambda e, v=val: self._select(v))
        self.popup.bind("<FocusOut>", lambda e: self._close_popup())
        self.popup.focus_set()

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
        self.canvas = tk.Canvas(self.frame, width=width, height=height, bg=bg, highlightthickness=0)
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

    def __init__(self, widget, text, delay=500):
        import tkinter as tk
        self.tk = tk
        self.widget = widget
        self.text = text
        self.delay = delay
        self.tip = None
        self.after_id = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, event=None):
        self._cancel()
        self.after_id = self.widget.after(self.delay, self._show)

    def _cancel(self):
        if self.after_id is not None:
            try:
                self.widget.after_cancel(self.after_id)
            except Exception:
                pass
            self.after_id = None

    def _show(self):
        if self.tip is not None or not self.text:
            return
        try:
            x = self.widget.winfo_rootx() + 12
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        except Exception:
            return
        self.tip = self.tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        try:
            self.tip.wm_attributes("-topmost", True)
        except Exception:
            pass
        self.tip.wm_geometry(f"+{x}+{y}")
        label = self.tk.Label(self.tip, text=self.text, bg="#111214", fg="#e6e6e8",
                               font=("Segoe UI", 9), padx=8, pady=4,
                               relief="solid", borderwidth=1)
        label.pack()

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
        self.root.title("FermaLoop")
        self.root.configure(bg=BG)
        self.window_sizes = load_window_sizes()

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
        self.status_var = tk.StringVar(value="Drag an audio file onto this window, or click Browse.")
        self.time_var = tk.StringVar(value="00:00.000")
        self.selection_duration_var = tk.StringVar(value="Selection: --")
        self._click_flag = None       # (x_pixel, time_str) or None
        self._click_flag_after_id = None
        self._live_update_after_id = None
        self._canvas_tooltip = None

        for var in (self.xfade_var, self.curve_var, self.auto_xfade_var,
                    self.snap_var, self.window_var):
            var.trace_add("write", self._on_param_changed)
        self.format_var.trace_add("write", self._on_format_changed)

        self._build_widgets()
        self._bind_shortcuts()
        if DND_AVAILABLE:
            self._enable_drag_and_drop()

        self._apply_saved_or_natural_size()
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

    def _on_auto_detect_clicked(self):
        self.auto_xfade_var.set(True)
        self._rebuild_autodetect_manual_cards()
        self._on_param_changed()

    def _on_manual_clicked(self):
        self.auto_xfade_var.set(False)
        self._rebuild_autodetect_manual_cards()
        self._on_param_changed()

    def _rebuild_autodetect_manual_cards(self):
        """Auto-detect and Manual crossfade are mutually exclusive (one
        underlying bool, auto_xfade_var), presented as two 'cards' with a
        distinct background shade from the rest of the section -- whichever
        one is NOT currently active dims (muted colors) but stays fully
        clickable, rather than being disabled outright."""
        tk = self.tk
        for child in self.auto_col_holder.winfo_children():
            child.destroy()
        for child in self.manual_col_holder.winfo_children():
            child.destroy()

        auto_on = self.auto_xfade_var.get()
        CARD_BG_ACTIVE, CARD_BG_INACTIVE = FIELD_BG, "#1c1d20"
        FG_ACTIVE, FG_INACTIVE = FG, MUTED
        CHECK_ACTIVE, CHECK_INACTIVE = ACCENT, MUTED

        # ---- Auto-detect card ----
        a_bg = CARD_BG_ACTIVE if auto_on else CARD_BG_INACTIVE
        a_fg = FG_ACTIVE if auto_on else FG_INACTIVE
        a_check = CHECK_ACTIVE if auto_on else CHECK_INACTIVE
        auto_card = tk.Frame(self.auto_col_holder, bg=a_bg)
        auto_card.pack(fill="both", expand=True)
        auto_proxy = tk.BooleanVar(value=auto_on)
        auto_cb = RoundedCheckbutton(auto_card, "Auto-detect crossfade length", auto_proxy,
                                      a_bg, a_fg, a_bg, a_check, BORDER, command=self._on_auto_detect_clicked)
        auto_cb.pack(anchor="w", padx=10, pady=(10, 2))
        ToolTip(auto_cb.frame, "Automatically pick the crossfade length that best matches\n"
                                "the head and tail of the selection, instead of a fixed value")
        auto_value_fg = FG_ACTIVE if auto_on else FG_INACTIVE
        tk.Label(auto_card, textvariable=self.auto_xfade_value_var, bg=a_bg, fg=auto_value_fg,
                 font=("Segoe UI", 9)).pack(anchor="w", padx=(34, 10), pady=(0, 10))

        # ---- Manual card ----
        m_bg = CARD_BG_ACTIVE if not auto_on else CARD_BG_INACTIVE
        m_fg = FG_ACTIVE if not auto_on else FG_INACTIVE
        m_check = CHECK_ACTIVE if not auto_on else CHECK_INACTIVE
        manual_card = tk.Frame(self.manual_col_holder, bg=m_bg)
        manual_card.pack(fill="both", expand=True)
        manual_proxy = tk.BooleanVar(value=not auto_on)
        manual_cb = RoundedCheckbutton(manual_card, "Manual crossfade(s)", manual_proxy,
                                        m_bg, m_fg, m_bg, m_check, BORDER, command=self._on_manual_clicked)
        manual_cb.pack(anchor="w", padx=10, pady=(10, 4))
        ToolTip(manual_cb.frame, "Crossfade duration in seconds (used when auto-detect is off)")
        field_row = tk.Frame(manual_card, bg=m_bg)
        field_row.pack(anchor="w", padx=10, pady=(0, 10))
        entry_field_bg = FIELD_BG if not auto_on else "#232427"
        self.xfade_entry = RoundedEntry(field_row, self.xfade_var, m_bg, entry_field_bg, m_fg, BORDER,
                                         height=26, radius=7, width=64)
        self.xfade_entry.pack(side="left")
        self._defocus_on_return(self.xfade_entry.entry)

        self._update_auto_crossfade_preview()

    def _defocus_on_return(self, entry_widget):
        """Numeric entry fields (crossfade, search window, etc.) don't lose
        keyboard focus on their own after Return -- Tkinter Entry widgets
        just don't do that by default. Without this, typing a value and
        pressing Enter left the field focused, so a SUBSEQUENT press of
        Space (meant as the global Play/Pause shortcut) typed a literal
        space character into the field instead of toggling playback."""
        entry_widget.bind("<Return>", lambda e: self.root.focus_set())
        entry_widget.bind("<KP_Enter>", lambda e: self.root.focus_set())

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
        stop-and-restart or an unexpected jump."""
        photo = self._get_icon(icon_name, size)
        if photo is not None:
            btn = self.ttk.Button(parent, image=photo, style=style, command=command, takefocus=0)
        else:
            btn = self.ttk.Button(parent, text=tooltip_text, style=style, command=command, takefocus=0)
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

    def _start_loop_animation(self):
        if self._loop_anim_after_id is not None:
            return  # already running
        self._loop_anim_frames = self._get_loop_animation_frames()
        self._loop_anim_index = 0
        self._animate_loop_icon()

    def _animate_loop_icon(self):
        if not self.preview_mode or self._loop_anim_frames is None:
            self._loop_anim_after_id = None
            return
        self.btn_loop.configure(image=self._loop_anim_frames[self._loop_anim_index])
        self._loop_anim_index = (self._loop_anim_index + 1) % len(self._loop_anim_frames)
        self._loop_anim_after_id = self.root.after(90, self._animate_loop_icon)

    def _stop_loop_animation(self):
        if self._loop_anim_after_id is not None:
            try:
                self.root.after_cancel(self._loop_anim_after_id)
            except Exception:
                pass
            self._loop_anim_after_id = None
        icon = self._get_icon("loop", self.ICON_SIZE, FG)
        if icon is not None:
            self.btn_loop.configure(image=icon)

    def _refresh_repeat_icon(self):
        """Repeat's displayed color reflects its own state, EXCEPT while the
        Loop crossfade-preview is actively playing -- Loop supersedes plain
        Repeat (it loops too, plus applies the crossfade), so Repeat's own
        color is visually suppressed to grey for the duration without
        actually changing its stored value."""
        if self.preview_mode:
            icon = self._get_icon("repeat", self.ICON_SIZE, FG)
        else:
            color = ACCENT if self.repeat_var.get() else FG
            icon = self._get_icon("repeat", self.ICON_SIZE, color)
        if icon is not None:
            self.btn_repeat.configure(image=icon)

    # ---------------- widget layout ----------------

    def _build_widgets(self):
        tk, ttk = self.tk, self.ttk
        outer = ttk.Frame(self.root, padding=16)
        outer.pack(fill="both", expand=True)

        header_row = ttk.Frame(outer); header_row.pack(fill="x", pady=(0, 10))
        ttk.Label(header_row, text="FermaLoop", style="Heading.TLabel").pack(side="left")
        btn_gear = self._make_icon_button(header_row, "info", "Keyboard Shortcuts...",
                                           self.open_shortcuts_dialog, size=20)
        btn_gear.pack(side="right")

        # file row (rounded entries)
        row = ttk.Frame(outer); row.pack(fill="x", pady=3)
        ttk.Label(row, text="Input", width=7).pack(side="left")
        in_entry = RoundedEntry(row, self.in_path_var, BG, FIELD_BG, FG, BORDER)
        in_entry.pack(side="left", fill="x", expand=True, padx=6)
        ToolTip(in_entry.frame, "Path to the audio file to load")
        btn_browse_in = ttk.Button(row, text="Browse", command=self.choose_input, takefocus=0)
        btn_browse_in.pack(side="left")
        ToolTip(btn_browse_in, "Choose an audio file from disk")

        row = ttk.Frame(outer); row.pack(fill="x", pady=3)
        ttk.Label(row, text="Save as", width=7).pack(side="left")
        out_entry = RoundedEntry(row, self.out_path_var, BG, FIELD_BG, FG, BORDER)
        out_entry.pack(side="left", fill="x", expand=True, padx=6)
        ToolTip(out_entry.frame, "Where the processed loop will be saved")
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
        info_row = ttk.Frame(outer); info_row.pack(fill="x", pady=(4, 6))
        ttk.Label(info_row, textvariable=self.time_var, style="Muted.TLabel",
                  font=("Segoe UI", 22, "bold")).pack(side="left")
        ttk.Label(info_row, textvariable=self.selection_duration_var, style="Muted.TLabel").pack(side="right")

        ttk.Label(outer, text="Drag to select, drag edges to adjust, click to move the playhead. "
                               "Scroll to zoom, Shift+Scroll to pan left/right.",
                  style="Muted.TLabel").pack(anchor="w", pady=(0, 4))

        # transport
        transport = ttk.Frame(outer); transport.pack(fill="x", pady=(4, 4))

        self.btn_play = self._make_icon_button(transport, "play", "Play", self.on_play_pause, size=self.ICON_SIZE)
        self.btn_play.pack(side="left", padx=(0, 4))
        self._play_tooltip = self._last_tooltip

        self.btn_stop = self._make_icon_button(transport, "stop", "Stop", self.on_stop, size=self.ICON_SIZE)
        self.btn_stop.pack(side="left", padx=4)

        self.btn_repeat = self._make_icon_button(transport, "repeat",
                                                   "Repeat (loop the raw selection -- no crossfade, "
                                                   "you'll hear a click at the seam)",
                                                   self.on_repeat_toggle, size=self.ICON_SIZE)
        self.btn_repeat.pack(side="left", padx=4)

        self.btn_loop = self._make_icon_button(transport, "loop",
                                                 "Audition Loop (play the processed/crossfaded selection, "
                                                 "looped). Previews the current selection's crossfade "
                                                 "without saving or cropping.",
                                                 self.on_loop_preview, size=self.ICON_SIZE)
        self.btn_loop.pack(side="left", padx=(16, 4))

        self.btn_crop = self._make_icon_button(transport, "crop",
                                                 "Crop to Selection. Crop is only needed once you want to "
                                                 "commit the working range.",
                                                 self.on_crop, size=self.ICON_SIZE)
        self.btn_crop.pack(side="left", padx=4)

        btn_stretch = self._make_icon_button(transport, "stretch",
                                              "PaulXStretch: extreme time-stretch the current selection",
                                              self.open_stretch_dialog, size=self.ICON_SIZE)
        btn_stretch.pack(side="left", padx=4)

        ttk.Frame(outer, height=1, style="Panel.TFrame").pack(fill="x", pady=14)

        # ---- crossfade options: Curve alone at top, then a 3-column row
        # (Snap / Auto-detect / Manual), all inside one rounded, thin-
        # bordered container. Auto-detect and Manual are mutually
        # exclusive; whichever is inactive dims but stays clickable. ----
        section_outer, section_inner = self._make_rounded_section(outer, PANEL, BORDER, radius=12, padding=12)
        section_outer.pack(fill="x", pady=(4, 10))

        curve_row = ttk.Frame(section_inner, style="Panel.TFrame")
        curve_row.pack(fill="x", pady=(0, 10))
        curve_wrapper = ttk.Frame(curve_row, style="Panel.TFrame")
        curve_wrapper.pack(expand=True)
        curve_combo = RoundedDropdown(curve_wrapper, self.curve_var, ["Equal power", "Linear"],
                                       PANEL, FIELD_BG, FG, BORDER, ACCENT, height=28, radius=8, width=140)
        curve_combo.pack()
        ToolTip(curve_combo.frame, "Curve: shapes how the crossfade blends the two ends together.\n"
                                    "Equal power: smoother, constant perceived loudness through the fade.\n"
                                    "Linear: simpler ramp, can dip slightly in the middle.")

        cols_row = ttk.Frame(section_inner, style="Panel.TFrame")
        cols_row.pack(fill="x")

        snap_col = tk.Frame(cols_row, bg=PANEL)
        snap_col.pack(side="left", fill="both", expand=True, padx=(0, 6))
        snap_cb = RoundedCheckbutton(snap_col, "Snap loop points to transients",
                            self.snap_var, PANEL, FG, FIELD_BG, ACCENT, BORDER,
                            command=self._toggle_window_entry)
        snap_cb.pack(anchor="w", pady=(4, 0))
        ToolTip(snap_cb.frame, "Trim the selection to the strongest nearby attack at each end,\n"
                                "so the loop starts/ends on the beat instead of an arbitrary sample.\n"
                                "Works alongside either Auto-detect or Manual crossfade.")
        field_row = ttk.Frame(snap_col, style="Panel.TFrame")
        field_row.pack(anchor="w", pady=(4, 4))
        ttk.Label(field_row, text="Search window(s):", style="Muted.TLabel").pack(side="left", padx=(24, 6))
        self.window_entry = RoundedEntry(field_row, self.window_var, PANEL, FIELD_BG, FG, BORDER,
                                          height=28, radius=8, width=70)
        self.window_entry.pack(side="left")
        self.window_entry.configure(state="disabled")
        self._defocus_on_return(self.window_entry.entry)
        ToolTip(self.window_entry.frame, "How far from each end to search for a transient (seconds) -- "
                                          "auto-populated, override by typing a new value")

        self.auto_col_holder = tk.Frame(cols_row, bg=PANEL)
        self.auto_col_holder.pack(side="left", fill="both", expand=True, padx=4)
        self.manual_col_holder = tk.Frame(cols_row, bg=PANEL)
        self.manual_col_holder.pack(side="left", fill="both", expand=True, padx=(4, 0))
        self._rebuild_autodetect_manual_cards()

        self._finalize_rounded_section(section_outer)

        btn_process = ttk.Button(outer, text="Process & Save", style="Accent.TButton",
                   command=self.run_process, takefocus=0)
        btn_process.pack(fill="x", pady=(10, 10))
        ToolTip(btn_process, "Crossfade the current selection and save it to the 'Save as' path")

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
            self.out_path_var.set(base + ext)
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
        self._refresh_loop_and_repeat_icons()
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.player.load(data, sr)

        self.in_path_var.set(path)
        # auto-fill Save As to the same directory as the loaded file, using
        # the currently-selected output FORMAT's extension (not necessarily
        # the input file's own extension, since only FLAC/MP4/MP3 are
        # offered as save targets)
        root_name, orig_ext = os.path.splitext(path)
        out_ext = FORMAT_EXT.get(self.format_var.get(), orig_ext)
        self.out_path_var.set(os.path.normpath(root_name + "_loop" + out_ext))

        self._click_flag = None
        self._redraw()
        self._update_selection_duration_label()
        self._update_auto_crossfade_preview()
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
        self._refresh_loop_and_repeat_icons()
        self.player.load(self.data, self.sr)
        self.player.set_selection(self.sel_start, self.sel_end)
        self._redraw()
        self._update_selection_duration_label()
        self._update_auto_crossfade_preview()

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
        self.canvas.create_text(w // 2, h // 2, text="Drag & drop audio file, select a region to loop",
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
        self.in_path_var.set("")
        self.out_path_var.set("")
        self.time_var.set("00:00.000")
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

        sx = self._sample_to_x(self.sel_start, w)
        ex = self._sample_to_x(self.sel_end, w)
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

    def _on_canvas_release(self, event):
        if self.data is None:
            self.drag_mode = None
            return
        if self.drag_mode == "pending":
            # a plain click (no meaningful drag): move the playhead there
            w = self.canvas.winfo_width()
            samp = self._x_to_sample(event.x, w)
            samp = max(0, min(samp, len(self.data)))
            if self.preview_mode and self.sel_start <= samp < self.sel_end:
                # clicking WITHIN the loop region while auditioning: stay in
                # preview mode (don't fall back to raw/unprocessed audio) --
                # this is what lets you scrub right up to the loop-back
                # point and hear the actual crossfaded wrap
                preview_cursor = self._raw_to_preview_cursor(samp)
                self.player.set_cursor(preview_cursor)
            else:
                # clicking outside the loop region: always operate on raw
                # audio, so you can freely check surrounding context
                self._exit_preview_mode()
                self.player.set_cursor(samp)
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
        """Keeps the Loop button's animation state and the Repeat button's
        suppressed/normal color in sync with self.preview_mode. Loop's
        button background never changes -- only its icon (grey static vs
        blue animated) communicates state, matching the confirmed design."""
        if self.preview_mode:
            self._start_loop_animation()
        else:
            self._stop_loop_animation()
        self._refresh_repeat_icon()

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
            # loop region to check surrounding context mid-audition
            self.player.swap_playing_buffer(self.data, self.sr)
            self.player.set_loop(self.repeat_var.get())
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
        else:
            if not self.preview_mode:
                # only touch selection/loop when starting fresh RAW playback;
                # while a processed Loop preview is paused, resuming it must
                # reuse the player's own (already-correct) internal bounds,
                # not overwrite them with raw file-space selection indices
                self.player.set_selection(self.sel_start, self.sel_end)
                self.player.set_loop(self.repeat_var.get())
            self.player.play()
            self._set_play_pause_icon(True)

    def on_stop(self):
        self._flash_button(self.btn_stop)
        self.player.stop()
        self._set_play_pause_icon(False)
        self._redraw()

    def on_rewind(self):
        self.player.rewind()
        self._redraw()

    def on_repeat_toggle(self):
        self.repeat_var.set(not self.repeat_var.get())
        self.player.set_loop(self.repeat_var.get())
        self._refresh_repeat_icon()

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
                f"Click Audition Loop to preview it, or Process & Save to use it."
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
        isn't ready to process."""
        if self.data is None:
            return

        if self.preview_mode and not silent:
            self._exit_preview_mode()
            self.player.stop()
            self._set_play_pause_icon(False)
            self.status_var.set("Stopped auditioning.")
            self._redraw()
            return

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
                self.player.set_loop(True)
                self.player.play()
            self.preview_mode = True
            self._refresh_loop_and_repeat_icons()
            self._set_play_pause_icon(True)

            dur = preview.shape[0] / self.sr
            self.status_var.set(
                f"Looping the processed preview ({dur:.2f}s, crossfade {used_xfade*1000:.0f} ms, "
                f"computed in {elapsed*1000:.0f} ms). Click within the selection to scrub, adjust "
                f"settings to update live, or click Audition Loop again to stop."
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
        self.push_undo()
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
        self._defocus_on_return(factor_entry.entry)
        ttk.Label(dlg, text="Typical range: 2-50 (higher takes longer and produces much longer output)",
                  background=BG, foreground=MUTED, font=("Segoe UI", 8)).pack(anchor="w", padx=14, pady=(2, 8))

        row = ttk.Frame(dlg); row.pack(fill="x", padx=14, pady=(4, 0))
        ttk.Label(row, text="Window size:", background=BG, foreground=FG).pack(side="left")
        window_var = tk.StringVar(value="0.25")
        stretch_window_entry = RoundedEntry(row, window_var, BG, FIELD_BG, FG, BORDER, height=28, radius=8, width=80)
        stretch_window_entry.pack(side="right")
        self._defocus_on_return(stretch_window_entry.entry)
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

                self.push_undo()
                self.player.stop()
                self.data = np.concatenate([self.data[:s], stretched, self.data[e:]], axis=0)
                self.sel_start = s
                self.sel_end = s + stretched.shape[0]
                self.zoom_start, self.zoom_end = 0, len(self.data)
                self.preview_mode = False
                self._refresh_loop_and_repeat_icons()
                self._set_play_pause_icon(False)
                self.time_var.set("00:00.000")
                self.player.load(self.data, self.sr)
                self._redraw()
                self._update_selection_duration_label()
                self._update_auto_crossfade_preview()
                self.zoom_to_selection()

                out_dur = stretched.shape[0] / self.sr
                self.status_var.set(
                    f"Stretched {sel_dur:.2f}s to {out_dur:.2f}s ({factor:.1f}x) in {elapsed:.1f}s. "
                    f"(Cmd/Ctrl+Z to undo.) Audition or Crop to build the loop."
                )
                close_dialog()
            except Exception as ex:
                result_label.configure(text=f"Failed: {ex}")

        btn_row = ttk.Frame(dlg); btn_row.pack(fill="x", padx=14, pady=14)
        ttk.Button(btn_row, text="Cancel", command=close_dialog).pack(side="right", padx=(6, 0))
        ttk.Button(btn_row, text="Stretch", style="Accent.TButton", command=apply_stretch).pack(side="right")

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

    def _poll_playhead(self):
        if self.data is not None and self.player.playing:
            self._redraw()
            display_samp = self._display_cursor_sample()
            self.time_var.set(format_time(display_samp / self.sr))
            if not self.player.playing:
                self._set_play_pause_icon(False)
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
            "stretch": self.open_stretch_dialog,
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

        ttk.Button(dlg, text="Done", command=lambda: on_close(), style="Accent.TButton").grid(
            row=len(rows), column=0, columnspan=2, pady=16, padx=10, sticky="ew")

        dlg.update_idletasks()
        required_w, required_h = dlg.winfo_reqwidth(), dlg.winfo_reqheight()
        saved = self.window_sizes.get("shortcuts", {})
        w, h = resolve_window_size(required_w, required_h, saved)
        dlg.minsize(required_w, required_h)

        # ---- position: docks to the main window's right edge, computed
        # fresh every time the dialog opens (no persistence -- see below).
        #
        # NOTE: this is the third attempt at this and it's still landing in
        # the wrong place, which rules out my working theory from the last
        # attempt (a corrupted persisted offset) since that version had NO
        # persistence at all and still failed. That means either winfo_x()/
        # winfo_y() are returning unexpected values on your system, or the
        # window manager is ignoring the geometry request outright -- and I
        # have no way to test real Tk window behavior on Windows from here,
        # so a fourth blind guess isn't a responsible use of your time.
        # Trying winfo_rootx()/rooty() instead (a genuinely different Tk
        # API, not just a reworded version of the same call), AND printing
        # the actual numbers being computed into the dialog itself -- if
        # it's STILL wrong, please tell me exactly what that debug line
        # says. That tells us definitively whether the calculation itself
        # is wrong (fixable) or whether the OS is overriding a correct
        # request outright (would need a different strategy entirely,
        # like giving up on auto-positioning and just remembering the
        # LAST place you manually dragged it to).
        root_x, root_y = self.root.winfo_rootx(), self.root.winfo_rooty()
        root_w = self.root.winfo_width()
        x = root_x + root_w + 10
        y = root_y
        dlg.geometry(f"{w}x{h}+{x}+{y}")
        dlg.after(20, lambda: dlg.geometry(f"{w}x{h}+{x}+{y}"))

        debug_text = (f"debug: root=({root_x},{root_y}) w={root_w}  ->  target=({x},{y})  "
                      f"[also winfo_x/y=({self.root.winfo_x()},{self.root.winfo_y()})]")
        ttk.Label(dlg, text=debug_text, background=BG, foreground=MUTED,
                  font=("Segoe UI", 7)).grid(row=len(rows) + 1, column=0, columnspan=2,
                                              sticky="ew", padx=10, pady=(0, 8))

        def on_close():
            save_shortcuts(self.shortcuts)
            try:
                self.window_sizes["shortcuts"] = {
                    "width": dlg.winfo_width(), "height": dlg.winfo_height(),
                }
                save_window_sizes(self.window_sizes)
            except Exception:
                pass
            dlg.destroy()

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
            result, used_xfade, start_trim, end_trim = _run_pipeline(
                segment, self.sr, xfade_seconds, curve, snap, window, self.auto_xfade_var.get(),
            )
            encode_from_pcm(result, self.sr, self.sampwidth, out_path,
                             mp3_quality=int(round(self.mp3_quality_var.get())))
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
            self.messagebox.showerror("FermaLoop", str(e))

    # ---------------- window sizing ----------------

    def _apply_saved_or_natural_size(self):
        """Sizes the window to fit everything on first paint (no manual
        resize needed), while respecting a larger size the user may have
        deliberately set last time the app was open."""
        self.root.update_idletasks()
        required_w = self.root.winfo_reqwidth()
        required_h = self.root.winfo_reqheight()
        saved = self.window_sizes.get("main")
        w, h = resolve_window_size(required_w, required_h, saved)
        self.root.geometry(f"{w}x{h}")
        self.root.minsize(required_w, required_h)

    def _on_close(self):
        try:
            self.window_sizes["main"] = {
                "width": self.root.winfo_width(),
                "height": self.root.winfo_height(),
            }
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
