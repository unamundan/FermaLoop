# FermaLoop — Development Guide

Technical reference for `FermaLoop.py`: architecture, dependencies,
build process, and known limitations. If you're looking for what the
app *does* and how to use it rather than how it's built, see
[`README.md`](README.md) for a full walkthrough, or the in-app "Prefs
and Help" window for a quicker in-context reference.

---

## Architecture

Everything lives in one file, `FermaLoop.py` (~6,150 lines), split into
three layers that don't depend on the GUI existing at all:

1. **DSP core** — pure functions operating on NumPy arrays: WAV read/write,
   ffmpeg-backed decode/encode for other formats, the crossfade algorithm,
   transient detection, and the PaulStretch time-stretcher. None of this
   imports Tkinter. `process_file()` is the single entry point that wires
   these together, and is what both the GUI and the CLI actually call.
2. **GUI** — a hand-rolled dark theme built entirely on `tk`/`ttk`, with
   custom widgets (`RoundedEntry`, `RoundedCheckbutton`, `RoundedRadio`,
   `RoundedDropdown`, `RoundedSlider`) rendered via Pillow rather than
   using ttk's native styling, since ttk's cross-platform theming can't
   express fully custom rounded/anti-aliased shapes. Falls back to plain
   Tk widgets if Pillow isn't installed.
3. **CLI** — `argparse`-based, sharing the exact same `process_file()`
   pipeline as the GUI. Running the script with arguments skips the GUI
   entirely (see [CLI usage](#cli-usage) below).

There's no separate build system, package structure, or requirements.txt
by design — this is meant to be readable and hackable as a single file,
not a distributed package. If it grows further, the DSP core is the part
most likely to be worth splitting out.

### Playback state design

`export_mode` (`"raw"` / `"repeat"` / `"loop"`) is the single source of
truth for what **Save** actually exports — deliberately kept separate
from `preview_mode` (whether the player currently holds a live, computed
LOOP buffer) and `player.playing` (whether the audio stream is actually
running). REPEAT and LOOP never start or stop playback on their own:
pressing either only arms or live-switches the mode, hot-swapping
already-playing audio if something's already running. **Space** and
**Stop** are the only controls that actually start or stop the stream.

This is worth preserving if you're modifying this code, not just an
implementation detail: an earlier version had REPEAT/LOOP force-start or
force-stop playback as a side effect of toggling, which repeatedly
produced subtle, hard-to-reproduce state bugs — stale mode flags
surviving a toggle, transport icons and status messages falling out of
sync with actual playback state, that kind of thing — across several
rounds of fixes before landing on the current, stricter separation.

### Key algorithms

- **Loop crossfade** (`loop_crossfade`, `_crossfade_core`): blends the
  tail into the head using either an equal-power (sin/cos) or linear
  fade curve. `auto_select_xfade` picks a crossfade length automatically
  by minimizing a cost function (`_seam_cost`) that combines waveform
  correlation, RMS energy matching, and a length penalty — shorter
  crossfades are preferred when they sound equally good, since they
  preserve more of the original transient content.
- **Transient detection** (`find_strongest_transient`, `snap_to_transients`):
  a simple onset-strength approach (`_onset_frames`) over short analysis
  frames, used to snap loop points to the nearest strong attack instead
  of an arbitrary sample boundary.
- **PaulXStretch** (`paulstretch`, `_paulstretch_mono`): an implementation
  of [Nasca Octavian Paul](https://www.paulnasca.com/)'s
  ["Paul's Extreme Sound Stretch"](https://hypermammut.sourceforge.net/paulstretch/)
  technique — randomizes FFT bin phase per analysis frame while
  preserving magnitude, which destroys phase coherence (rhythm and
  transients smear into texture) but avoids the comb-filtering/metallic
  artifacts that phase vocoders and granular stretchers suffer at
  extreme ratios. The feature name specifically nods to
  [Xenakios' PaulXStretch](https://sonosaurus.com/paulxstretch/)
  ([source](https://github.com/essej/paulxstretch)), a modern
  continuation of the original tool. This is a from-scratch
  implementation of the *algorithm* — no code is shared with either
  project — not a port of either application, and doesn't replicate
  their full feature sets (spectral filtering, per-band controls, etc.),
  just the core stretch technique. Suited to ambient/drone/texture
  material, not rhythmic content — the smearing is inherent to the
  technique.

---

## Dependencies

| Package | Required? | What breaks without it |
|---|---|---|
| `numpy` | Yes | Everything — the DSP core is NumPy-array-based throughout |
| Tkinter | Yes (GUI only) | Ships with standard python.org installers on Windows/macOS; may need a separate OS package on Linux (e.g. `python3-tk`) |
| `ffmpeg` | Yes, for non-WAV formats | WAV still works without it; AIFF/MP3/MP4/FLAC decode+encode both require it. The built executables bundle a static ffmpeg binary so end users don't need it installed separately — see `_find_ffmpeg()`, which checks for a bundled copy first, then falls back to `PATH` |
| `Pillow` (PIL) | No | Falls back to plain Tk widgets and simple line-drawn waveforms instead of the custom rounded/anti-aliased UI |
| `sounddevice` | No | Play/Pause, Stop, Rewind, REPEAT, and LOOP preview all become inert; Crop, PaulXStretch, file processing, and the CLI are unaffected, since none of them depend on actually playing audio |
| `tkinterdnd2` | No | Drag-and-drop file loading is disabled; the Browse buttons still work |

Everything except `numpy` and Tkinter degrades gracefully rather than
crashing at import time — the app checks `..._AVAILABLE` flags for each
optional dependency and disables just the affected feature.

Install for development:
```
pip install numpy Pillow sounddevice tkinterdnd2
```

---

## Running from source

```
python FermaLoop.py
```
opens the GUI. Any arguments switch to headless CLI mode instead:

```
python FermaLoop.py input.wav output.wav --auto-xfade --snap-transients
```

### CLI usage

```
usage: FermaLoop.py [-h] [--xfade XFADE] [--auto-xfade]
                    [--curve {equal_power,linear}] [--snap-transients]
                    [--transient-window TRANSIENT_WINDOW]
                    [--mp3-quality {0,1,2,3,4,5,6,7,8,9}]
                    input output
```
`--xfade` and `--auto-xfade` are mutually exclusive in effect (explicit
`--xfade` wins); if neither is given, `--auto-xfade` is assumed.

---

## Building the executables

`.github/workflows/build.yml` produces self-contained Windows and macOS
builds via PyInstaller, triggered on push to `main` or manually from the
Actions tab. Both bundle ffmpeg internally.

A few build-specific decisions worth knowing about if you're touching
this file:

- **`--onedir`, not `--onefile`.** `--onefile` re-extracts the entire
  bundled Python runtime to a fresh temp directory on *every launch*,
  which measurably delayed startup (worse on macOS than Windows) and is
  also a well-documented trigger for antivirus false-positive heuristics,
  since self-extracting compressed executables are a common malware
  pattern. `--onedir` ships everything already unpacked.
- **Version metadata** comes from `APP_VERSION`/`APP_AUTHOR`/
  `APP_COPYRIGHT`/`APP_DESCRIPTION` near the top of `FermaLoop.py` — the
  single source of truth for both platforms. `generate_version_info.py`
  turns these into the Windows `.exe` version resource (populates the
  "Details" tab in File Properties); `update_macos_plist.py` patches the
  built `.app`'s `Info.plist` after the fact (populates "Get Info").
  Neither script imports the app itself — they read the constants via a
  plain-text regex, so building doesn't require every runtime dependency
  to already be installed just to extract four strings.
- **ffmpeg** is downloaded fresh per build from BtbN/FFmpeg-Builds
  (Windows, with a gyan.dev fallback) and evermeet.cx (macOS) — not
  vendored into the repo.

---

## Supported operating systems

**Honest answer: this isn't independently pinned or verified against a
specific minimum OS version, and that's worth understanding rather than
taking a number at face value.**

The workflow builds against GitHub's `macos-latest` and `windows-latest`
runner labels, which are moving targets — GitHub periodically repoints
them at newer OS images, and that has happened recently and specifically
enough to matter here: `macos-latest` was in the middle of migrating from
macOS 15 (Sequoia) to macOS 26 (Tahoe) as of mid-2026, and `windows-latest`
similarly moved from a Visual Studio 2022/Windows Server 2022 baseline to
Windows Server 2025 with Visual Studio 2026 around the same period. That
means the actual minimum OS version a built binary can run on can shift
underneath the workflow without any change to this repo at all.

In practice, Python/PyInstaller-built executables are usually
backward-compatible well beyond the exact OS version that built them —
but "usually" isn't a substitute for testing. If you need a *guaranteed*
minimum supported version:

- Pin the runner explicitly instead of using `-latest` (e.g.
  `runs-on: macos-15` or `runs-on: windows-2022`) to stop the floor from
  moving without your knowledge, and
- Actually run the built artifact on the oldest OS version you want to
  claim support for, and update this section with what you confirm.

**Linux**: the source runs under Linux Tkinter with no code changes
needed (there's nothing platform-specific in the DSP core, and the GUI
only branches on `sys.platform` for a handful of macOS-specific things —
see `IS_MACOS` — none of it Linux-exclusionary). `build.yml` does **not**
currently produce a Linux build; there's no packaged binary to download
for Linux yet. If you want one, adding a `build-linux` job following the
same PyInstaller pattern as the other two jobs is the place to start.

---

## Known caveats

- **macOS Gatekeeper**: the bundled ffmpeg binary and the app itself are
  unsigned. First launch may trigger Gatekeeper's unidentified-developer
  warning (right-click → Open bypasses it once; code-signing and
  notarization would remove this but require an Apple Developer account).
- **Windows SmartScreen / antivirus**: unsigned PyInstaller executables
  are commonly flagged by Windows Defender and other AV heuristics — see
  the `--onedir` note above. This is a widely-documented PyInstaller
  community issue, not specific to this app, and reporting the false
  positive to the AV vendor is generally the only real fix short of code
  signing.
- **macOS: a brief Finder flash on first launch or first file access**:
  the first time in a login session that FermaLoop's drag-and-drop
  support initializes (`tkinterdnd2`, used to construct the root window
  itself) or its native file-open dialog is used, macOS may briefly
  show/close Finder windows tied to network-mounted volumes while it
  enumerates all mounted drives. This is a one-time OS-level cost paid
  by whichever app is first to touch these particular Cocoa APIs in a
  session — not something specific to FermaLoop, and not a sign of
  instability. It's cached by the OS and doesn't recur on subsequent
  launches until the next login/reboot.
- **Playback stream configuration**: `AudioPlayer` opens its
  `sounddevice.OutputStream` using the OS/PortAudio's own default
  latency and blocksize, not an aggressive explicit setting. An earlier,
  more demanding configuration (`blocksize=256`, `latency="low"`) was
  confirmed via direct A/B testing to cause an intermittent popping
  sound during LOOP/REPEAT playback and was removed as the default; it
  remains only as a fallback if the OS can't open a stream with default
  settings at all. The trade-off: a mid-playback jump (an explicit seek
  or click) may have a very slightly longer window before the declick
  ramp's new content reaches the speaker than the old low-latency
  setting provided — a narrow, one-time-per-seek concern, judged less
  disruptive than a pop on every loop cycle during ordinary playback.
- **PaulXStretch is a from-scratch reimplementation** of the underlying
  algorithm, not a port of the reference application — see the algorithm
  note above.

---

## License

MIT — see [`LICENSE`](LICENSE). Free to use, modify, fork, and
distribute, including commercially, with attribution — one of the most
permissive and by far the most widely used open source license in
existence.

Third-party dependencies (numpy, Pillow, sounddevice, tkinterdnd2) all
carry their own separate, permissive licenses and aren't affected by
this project's license choice; ffmpeg is invoked as a separate
subprocess, not linked into this codebase, so its own license terms
don't extend to this project either.
