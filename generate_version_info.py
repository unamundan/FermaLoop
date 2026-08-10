#!/usr/bin/env python3
"""
Generates the Windows version-resource file PyInstaller's --version-file
option expects (the VSVersionInfo(...) structure shown in PyInstaller's
own docs), reading APP_VERSION/APP_AUTHOR/APP_COPYRIGHT/APP_DESCRIPTION
directly out of FermaLoop.py via a plain text regex -- deliberately NOT
importing the module, since that would require every runtime dependency
(numpy, Pillow, sounddevice, tkinterdnd2) to already be installed just to
read four string constants.

Usage: python generate_version_info.py FermaLoop.py version_info.txt
"""
import re
import sys
import ast


def extract_constant(source, name, default):
    # matches: NAME = "..." or NAME = '...' -- deliberately does NOT
    # anchor to end-of-line after the closing quote, since a trailing
    # inline comment (e.g. "APP_AUTHOR = \"X\"  # TODO: ...") would
    # otherwise silently fail to match and fall back to the default
    # without any error, which is exactly what happened here initially.
    #
    # Captures the FULL quoted literal (quotes included) and parses it
    # with ast.literal_eval rather than just grabbing the inner text --
    # a raw-text regex match on APP_COPYRIGHT = "\u00a9 2026 X" would
    # otherwise capture the literal six characters backslash-u-0-0-a-9
    # instead of the actual (c) symbol, since regex doesn't interpret
    # Python escape sequences the way executing the source file would.
    match = re.search(rf'^{name}\s*=\s*(["\'].+?["\'])', source, re.MULTILINE)
    if not match:
        return default
    try:
        return ast.literal_eval(match.group(1))
    except (ValueError, SyntaxError):
        return default


def version_tuple(version_str):
    """'1.0.0' -> (1, 0, 0, 0); Windows file-version resources are always
    a 4-part integer tuple, padding with zeros/truncating as needed."""
    parts = [p for p in re.split(r"[.\-+]", version_str) if p.isdigit()]
    parts = [int(p) for p in parts[:4]]
    while len(parts) < 4:
        parts.append(0)
    return tuple(parts)


def main():
    if len(sys.argv) != 3:
        print("Usage: python generate_version_info.py <source.py> <output.txt>")
        sys.exit(1)
    source_path, output_path = sys.argv[1], sys.argv[2]

    with open(source_path, "r", encoding="utf-8") as f:
        source = f.read()

    version = extract_constant(source, "APP_VERSION", "0.0.0")
    author = extract_constant(source, "APP_AUTHOR", "Unknown")
    copyright_str = extract_constant(source, "APP_COPYRIGHT", f"\u00a9 {author}")
    description = extract_constant(source, "APP_DESCRIPTION", "FermaLoop")

    vt = version_tuple(version)
    file_version_str = ".".join(str(p) for p in vt)

    content = f'''# UTF-8
VSVersionInfo(
  ffi=FixedFileInfo(
    filevers={vt!r},
    prodvers={vt!r},
    mask=0x3f,
    flags=0x0,
    OS=0x4,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo(
      [
      StringTable(
        u'040904B0',
        [StringStruct(u'CompanyName', u'{author}'),
        StringStruct(u'FileDescription', u'{description}'),
        StringStruct(u'FileVersion', u'{file_version_str}'),
        StringStruct(u'InternalName', u'FermaLoop'),
        StringStruct(u'LegalCopyright', u'{copyright_str}'),
        StringStruct(u'OriginalFilename', u'FermaLoop.exe'),
        StringStruct(u'ProductName', u'FermaLoop'),
        StringStruct(u'ProductVersion', u'{version}')])
      ]),
    VarFileInfo([VarStruct(u'Translation', [1033, 1200])])
  ]
)
'''

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Generated {output_path} for version {version} (file version {file_version_str})")


if __name__ == "__main__":
    main()
