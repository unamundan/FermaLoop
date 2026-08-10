#!/usr/bin/env python3
"""
Adds version/author/copyright metadata to a PyInstaller-built .app
bundle's Info.plist -- without this, macOS's "Get Info" panel shows
nothing meaningful for version, author, or copyright, since PyInstaller's
default generated Info.plist doesn't include any of these fields.

Reads APP_VERSION/APP_AUTHOR/APP_COPYRIGHT/APP_DESCRIPTION out of
FermaLoop.py via the same plain-text regex approach as
generate_version_info.py (the Windows equivalent), so both platforms'
metadata come from the exact same source and can't drift out of sync.

Usage: python update_macos_plist.py FermaLoop.py path/to/FermaLoop.app
"""
import re
import sys
import os
import ast
import plistlib


def extract_constant(source, name, default):
    # see the matching function in generate_version_info.py for why this
    # captures the full quoted literal and uses ast.literal_eval rather
    # than a raw regex-captured inner string -- unlike the Windows
    # version-info file (which gets re-parsed as Python by PyInstaller
    # later, so a raw "\u00a9" text sequence happens to still work out),
    # this value gets written directly into the plist as final data, so
    # an un-evaluated escape sequence would end up wrong in the actual
    # shipped Info.plist with no error to catch it.
    match = re.search(rf'^{name}\s*=\s*(["\'].+?["\'])', source, re.MULTILINE)
    if not match:
        return default
    try:
        return ast.literal_eval(match.group(1))
    except (ValueError, SyntaxError):
        return default


def main():
    if len(sys.argv) != 3:
        print("Usage: python update_macos_plist.py <source.py> <path/to/App.app>")
        sys.exit(1)
    source_path, app_path = sys.argv[1], sys.argv[2]

    with open(source_path, "r", encoding="utf-8") as f:
        source = f.read()

    version = extract_constant(source, "APP_VERSION", "0.0.0")
    author = extract_constant(source, "APP_AUTHOR", "Unknown")
    copyright_str = extract_constant(source, "APP_COPYRIGHT", f"\u00a9 {author}")

    plist_path = os.path.join(app_path, "Contents", "Info.plist")
    with open(plist_path, "rb") as f:
        plist = plistlib.load(f)

    # CFBundleShortVersionString: the human-facing version ("1.0.0"),
    # what actually shows in Finder's Get Info panel as "Version"
    plist["CFBundleShortVersionString"] = version
    # CFBundleVersion: the internal build-version string macOS requires
    # to be present and monotonically comparable across updates -- reuse
    # the same value, since this app doesn't track separate build numbers
    plist["CFBundleVersion"] = version
    # NSHumanReadableCopyright: shown in Get Info as "Copyright"
    plist["NSHumanReadableCopyright"] = copyright_str

    with open(plist_path, "wb") as f:
        plistlib.dump(plist, f)

    print(f"Updated {plist_path}: version={version}, copyright={copyright_str!r}")


if __name__ == "__main__":
    main()
