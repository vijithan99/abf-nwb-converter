# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 10:22:44 2026

@author: vijit
"""

from pathlib import Path
import sys
from nwbinspector import inspect_nwbfile

INPUT_FOLDER = Path.cwd() / "nwbData"

def main():
    nwb_files = sorted(path for path in INPUT_FOLDER.rglob("*") if path.is_file() and path.suffix.lower() == ".nwb")

    if not nwb_files:
        print(f"No NWB files found under: {INPUT_FOLDER}")
        return

    print(f"Found {len(nwb_files)} NWB files.\n")

    for index, nwb_path in enumerate(nwb_files, start=1):
        relative_path = nwb_path.relative_to(INPUT_FOLDER)

        print("=" * 80)
        print(f"[{index}/{len(nwb_files)}] {relative_path}")

        try:
            issues = list(
                inspect_nwbfile(
                    nwbfile_path=nwb_path
                )
            )

            if not issues:
                print("PASS: No NWBInspector issues found.")
                continue

            for issue in issues:
                importance = issue.importance.name
                location = issue.location or "/"
                check_name = issue.check_function_name or "unknown_check"

                print(
                    f"{importance}\n"
                    f"  Check: {check_name}\n"
                    f"  Location: {location}\n"
                    f"  Message: {issue.message}\n"
                )

        except Exception as exc:
            print(
                f"FAILED {nwb_path}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    print("\nInspection complete.")


if __name__ == "__main__":
    main()