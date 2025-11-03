#!/usr/bin/env python3
"""
Compile 'bare' and 'makeup_04' images into separate subfolders.

Input structure:
PARENT/
  000000/
    bare.jpg
    makeup_01.jpg
    makeup_02.jpg
    makeup_03.jpg
    makeup_04.jpg
    makeup_05.jpg
  000001/
    ...

Output structure:
OUTPUT/
  bare/
    0001_bare_000000.jpg
    0002_bare_000001.png
    ...
  makeup_04/
    0001_makeup_04_000000.jpg
    0002_makeup_04_000001.png
    ...
"""

from pathlib import Path
import shutil
import sys

# -------------------- CONFIG --------------------
PARENT_DIR = Path("E:\FFHQ-Makeup\FFHQ-Makeup")   # <- change this
OUTPUT_DIR = Path("./compiled_output_10k")             # <- change if you like
SAMPLE_COUNT = 10000                                 # <- how many subject folders to take
REQUIRE_ALL_6 = False                              # enforce all 6 pics exist?
USE_SYMLINKS = False                               # symlink instead of copy
# ------------------------------------------------

REQUIRED_KEYS = ["bare", "makeup_04"]
ALL_KEYS = ["bare", "makeup_01", "makeup_02", "makeup_03", "makeup_04", "makeup_05"]
GLOB_EXTS = ["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"]


def find_first_with_any_ext(folder: Path, stem: str) -> Path | None:
    """Return first file that matches stem with any known extension."""
    for ext in GLOB_EXTS:
        for candidate in [folder / f"{stem}.{ext}", folder / f"{stem}.{ext.upper()}"]:
            if candidate.exists():
                return candidate
    matches = list(folder.glob(f"{stem}.*"))
    return matches[0] if matches else None


def folder_has_required(folder: Path, require_all_6: bool = False) -> bool:
    for k in REQUIRED_KEYS:
        if find_first_with_any_ext(folder, k) is None:
            return False
    if require_all_6:
        for k in ALL_KEYS:
            if find_first_with_any_ext(folder, k) is None:
                return False
    return True


def list_candidate_folders(parent: Path) -> list[Path]:
    if not parent.exists():
        sys.exit(f"Parent folder not found: {parent}")
    subs = [p for p in parent.iterdir() if p.is_dir() and p.name.isdigit()]
    subs.sort(key=lambda p: int(p.name))
    return [p for p in subs if folder_has_required(p, REQUIRE_ALL_6)]


def safe_copy(src: Path, dst: Path, symlink: bool = False):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if symlink:
        try:
            if dst.exists():
                dst.unlink()
            dst.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dst)
    else:
        shutil.copy2(src, dst)


def main():
    candidates = list_candidate_folders(PARENT_DIR)
    print(f"Found {len(candidates)} valid folders.")

    if len(candidates) < SAMPLE_COUNT:
        sys.exit(f"Not enough valid folders: requested {SAMPLE_COUNT}, found {len(candidates)}.")

    picked = candidates[:SAMPLE_COUNT]
    print(f"Selected {len(picked)} folders (sequential).")

    # Output subfolders
    bare_dir = OUTPUT_DIR / "bare"
    makeup_dir = OUTPUT_DIR / "makeup_04"
    bare_dir.mkdir(parents=True, exist_ok=True)
    makeup_dir.mkdir(parents=True, exist_ok=True)

    # Copy 'bare'
    print("Copying 'bare' images...")
    for idx, folder in enumerate(picked, start=1):
        src = find_first_with_any_ext(folder, "bare")
        if src:
            out_name = f"{idx:04d}_bare_{folder.name}{src.suffix}"
            safe_copy(src, bare_dir / out_name, symlink=USE_SYMLINKS)

    # Copy 'makeup_04'
    print("Copying 'makeup_04' images...")
    for idx, folder in enumerate(picked, start=1):
        src = find_first_with_any_ext(folder, "makeup_04")
        if src:
            out_name = f"{idx:04d}_makeup_04_{folder.name}{src.suffix}"
            safe_copy(src, makeup_dir / out_name, symlink=USE_SYMLINKS)

    print("Done.")
    print(f"Output written to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
