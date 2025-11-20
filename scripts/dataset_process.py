#!/usr/bin/env python3
"""
Compile all 6 makeup styles into folders
"""

from pathlib import Path
import shutil
import sys
from typing import List, Optional
from tqdm import tqdm

# ===== CORRECTED PATHS =====
PARENT_DIR = Path("/home/msai/c250116/AI6132-GenAI-For-Visual-Synthesis/data/FFHQ-Makeup")
OUTPUT_DIR = Path("/home/msai/c250116/AI6132-GenAI-For-Visual-Synthesis/data/processed")
SAMPLE_COUNT = 10000
REQUIRE_ALL_6 = True
USE_SYMLINKS = False
# ===========================

ALL_STYLES = ["bare", "makeup_01", "makeup_02", "makeup_03", "makeup_04", "makeup_05"]
GLOB_EXTS = ["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"]

def find_first_with_any_ext(folder: Path, stem: str) -> Optional[Path]:
    """Find file with any extension"""
    for ext in GLOB_EXTS:
        for candidate in [folder / f"{stem}.{ext}", folder / f"{stem}.{ext.upper()}"]:
            if candidate.exists():
                return candidate
    return None

def folder_has_all_styles(folder: Path) -> bool:
    """Check if folder contains all 6 styles"""
    for style in ALL_STYLES:
        if find_first_with_any_ext(folder, style) is None:
            return False
    return True

def list_candidate_folders(parent: Path) -> List[Path]:
    """List all valid folders"""
    if not parent.exists():
        print(f"? ERROR: Parent folder not found: {parent}")
        sys.exit(1)
    
    print(f"Scanning directory: {parent}")
    
    # List all subdirectories with numeric names
    subs = [p for p in parent.iterdir() if p.is_dir() and p.name.isdigit()]
    print(f"Found {len(subs)} numbered folders")
    
    if len(subs) == 0:
        print(f"\n? No numbered folders found!")
        print(f"\nDirectory contents:")
        for item in list(parent.iterdir())[:10]:
            print(f"  - {item.name} ({'dir' if item.is_dir() else 'file'})")
        sys.exit(1)
    
    subs.sort(key=lambda p: int(p.name))
    
    # Check for complete folders
    if REQUIRE_ALL_6:
        valid = [p for p in subs if folder_has_all_styles(p)]
        print(f"Folders with all 6 styles: {len(valid)}/{len(subs)}")
    else:
        valid = subs
    
    # Show sample
    if len(subs) > 0:
        sample = subs[0]
        print(f"\nSample folder ({sample.name}):")
        for item in sorted(sample.iterdir())[:10]:
            print(f"  - {item.name}")
    
    return valid

def safe_copy(src: Path, dst: Path, symlink: bool = False):
    """Copy or symlink file"""
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
    print("=" * 60)
    print("AI6132: Multi-Style Makeup Dataset Compilation")
    print("=" * 60)
    
    candidates = list_candidate_folders(PARENT_DIR)
    
    if len(candidates) < SAMPLE_COUNT:
        print(f"\n? Warning: Found {len(candidates)} valid folders, but need {SAMPLE_COUNT}")
        print(f"   Will process all {len(candidates)} available folders")
        picked = candidates
    else:
        picked = candidates[:SAMPLE_COUNT]
        print(f"\n? Selected {len(picked)} folders for processing")
    
    # Create output directories
    style_dirs = {}
    for style in ALL_STYLES:
        style_dir = OUTPUT_DIR / style
        style_dir.mkdir(parents=True, exist_ok=True)
        style_dirs[style] = style_dir
    
    # Copy all styles
    for style in ALL_STYLES:
        print(f"\nProcessing '{style}'...")
        for idx, folder in enumerate(tqdm(picked, desc=f"  {style}"), start=1):
            src = find_first_with_any_ext(folder, style)
            if src:
                out_name = f"{idx:04d}_{style}_{folder.name}{src.suffix}"
                safe_copy(src, style_dirs[style] / out_name, USE_SYMLINKS)
    
    print("\n" + "=" * 60)
    print("? Compilation Complete!")
    print("=" * 60)
    for style in ALL_STYLES:
        count = len(list(style_dirs[style].iterdir()))
        print(f"  {style:12s}: {count:5d} images")

if __name__ == "__main__":
    main()
