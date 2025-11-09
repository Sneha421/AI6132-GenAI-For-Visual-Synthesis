"""
Compile all FFHQ-Makeup categories (bare, makeup_01~05)
into separate subfolders with progress visualization.
"""

from pathlib import Path
import shutil
import sys
from tqdm import tqdm

# -------------------- CONFIG --------------------
PARENT_DIR = Path(r"FFHQ-Makeup")
OUTPUT_DIR = Path("./compiled_output_all")
SAMPLE_COUNT = None
REQUIRE_ALL_6 = True
USE_SYMLINKS = False
# ------------------------------------------------

ALL_KEYS = ["bare", "makeup_01", "makeup_02", "makeup_03", "makeup_04", "makeup_05"]
GLOB_EXTS = ["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"]


def find_first_with_any_ext(folder: Path, stem: str) -> Path | None:
    for ext in GLOB_EXTS:
        for candidate in [folder / f"{stem}.{ext}", folder / f"{stem}.{ext.upper()}"]:
            if candidate.exists():
                return candidate
    matches = list(folder.glob(f"{stem}.*"))
    return matches[0] if matches else None


def folder_has_required(folder: Path, require_all_6: bool = False) -> bool:
    if require_all_6:
        return all(find_first_with_any_ext(folder, k) is not None for k in ALL_KEYS)
    return find_first_with_any_ext(folder, "bare") is not None


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
    total = len(candidates)
    print(f"Found {total} valid subject folders.")

    if SAMPLE_COUNT is None or SAMPLE_COUNT <= 0 or SAMPLE_COUNT > total:
        picked = candidates
        print(f"Using ALL {total} subjects.")
    else:
        picked = candidates[:SAMPLE_COUNT]
        print(f"Using first {SAMPLE_COUNT} subjects.")

    for key in ALL_KEYS:
        (OUTPUT_DIR / key).mkdir(parents=True, exist_ok=True)

    for key in ALL_KEYS:
        print(f"\nProcessing '{key}' images...")
        out_dir = OUTPUT_DIR / key

        for idx, folder in enumerate(tqdm(picked, desc=f"{key:10}", ncols=90)):
            src = find_first_with_any_ext(folder, key)
            if src:
                out_name = f"{idx+1:05d}_{key}_{folder.name}{src.suffix}"
                safe_copy(src, out_dir / out_name, symlink=USE_SYMLINKS)

    print("\n✅ All done.")
    print(f"📁 Output directory: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
