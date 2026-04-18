"""
Rename GlaS sample files by removing duplicated extensions.
  .bmp.bmp -> .bmp
  .png.png -> .png

Usage:
    python scripts/rename_glas_files.py

Set DRY_RUN = True first to preview changes without touching any files.
Set DRY_RUN = False to apply the renames.
"""

import os
import re

# ── Config ────────────────────────────────────────────────────────────────────

GLAS_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "glas_sample")

DRY_RUN = False  # set to True to preview without changes

SUBFOLDERS = ["gt_masks", "images", "pred_masks"]

# ── Logic ─────────────────────────────────────────────────────────────────────

def fix_name(filename: str) -> str | None:
    """
    Return the corrected filename if it has a duplicated extension, else None.

    Examples
    --------
    'train_001.bmp.bmp' -> 'train_001.bmp'
    'train_001.png.png' -> 'train_001.png'
    'train_001.bmp'     -> None  (already correct)
    """
    # Match pattern: name + ext + same ext (e.g. .bmp.bmp or .png.png)
    match = re.fullmatch(r"(.+)(\.[a-zA-Z0-9]+)\2", filename)
    if match:
        return match.group(1) + match.group(2)
    return None


def process_folder(folder_path: str, dry_run: bool) -> int:
    """Scan folder, rename files with duplicated extensions. Returns rename count."""
    if not os.path.isdir(folder_path):
        print(f"  [SKIP] Folder not found: {folder_path}")
        return 0

    renamed = 0
    for filename in sorted(os.listdir(folder_path)):
        new_name = fix_name(filename)
        if new_name is None:
            continue  # already correctly named

        src = os.path.join(folder_path, filename)
        dst = os.path.join(folder_path, new_name)

        if os.path.exists(dst):
            print(f"  [SKIP] Target already exists: {new_name}")
            continue

        if dry_run:
            print(f"  [DRY]  {filename}  ->  {new_name}")
        else:
            os.rename(src, dst)
            print(f"  [OK]   {filename}  ->  {new_name}")
        renamed += 1

    return renamed


def main():
    mode = "DRY RUN (no files changed)" if DRY_RUN else "LIVE RUN (files will be renamed)"
    print(f"\n{'='*55}")
    print(f"  GlaS File Renamer  |  {mode}")
    print(f"  Root: {GLAS_ROOT}")
    print(f"{'='*55}\n")

    total = 0
    for subfolder in SUBFOLDERS:
        path = os.path.join(GLAS_ROOT, subfolder)
        print(f"[{subfolder}]")
        count = process_folder(path, DRY_RUN)
        if count == 0:
            print("  (nothing to rename)")
        total += count

    print(f"\n{'='*55}")
    label = "would be renamed" if DRY_RUN else "renamed"
    print(f"  Total files {label}: {total}")
    if DRY_RUN:
        print("  Set DRY_RUN = False to apply changes.")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
