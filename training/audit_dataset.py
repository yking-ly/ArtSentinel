# audit_dataset.py
import os
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import imagehash

# ---- CONFIG ----
# Point this to your current dataset root.
# For the new binary set, use: C:\ArtSentinel\data\binary
# For the original 30-class set, use: C:\ArtSentinel\data\train (or the parent with train/test)
ROOT = Path(r"C:\ArtSentinel\data\binary")

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
PHASH_SAMPLE_CAP = 5000  # limit for near-duplicate scanning per split to keep it fast
MIN_SIDE = 224           # treat images smaller than this as "tiny"

def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXT

def list_class_dirs(split_dir: Path):
    """Return class folders under a split directory (e.g., .../train)."""
    return [d for d in split_dir.iterdir() if d.is_dir()]

def audit_dir(dir_path: Path):
    """
    Audit a directory that directly contains class folders.
    Works for:
      - ...\data\binary\train (Bot-Made, Brush-Made)
      - ...\data\train (30 class folders)
    """
    class_dirs = list_class_dirs(dir_path)
    if not class_dirs:
        print(f"[WARN] No class folders found under {dir_path}")
        return {}, {"bad_open": [], "non_rgb": [], "tiny": []}, {}

    summary = {}
    issues = {"bad_open": [], "non_rgb": [], "tiny": []}
    dup_map = {}

    # Collect candidate images for near-duplicate phash scan (sample across classes)
    phash_candidates = []

    for cdir in class_dirs:
        imgs = [p for p in cdir.rglob("*") if p.is_file() and is_image(p)]
        counts = {"total": len(imgs), "rgb": 0, "cmyk": 0, "gray": 0, "other": 0, "tiny": 0}
        for p in imgs:
            try:
                im = Image.open(p)
                mode = im.mode
                if mode == "RGB":
                    counts["rgb"] += 1
                elif mode == "CMYK":
                    counts["cmyk"] += 1
                    issues["non_rgb"].append(str(p))
                elif mode in ("L", "LA"):
                    counts["gray"] += 1
                    issues["non_rgb"].append(str(p))
                else:
                    counts["other"] += 1
                    issues["non_rgb"].append(str(p))
                if min(im.size) < MIN_SIDE:
                    counts["tiny"] += 1
                    issues["tiny"].append(str(p))
            except UnidentifiedImageError:
                issues["bad_open"].append(str(p))
            except Exception:
                # Skip other IO errors but continue
                pass

        summary[cdir.name] = counts

        # add a subset for phash scan
        phash_candidates.extend(imgs[: max(1, PHASH_SAMPLE_CAP // max(1, len(class_dirs)))])

    # Near-duplicate (perceptual hash) scan over sampled files
    hmap = {}
    for p in phash_candidates:
        try:
            im = Image.open(p).convert("RGB")
            h = str(imagehash.phash(im))
            hmap.setdefault(h, []).append(str(p))
        except Exception:
            continue
    for h, lst in hmap.items():
        if len(lst) > 1:
            dup_map[h] = lst

    return summary, issues, dup_map

def has_train_test(root: Path) -> bool:
    return (root / "train").exists() and (root / "test").exists()

def pretty_print(split_name: str, summary: dict, issues: dict, dup_map: dict):
    print(f"\n== {split_name.upper()} ==")
    # Align keys for stable display
    for cls in sorted(summary.keys()):
        c = summary[cls]
        print(
            f"{cls:14s} total={c['total']:5d} rgb={c['rgb']:5d} "
            f"cmyk={c['cmyk']:4d} gray={c['gray']:4d} other={c['other']:4d} tiny={c['tiny']:4d}"
        )
    print(
        f"Issues: bad_open={len(issues['bad_open'])}, "
        f"non_rgb={len(issues['non_rgb'])}, tiny={len(issues['tiny'])}"
    )
    if dup_map:
        dcount = sum(len(v) for v in dup_map.values())
        print(f"Near-duplicates: {len(dup_map)} groups, {dcount} files")
    print("Tip: convert CMYK/gray → RGB and drop tiny/bad files.")

def main():
    root = ROOT
    if has_train_test(root):
        # Audit both splits (e.g., data/binary/{train,test} OR data/{train,test})
        for split in ("train", "test"):
            split_dir = root / split
            if split_dir.exists():
                s, issues, dup = audit_dir(split_dir)
                pretty_print(split, s, issues, dup)
            else:
                print(f"[WARN] Missing split folder: {split_dir}")
    else:
        # Audit a single-level directory that directly contains classes
        s, issues, dup = audit_dir(root)
        pretty_print(root.name, s, issues, dup)

if __name__ == "__main__":
    main()
