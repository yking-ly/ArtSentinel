import os
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# ✅ Path to original dataset
ROOT = Path(r"C:\ArtSentinel\data")

SRC_TRAIN = ROOT / "train"
SRC_TEST = ROOT / "test"

# ✅ Output binary dataset
OUT = ROOT / "binary"
BT = OUT / "train" / "Bot-Made"
BR = OUT / "train" / "Brush-Made"
BT_T = OUT / "test" / "Bot-Made"
BR_T = OUT / "test" / "Brush-Made"

for d in [BT, BR, BT_T, BR_T]:
    d.mkdir(parents=True, exist_ok=True)

def classify_folder(name: str) -> str:
    """
    AI folders start with 'AI_'
    Everything else = Real
    """
    if name.lower().startswith("ai_"):
        return "Bot-Made"
    return "Brush-Made"

def copy_split(split_src: Path, split_out_bt: Path, split_out_br: Path):
    for folder in split_src.iterdir():
        if not folder.is_dir():
            continue

        cls = classify_folder(folder.name)
        target = split_out_bt if cls == "Bot-Made" else split_out_br

        files = list(folder.glob("*"))
        print(f"Copying {folder.name} → {cls} ({len(files)} files)")

        def copy_file(src):
            try:
                shutil.copy2(src, target / src.name)
            except Exception:
                pass

        with ThreadPoolExecutor(max_workers=16) as ex:
            ex.map(copy_file, files)

print("\n=== TRAIN SET ===")
copy_split(SRC_TRAIN, BT, BR)

print("\n=== TEST SET ===")
copy_split(SRC_TEST, BT_T, BR_T)

print("\n✅ Done! Binary dataset created successfully at:", OUT)
