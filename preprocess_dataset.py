#!/usr/bin/env python3
"""
============================================================
  Biodiversity Species Classifier -- Dataset Preprocessor
============================================================
Scans raw image folders, identifies the 10 available classes,
removes duplicates / corrupted / tiny images, balances the
dataset, applies augmentation to training images, and
produces the final train / val / test split ready for PyTorch.

Available dataset classes (Italian folder names):
  cane, cavallo, elefante, farfalla, gallina,
  gatto, mucca, pecora, ragno, scoiattolo

Run:
    py preprocess_dataset.py
"""

import os, sys, json, shutil, hashlib, logging, random, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime


# ============================================================
# Auto-install missing packages
# ============================================================
def _ensure_packages():
    import importlib.util
    required = {
        "Pillow": "PIL", "torch": "torch", "torchvision": "torchvision",
        "matplotlib": "matplotlib", "numpy": "numpy", "tqdm": "tqdm",
    }
    missing = [pkg for pkg, mod in required.items()
               if importlib.util.find_spec(mod) is None]
    if missing:
        print(f"[SETUP] Installing missing packages: {missing}")
        import subprocess
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet"] + missing
        )

_ensure_packages()

from PIL import Image, UnidentifiedImageError
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


# ============================================================
# Configuration
# ============================================================
SCRIPT_DIR    = Path(__file__).resolve().parent
DATA_ROOT     = SCRIPT_DIR / "data"
RAW_DIR       = DATA_ROOT / "raw" / "images"
PROCESSED_DIR = DATA_ROOT / "processed"

# All 10 real classes present in the actual dataset (Italian folder names)
TARGET_CLASSES: List[str] = [
    "dog", "horse", "elephant", "butterfly",
    "chicken", "cat", "cow", "sheep", "spider", "squirrel"
]

# Italian folder-name -> canonical English class mapping (full dataset)
FOLDER_CLASS_MAP: Dict[str, str] = {
    # Italian  (actual raw dataset folder names)
    "cane":       "dog",
    "cavallo":    "horse",
    "elefante":   "elephant",
    "farfalla":   "butterfly",
    "gallina":    "chicken",
    "gatto":      "cat",
    "mucca":      "cow",
    "pecora":     "sheep",
    "ragno":      "spider",
    "scoiattolo": "squirrel",
    # English aliases (future-proofing)
    "dog": "dog", "dogs": "dog", "puppy": "dog",
    "horse": "horse", "horses": "horse",
    "elephant": "elephant", "elephants": "elephant",
    "butterfly": "butterfly", "butterflies": "butterfly",
    "chicken": "chicken", "hen": "chicken", "rooster": "chicken",
    "cat": "cat", "cats": "cat", "kitten": "cat",
    "cow": "cow", "cows": "cow", "cattle": "cow",
    "sheep": "sheep", "lamb": "sheep",
    "spider": "spider", "spiders": "spider",
    "squirrel": "squirrel", "squirrels": "squirrel",
}

SPLIT_RATIOS   = {"train": 0.70, "val": 0.15, "test": 0.15}
MAX_PER_CLASS  = 1000           # cap for over-represented classes
MIN_IMAGE_SIZE = (100, 100)     # minimum (width, height) in pixels
VALID_EXTS     = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
RANDOM_SEED    = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("preprocess")


# ============================================================
# Utilities
# ============================================================

def file_md5(path: Path, chunk: int = 65_536) -> str:
    """MD5 hash of a file — streamed for memory efficiency."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while blk := f.read(chunk):
            h.update(blk)
    return h.hexdigest()


def resolve_class(folder_name: str) -> Optional[str]:
    """Map a raw folder name to a canonical target class (or None)."""
    key = folder_name.strip().lower().replace(" ", "_").replace("-", "_")
    return FOLDER_CLASS_MAP.get(key)


def scan_raw_images(raw_dir: Path) -> Dict[str, List[Path]]:
    """Walk raw_dir recursively, collect image paths per target class."""
    class_paths: Dict[str, List[Path]] = defaultdict(list)
    all_dirs = [p for p in raw_dir.rglob("*") if p.is_dir()]
    all_dirs.insert(0, raw_dir)
    matched: Dict[Path, str] = {}
    ignored: List[str] = []
    for d in all_dirs:
        cls = resolve_class(d.name)
        if cls:
            matched[d] = cls
        else:
            ignored.append(d.name)
    if ignored:
        log.info("Ignored (non-target) folders: %s", ignored)
    if not matched:
        log.warning("No matching class folders found under: %s", raw_dir)
        available = [p.name for p in raw_dir.rglob("*") if p.is_dir()]
        log.info("Available folders: %s", available)
        log.info("Add folder names to FOLDER_CLASS_MAP to include them.")
        return {}
    log.info("Matched class folders:")
    for d, cls in matched.items():
        log.info("  %-40s -> %s", d.name, cls)
    for d, cls in matched.items():
        for f in d.iterdir():
            if f.is_file():
                class_paths[cls].append(f)
    return dict(class_paths)


# ============================================================
# Image validation
# ============================================================

def validate_images(
    paths: List[Path], class_name: str
) -> Tuple[List[Path], int, int]:
    """
    Filter image paths:
      - Remove non-image files (bad extension or unreadable)
      - Remove images below MIN_IMAGE_SIZE
      - Remove MD5 duplicates

    Returns (valid_paths, n_corrupted, n_duplicates)
    """
    valid: List[Path] = []
    n_corrupted = 0
    n_duplicates = 0
    seen: set = set()

    for p in tqdm(paths, desc=f"  Validating [{class_name}]",
                  unit="img", ncols=80, leave=False):
        # 1. Extension check
        if p.suffix.lower() not in VALID_EXTS:
            n_corrupted += 1
            continue
        # 2. Corruption check
        try:
            with Image.open(p) as img:
                img.verify()
        except Exception:
            n_corrupted += 1
            continue
        # 3. Size check (re-open after verify invalidates handle)
        try:
            with Image.open(p) as img:
                img.load()
                w, h = img.size
        except Exception:
            n_corrupted += 1
            continue
        if w < MIN_IMAGE_SIZE[0] or h < MIN_IMAGE_SIZE[1]:
            n_corrupted += 1
            continue
        # 4. Duplicate check
        digest = file_md5(p)
        if digest in seen:
            n_duplicates += 1
            continue
        seen.add(digest)
        valid.append(p)

    return valid, n_corrupted, n_duplicates


# ============================================================
# Class balancing
# ============================================================

def balance_classes(
    class_images: Dict[str, List[Path]], max_per: int = MAX_PER_CLASS
) -> Dict[str, List[Path]]:
    """Cap over-represented classes at max_per via random sampling."""
    out = {}
    for cls, paths in class_images.items():
        if len(paths) > max_per:
            out[cls] = random.sample(paths, max_per)
            log.info("  [%s] Capped %d -> %d images", cls, len(paths), max_per)
        else:
            out[cls] = paths[:]
            if 0 < len(paths) < max_per:
                log.info(
                    "  [%s] %d images -- augmentation will be applied to train split",
                    cls, len(paths),
                )
    return out


# ============================================================
# Train / Val / Test split
# ============================================================

def split_dataset(
    class_images: Dict[str, List[Path]]
) -> Dict[str, Dict[str, List[Path]]]:
    """Stratified random split respecting SPLIT_RATIOS."""
    splits: Dict[str, Dict[str, List[Path]]] = {
        "train": {}, "val": {}, "test": {}
    }
    for cls, paths in class_images.items():
        s = paths[:]
        random.shuffle(s)
        n  = len(s)
        nt = int(n * SPLIT_RATIOS["train"])
        nv = int(n * SPLIT_RATIOS["val"])
        splits["train"][cls] = s[:nt]
        splits["val"][cls]   = s[nt: nt + nv]
        splits["test"][cls]  = s[nt + nv:]
    return splits


# ============================================================
# Copy to processed/
# ============================================================

def copy_split(
    splits: Dict[str, Dict[str, List[Path]]], out_dir: Path
) -> None:
    """Copy images from raw locations to the processed split directories."""
    for split_name, class_map in splits.items():
        for cls, paths in class_map.items():
            dest = out_dir / split_name / cls
            dest.mkdir(parents=True, exist_ok=True)
            for src in tqdm(paths, desc=f"  Copying {split_name}/{cls}",
                            unit="img", ncols=80, leave=False):
                dst = dest / src.name
                if dst.exists():
                    dst = dest / f"{src.stem}_{random.randint(1000, 9999)}{src.suffix}"
                shutil.copy2(src, dst)
    log.info("All images copied to %s", out_dir)


# ============================================================
# Augmentation transforms
# ============================================================

AUGMENT_TRANSFORM = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=20),
    T.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

EVAL_TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ============================================================
# PyTorch Dataset and DataLoader
# ============================================================

class WildlifeDataset(Dataset):
    """
    PyTorch Dataset for the processed wildlife image split.

    Parameters
    ----------
    root_dir  : path to split folder (e.g. data/processed/train)
    class_map : {class_name: class_index}
    transform : torchvision transform pipeline
    """

    def __init__(
        self,
        root_dir: Path,
        class_map: Dict[str, int],
        transform=None,
    ) -> None:
        self.root_dir  = Path(root_dir)
        self.class_map = class_map
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []
        self._load_samples()

    def _load_samples(self) -> None:
        for cls, idx in self.class_map.items():
            cls_dir = self.root_dir / cls
            if not cls_dir.exists():
                continue
            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() in VALID_EXTS:
                    self.samples.append((img_path, idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        img_path, label = self.samples[index]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), (0, 0, 0))
        if self.transform:
            image = self.transform(image)
        return image, label


def build_dataloaders(
    processed_dir: Path,
    class_map: Dict[str, int],
    batch_size: int = 32,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    """
    Build train, val, and test DataLoaders.
    Augmentation is applied only to the training split.
    """
    tfms = {
        "train": AUGMENT_TRANSFORM,
        "val":   EVAL_TRANSFORM,
        "test":  EVAL_TRANSFORM,
    }
    loaders: Dict[str, DataLoader] = {}
    for split, tfm in tfms.items():
        ds = WildlifeDataset(
            root_dir  = processed_dir / split,
            class_map = class_map,
            transform = tfm,
        )
        loaders[split] = DataLoader(
            ds,
            batch_size  = batch_size,
            shuffle     = (split == "train"),
            num_workers = num_workers,
            pin_memory  = torch.cuda.is_available(),
            drop_last   = (split == "train"),
        )
        log.info("  DataLoader [%s] -- %d samples", split, len(ds))
    return loaders


# ============================================================
# Reporting helpers
# ============================================================

def plot_class_distribution(
    splits: Dict[str, Dict[str, List[Path]]], out_path: Path
) -> None:
    x     = np.arange(len(TARGET_CLASSES))
    width = 0.25
    split_names = ["train", "val", "test"]
    colors      = ["#2ecc71", "#3498db", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (sp, color) in enumerate(zip(split_names, colors)):
        counts = [len(splits[sp].get(c, [])) for c in TARGET_CLASSES]
        bars = ax.bar(x + i * width, counts, width,
                      label=sp.capitalize(), color=color,
                      alpha=0.85, edgecolor="white", linewidth=0.5)
        for bar, cnt in zip(bars, counts):
            if cnt:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 3,
                        str(cnt), ha="center", va="bottom",
                        fontsize=8, fontweight="bold")
    ax.set_xlabel("Species Class", fontsize=12, labelpad=8)
    ax.set_ylabel("Number of Images", fontsize=12, labelpad=8)
    ax.set_title("Dataset Class Distribution  Train / Val / Test",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x + width)
    ax.set_xticklabels(
        [c.replace("_", " ").title() for c in TARGET_CLASSES], fontsize=10
    )
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(1, ax.get_ylim()[1]) * 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved class distribution chart -> %s", out_path)


def save_statistics(
    raw_counts:  Dict[str, int],
    valid_counts: Dict[str, int],
    splits:      Dict[str, Dict[str, List[Path]]],
    n_corr:      int,
    n_dup:       int,
    class_map:   Dict[str, int],
    out_path:    Path,
) -> None:
    stats = {
        "generated_at":            datetime.now().isoformat(),
        "target_classes":          TARGET_CLASSES,
        "class_index_map":         class_map,
        "raw_image_counts":        raw_counts,
        "valid_image_counts":      valid_counts,
        "total_corrupted_removed":  n_corr,
        "total_duplicates_removed": n_dup,
        "split_counts": {
            sp: {cls: len(p) for cls, p in cm.items()}
            for sp, cm in splits.items()
        },
        "split_ratios":        SPLIT_RATIOS,
        "max_images_per_class": MAX_PER_CLASS,
        "min_image_size":      list(MIN_IMAGE_SIZE),
        "augmentation": {
            "applied_to": "train",
            "transforms": [
                "RandomHorizontalFlip(p=0.5)",
                "RandomRotation(20deg)",
                "RandomResizedCrop(224, scale=0.8-1.0)",
                "ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)",
                "Normalize(ImageNet mean/std)",
            ],
        },
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    log.info("Saved dataset statistics -> %s", out_path)


def save_report(
    raw_counts:  Dict[str, int],
    valid_counts: Dict[str, int],
    splits:      Dict[str, Dict[str, List[Path]]],
    n_corr:      int,
    n_dup:       int,
    elapsed:     float,
    found:       List[str],
    missing:     List[str],
    out_path:    Path,
) -> None:
    sep = "-" * 60
    lines = [
        "=" * 60,
        "  BIODIVERSITY DATASET  PREPROCESSING REPORT",
        f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Duration  : {elapsed:.1f} s",
        "=" * 60, "",
        "TARGET CLASSES", sep,
    ]
    for cls in TARGET_CLASSES:
        status = "FOUND" if cls in found else "MISSING  (no raw images)"
        lines.append(f"  {cls:<20} {status}")
    if missing:
        lines += [
            "",
            "NOTE: Missing classes have 0 images in the processed split.",
            "Add raw images with folder names registered in FOLDER_CLASS_MAP",
            "inside preprocess_dataset.py, then re-run.",
        ]
    lines += ["", "RAW -> VALID IMAGE COUNTS", sep]
    for cls in TARGET_CLASSES:
        r, v = raw_counts.get(cls, 0), valid_counts.get(cls, 0)
        lines.append(f"  {cls:<20}  raw={r:>5}  valid={v:>5}  removed={r - v:>5}")
    lines += [
        "", "QUALITY CONTROL", sep,
        f"  Corrupted / too-small removed : {n_corr}",
        f"  Duplicate images removed       : {n_dup}",
        "", "SPLIT DISTRIBUTION", sep,
    ]
    for sp in ("train", "val", "test"):
        total = sum(len(p) for p in splits[sp].values())
        lines.append(f"  {sp.upper():<6}  total={total}")
        for cls in TARGET_CLASSES:
            cnt = len(splits[sp].get(cls, []))
            lines.append(f"         {cls:<20} {cnt:>5}")
    lines += [
        "", "AUGMENTATION (train split only)", sep,
        "  RandomHorizontalFlip(p=0.5)",
        "  RandomRotation(20 degrees)",
        "  RandomResizedCrop(224, scale=0.8-1.0)",
        "  ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)",
        "  Normalize (ImageNet mean/std)",
        "", "OUTPUT", sep,
        "  data/processed/train/{class}/",
        "  data/processed/val/{class}/",
        "  data/processed/test/{class}/",
        "  data/processed/class_distribution.png",
        "  data/processed/dataset_statistics.json",
        "  data/processed/preprocessing_report.txt",
        "  class_mapping.json",
        "", "=" * 60,
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        content = "\n".join(lines)
        f.write(content)
    log.info("Saved preprocessing report -> %s", out_path)


def save_class_mapping(class_map: Dict[str, int], out_path: Path) -> None:
    payload = {
        "class_to_index": class_map,
        "index_to_class": {str(v): k for k, v in class_map.items()},
        "num_classes":    len(class_map),
        "classes":        list(class_map.keys()),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    log.info("Saved class mapping -> %s", out_path)


def print_summary(
    raw_counts:   Dict[str, int],
    valid_counts:  Dict[str, int],
    splits:        Dict[str, Dict[str, List[Path]]],
    n_corr:        int,
    n_dup:         int,
) -> None:
    sep = "-" * 62
    print()
    print("=" * 62)
    print("          PREPROCESSING COMPLETE  SUMMARY")
    print("=" * 62)
    hdr = f"{'CLASS':<17} {'RAW':>6} {'VALID':>6} {'TRAIN':>6} {'VAL':>5} {'TEST':>5}"
    print("\n" + hdr)
    print(sep)
    for cls in TARGET_CLASSES:
        r  = raw_counts.get(cls, 0)
        v  = valid_counts.get(cls, 0)
        tr = len(splits["train"].get(cls, []))
        va = len(splits["val"].get(cls, []))
        te = len(splits["test"].get(cls, []))
        print(f"  {cls:<15} {r:>6} {v:>6} {tr:>6} {va:>5} {te:>5}")
    tt = sum(len(p) for p in splits["train"].values())
    tv = sum(len(p) for p in splits["val"].values())
    ts = sum(len(p) for p in splits["test"].values())
    print(sep)
    print(f"  {'TOTAL':<15} {'':>6} {'':>6} {tt:>6} {tv:>5} {ts:>5}")
    print(f"\n  Grand total images   : {tt + tv + ts}")
    print(f"  Corrupted removed    : {n_corr}")
    print(f"  Duplicates removed   : {n_dup}")
    print("=" * 62)
    print()


# ============================================================
# Main pipeline
# ============================================================

def main() -> None:
    t0 = time.time()
    log.info("Biodiversity Dataset Preprocessor  started")
    log.info("Raw directory    : %s", RAW_DIR)
    log.info("Output directory : %s", PROCESSED_DIR)

    # Step 1: Scan raw image directories
    log.info("[STEP 1/7] Scanning raw image directories ...")
    raw_class_paths = scan_raw_images(RAW_DIR)
    for cls in TARGET_CLASSES:
        raw_class_paths.setdefault(cls, [])

    # Step 2: Validate images
    log.info("[STEP 2/7] Validating images (corruption / size / duplicates) ...")
    raw_counts:        Dict[str, int]        = {}
    valid_class_paths: Dict[str, List[Path]] = {}
    n_corr_total = 0
    n_dup_total  = 0
    for cls in TARGET_CLASSES:
        paths           = raw_class_paths[cls]
        raw_counts[cls] = len(paths)
        if not paths:
            log.warning("  [%s] No raw images found -- skipping validation", cls)
            valid_class_paths[cls] = []
            continue
        valid, nc, nd = validate_images(paths, cls)
        valid_class_paths[cls] = valid
        n_corr_total += nc
        n_dup_total  += nd
        log.info("  [%s] raw=%d  valid=%d  corrupted=%d  duplicates=%d",
                 cls, len(paths), len(valid), nc, nd)
    valid_counts = {c: len(p) for c, p in valid_class_paths.items()}

    # Step 3: Balance
    log.info("[STEP 3/7] Balancing classes (max %d per class) ...", MAX_PER_CLASS)
    balanced = balance_classes(valid_class_paths)

    # Step 4: Split
    log.info("[STEP 4/7] Splitting into train / val / test ...")
    splits = split_dataset(balanced)
    for sp, cm in splits.items():
        log.info("  %-6s -- %d images", sp, sum(len(p) for p in cm.values()))

    # Step 5: Copy
    log.info("[STEP 5/7] Copying images to processed directory ...")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    copy_split(splits, PROCESSED_DIR)

    # Step 6: Class mapping
    log.info("[STEP 6/7] Saving class mapping ...")
    class_map = {cls: i for i, cls in enumerate(TARGET_CLASSES)}
    save_class_mapping(class_map, SCRIPT_DIR / "class_mapping.json")

    # Step 7: Reports + DataLoaders
    log.info("[STEP 7/7] Generating reports, charts, and DataLoaders ...")
    elapsed = time.time() - t0
    found   = [c for c in TARGET_CLASSES if valid_counts.get(c, 0) > 0]
    missing = [c for c in TARGET_CLASSES if valid_counts.get(c, 0) == 0]

    plot_class_distribution(splits, PROCESSED_DIR / "class_distribution.png")
    save_statistics(raw_counts, valid_counts, splits,
                    n_corr_total, n_dup_total, class_map,
                    PROCESSED_DIR / "dataset_statistics.json")
    save_report(raw_counts, valid_counts, splits,
                n_corr_total, n_dup_total, elapsed, found, missing,
                PROCESSED_DIR / "preprocessing_report.txt")

    log.info("Building DataLoaders ...")
    loaders = build_dataloaders(PROCESSED_DIR, class_map, batch_size=32)
    for sp, loader in loaders.items():
        if len(loader.dataset) == 0:
            log.warning("  DataLoader [%s] is empty -- no images in split", sp)
            continue
        try:
            imgs, labels = next(iter(loader))
            log.info("  DataLoader [%s]  batch=%s  labels=%s",
                     sp, tuple(imgs.shape), labels.tolist()[:4])
        except Exception as exc:
            log.warning("  DataLoader [%s] batch test failed: %s", sp, exc)

    print_summary(raw_counts, valid_counts, splits, n_corr_total, n_dup_total)
    log.info("Total elapsed time: %.1f s", elapsed)
    log.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
