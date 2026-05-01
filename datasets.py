"""
Open-source 3D dataset downloader for training the OccupancyNetwork.

Supported datasets
──────────────────
modelnet40   Princeton ModelNet40 — 12,311 meshes across 40 categories (.off)
             Direct download, no registration needed.
             URL: https://modelnet.cs.princeton.edu/ModelNet40.zip  (~435 MB)

modelnet10   ModelNet10 — 4,899 meshes across 10 categories (subset of the above)
             Faster to download (~70 MB)

objaverse    Objaverse — 800K+ annotated 3D objects (.glb) from Hugging Face
             Requires: pip install objaverse
             Huge variety; can filter by category / keyword.

Usage
─────
  python -c "import datasets; datasets.download('modelnet40')"
  python -c "import datasets; datasets.list_categories('modelnet40')"

  # From CLI (via train.py):
  python train.py --dataset modelnet40 --categories chair,table,lamp --n 200
  python train.py --dataset objaverse  --categories chair --n 100
"""

import os
import sys
import glob
import json
import urllib.request
import zipfile
import shutil
from pathlib import Path
from typing import List, Optional, Dict

# Default cache directory
CACHE_DIR = os.path.join(os.path.dirname(__file__), ".dataset_cache")

# ─────────────────────────────────────────────────────────────────────────────
# Dataset registry
# ─────────────────────────────────────────────────────────────────────────────

_DATASETS: Dict[str, dict] = {
    "modelnet40": {
        "url":         "https://modelnet.cs.princeton.edu/ModelNet40.zip",
        "zip_name":    "ModelNet40.zip",
        "root_folder": "ModelNet40",
        "format":      ".off",
        "description": "40-class CAD mesh dataset from Princeton (~435 MB)",
        "categories": [
            "airplane","bathtub","bed","bench","bookshelf","bottle","bowl","car",
            "chair","cone","cup","curtain","desk","door","dresser","flower_pot",
            "glass_box","guitar","keyboard","lamp","laptop","mantel","monitor",
            "night_stand","person","piano","plant","radio","range_hood","sink",
            "sofa","stairs","stool","table","tent","toilet","tv_stand","vase",
            "wardrobe","xbox",
        ],
    },
    "modelnet10": {
        "url":         "https://modelnet.cs.princeton.edu/ModelNet10.zip",
        "zip_name":    "ModelNet10.zip",
        "root_folder": "ModelNet10",
        "format":      ".off",
        "description": "10-class subset of ModelNet (~70 MB, faster download)",
        "categories": [
            "bathtub","bed","chair","desk","dresser",
            "monitor","night_stand","sofa","table","toilet",
        ],
    },
    "objaverse": {
        "url":         None,  # handled via the objaverse Python package
        "description": "800K+ annotated 3D objects from Hugging Face (requires: pip install objaverse)",
        "format":      ".glb",
        "categories":  [],    # dynamic — see list_categories('objaverse')
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def list_datasets() -> List[str]:
    return list(_DATASETS.keys())


def describe(name: str) -> str:
    if name not in _DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list_datasets()}")
    d = _DATASETS[name]
    return f"{name}: {d['description']}"


def list_categories(name: str) -> List[str]:
    if name not in _DATASETS:
        raise ValueError(f"Unknown dataset '{name}'")
    if name == "objaverse":
        return _objaverse_categories()
    return _DATASETS[name]["categories"]


def download(name: str, cache_dir: str = CACHE_DIR) -> str:
    """
    Download and extract the dataset if not already cached.
    Returns the root directory of the extracted dataset.
    """
    if name not in _DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list_datasets()}")
    if name == "objaverse":
        raise ValueError("Objaverse is streamed, not bulk-downloaded. "
                         "Use get_files() directly.")
    d = _DATASETS[name]
    root = os.path.join(cache_dir, d["root_folder"])
    if os.path.isdir(root):
        print(f"  Already cached: {root}")
        return root

    os.makedirs(cache_dir, exist_ok=True)
    zip_path = os.path.join(cache_dir, d["zip_name"])

    if not os.path.exists(zip_path):
        print(f"  Downloading {name} from {d['url']}")
        print(f"  {d['description']}")
        _download_with_progress(d["url"], zip_path)

    print(f"  Extracting to {cache_dir}…")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(cache_dir)

    # Clean up zip to save space
    os.remove(zip_path)
    print(f"  Done → {root}")
    return root


def get_files(
    name:       str,
    categories: Optional[List[str]] = None,
    split:      str = "train",
    n:          Optional[int] = None,
    cache_dir:  str = CACHE_DIR,
    prompt:     str = "",
) -> List[str]:
    """
    Return a list of mesh file paths for the given dataset, categories, and split.

    name       : 'modelnet40', 'modelnet10', or 'objaverse'
    categories : list of category names, or None for all
    split      : 'train' | 'test' | 'all'
    n          : max number of files to return (None = all)
    cache_dir  : where datasets are cached
    prompt     : for objaverse, used as a search query
    """
    if name == "objaverse":
        return _objaverse_files(categories, n, cache_dir, prompt)

    root = download(name, cache_dir)
    avail = _DATASETS[name]["categories"]
    cats = _resolve_categories(categories, avail, name)
    fmt  = _DATASETS[name]["format"]

    files = []
    for cat in cats:
        if split in ("train", "all"):
            files += glob.glob(os.path.join(root, cat, "train", f"*{fmt}"))
        if split in ("test", "all"):
            files += glob.glob(os.path.join(root, cat, "test",  f"*{fmt}"))

    if not files:
        raise RuntimeError(
            f"No {fmt} files found for {name}/{categories} in {root}.\n"
            f"Try re-downloading: delete {cache_dir} and run again."
        )

    files = sorted(files)
    if n is not None:
        import random; random.seed(0)
        files = random.sample(files, min(n, len(files)))
    return files


# ─────────────────────────────────────────────────────────────────────────────
# Objaverse helpers
# ─────────────────────────────────────────────────────────────────────────────

def _objaverse_categories() -> List[str]:
    """Return top-level Objaverse tags (requires objaverse package)."""
    try:
        import objaverse
        anns = objaverse.load_annotations()
        # Collect all tags
        tags = {}
        for uid, ann in anns.items():
            for tag in ann.get("tags", []):
                tags[tag.get("name", "")] = tags.get(tag.get("name",""), 0) + 1
        return [t for t, _ in sorted(tags.items(), key=lambda x: -x[1])[:200]]
    except ImportError:
        print("objaverse not installed. Run: pip install objaverse")
        return []


def _objaverse_files(categories, n, cache_dir, prompt) -> List[str]:
    try:
        import objaverse
    except ImportError:
        raise ImportError(
            "objaverse package not installed.\n"
            "Run: pip install objaverse\n"
            "Then retry."
        )

    print("  Loading Objaverse annotations…")
    all_uids = objaverse.load_uids()
    anns     = objaverse.load_annotations(all_uids)

    # Filter by categories / prompt
    target_tags = set()
    if categories:
        target_tags.update(c.lower() for c in categories)
    if prompt:
        import re
        target_tags.update(re.findall(r"[a-z]+", prompt.lower()))

    if target_tags:
        selected = [
            uid for uid, ann in anns.items()
            if any(
                t.get("name", "").lower() in target_tags
                for t in ann.get("tags", [])
            )
        ]
    else:
        selected = list(anns.keys())

    if not selected:
        print(f"  Warning: no objects matched tags {target_tags}. Using random sample.")
        import random; random.seed(0)
        selected = random.sample(list(anns.keys()), min(n or 100, len(anns)))

    if n is not None:
        import random; random.seed(0)
        selected = random.sample(selected, min(n, len(selected)))

    print(f"  Downloading {len(selected)} Objaverse objects…")
    objects = objaverse.load_objects(
        uids=selected,
        download_processes=4,
    )
    return list(objects.values())  # local .glb file paths


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_categories(categories, available, name) -> List[str]:
    if categories is None or categories == ["all"]:
        return available
    invalid = [c for c in categories if c not in available]
    if invalid:
        print(f"  Warning: unknown categories for {name}: {invalid}")
        print(f"  Available: {available}")
    return [c for c in categories if c in available] or available


def _download_with_progress(url: str, dest: str):
    """Download a file with a simple progress bar."""
    def _hook(count, block_size, total_size):
        if total_size <= 0:
            return
        frac    = min(count * block_size / total_size, 1.0)
        filled  = int(40 * frac)
        bar     = "█" * filled + "░" * (40 - filled)
        mb_done = count * block_size / 1e6
        mb_tot  = total_size / 1e6
        print(f"\r  [{bar}] {mb_done:.1f}/{mb_tot:.1f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_hook)
    print()  # newline after progress bar


# ─────────────────────────────────────────────────────────────────────────────
# Stand-alone usage
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="3D dataset downloader")
    p.add_argument("action",   choices=["download","list","categories","info"])
    p.add_argument("--dataset",default="modelnet10")
    p.add_argument("--cache",  default=CACHE_DIR)
    args = p.parse_args()

    if args.action == "list":
        for ds in list_datasets():
            print(f"  {ds:<16} {_DATASETS[ds]['description']}")
    elif args.action == "info":
        print(describe(args.dataset))
    elif args.action == "categories":
        cats = list_categories(args.dataset)
        print(f"{args.dataset} categories ({len(cats)}):")
        for c in cats: print(f"  {c}")
    elif args.action == "download":
        download(args.dataset, args.cache)
