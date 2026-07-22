"""Load the existing PyG caches for DrugOOD IC50."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch_geometric.data import InMemoryDataset


def discover_data_root() -> Path:
    baselines_root = Path(__file__).resolve().parents[2]
    configured = os.environ.get("DRUGOOD_DATA_ROOT")
    candidates = (
        *((Path(configured).expanduser(),) if configured else ()),
        baselines_root.parent / "Graph-OOD-Lab" / "data" / "DrugOOD",
        Path("/workspace/Graph-OOD-Lab/data/DrugOOD"),
        Path("/home/jylim/Graph-OOD-Lab/data/DrugOOD"),
        Path("/home/jylim/project/Graph-OOD-Lab/data/DrugOOD"),
        Path("/home/irteam/Graph-OOD-Lab/data/DrugOOD"),
        Path("/home/irteam/project/Graph-OOD-Lab/data/DrugOOD"),
    )
    return next((path for path in candidates if path.is_dir()), candidates[0])


class CachedDrugOOD(InMemoryDataset):
    def __init__(self, path: Path):
        super().__init__(root=None)
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing DrugOOD split: {path}\n"
                "Pass --data-root explicitly or generate the matching PyG cache."
            )
        self.data, self.slices = torch.load(path, map_location="cpu", weights_only=False)


def dataset_stem(subset: str, domain: str) -> str:
    return f"drugood_lbap_{subset}_ic50_{domain}"


def load_splits(data_root: Path, subset: str, domain: str):
    stem = dataset_stem(subset, domain)
    splits = {
        split: CachedDrugOOD(data_root / f"{stem}_{split}.pt")
        for split in ("train", "ood_val", "ood_test")
    }
    for split in ("iid_val", "iid_test"):
        path = data_root / f"{stem}_{split}.pt"
        if path.is_file():
            splits[split] = CachedDrugOOD(path)
    return stem, splits
