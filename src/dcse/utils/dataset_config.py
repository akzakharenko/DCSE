import yaml
from dataclasses import dataclass
from typing import List, Dict, Set, Any


@dataclass
class DatasetConfig:
    input_fasta: str
    root_dir: str

    valid_bases: Set[str]

    num_datasets: int
    uniform_percent: float

    min_pairs: int
    max_pairs: int

    param_space: Dict[str, List[int]]
    target_species: Set[str]


def _convert_types(cfg: Dict[str, Any]) -> Dict[str, Any]:

    cfg = dict(cfg)

    if "valid_bases" in cfg:
        cfg["valid_bases"] = set(cfg["valid_bases"])

    if "target_species" in cfg:
        cfg["target_species"] = set(cfg["target_species"])

    return cfg


def load_dataset_config(path: str) -> DatasetConfig:

    with open(path, "r") as f:
        raw_cfg = yaml.safe_load(f)

    cfg = _convert_types(raw_cfg)

    return DatasetConfig(**cfg)