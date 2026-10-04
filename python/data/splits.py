"""
Patient-level stratified dataset splitting for BraTS 2020.
Guarantees zero leakage across train, val, and test splits.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def create_stratified_split(
    manifest_df: pd.DataFrame,
    ratios: List[float] = [0.70, 0.10, 0.20],
    seed: int = 42,
    output_json: str = "data/splits/split_v1.json"
) -> Dict[str, List[str]]:
    assert len(ratios) == 3 and abs(sum(ratios) - 1.0) < 1e-4, "Ratios must sum to 1.0"
    train_ratio, val_ratio, test_ratio = ratios

    # First split: train+val vs test (80% vs 20%)
    test_frac = test_ratio
    train_val_df, test_df = train_test_split(
        manifest_df,
        test_size=test_frac,
        stratify=manifest_df["grade"],
        random_state=seed,
        shuffle=True
    )

    # Second split: train vs val (70/80 = 87.5% train, 10/80 = 12.5% val)
    val_rel_frac = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_rel_frac,
        stratify=train_val_df["grade"],
        random_state=seed,
        shuffle=True
    )

    splits = {
        "train": sorted(train_df["patient_id"].tolist()),
        "val": sorted(val_df["patient_id"].tolist()),
        "test": sorted(test_df["patient_id"].tolist()),
        "metadata": {
            "seed": seed,
            "ratios": ratios,
            "total_patients": len(manifest_df),
            "train_count": len(train_df),
            "val_count": len(val_df),
            "test_count": len(test_df),
            "train_grades": train_df["grade"].value_counts().to_dict(),
            "val_grades": val_df["grade"].value_counts().to_dict(),
            "test_grades": test_df["grade"].value_counts().to_dict(),
        }
    }

    # Verify zero leakage
    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])

    assert len(train_set & val_set) == 0, "Leakage between train and val!"
    assert len(train_set & test_set) == 0, "Leakage between train and test!"
    assert len(val_set & test_set) == 0, "Leakage between val and test!"
    assert len(train_set | val_set | test_set) == len(manifest_df), "Missing patients in splits!"

    logger.info(f"Split created: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")
    logger.info(f"Train grades: {splits['metadata']['train_grades']}")
    logger.info(f"Val grades: {splits['metadata']['val_grades']}")
    logger.info(f"Test grades: {splits['metadata']['test_grades']}")

    out_p = Path(output_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(splits, f, indent=2)

    logger.info(f"Saved immutable splits to {output_json}")
    return splits


def load_splits(split_json_path: str) -> Dict[str, List[str]]:
    with open(split_json_path, "r") as f:
        return json.load(f)
