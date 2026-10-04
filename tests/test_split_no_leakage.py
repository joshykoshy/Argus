"""
Automated unit tests for dataset splitting and zero patient leakage.
"""

import json
from pathlib import Path
import pytest
import pandas as pd
from python.data.manifest import build_manifest
from python.data.splits import create_stratified_split, load_splits


def test_manifest_completeness():
    manifest_path = Path("C:/Users/Mayan/.gemini/antigravity/scratch/ecte408/data/manifest.csv")
    if not manifest_path.exists():
        raw_dir = r"C:\Users\Mayan\.cache\kagglehub\datasets\awsaf49\brats20-dataset-training-validation\versions\1\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData"
        mapping_csv = f"{raw_dir}/name_mapping.csv"
        df = build_manifest(raw_dir, mapping_csv, str(manifest_path))
    else:
        df = pd.read_csv(manifest_path)

    assert len(df) == 369, f"Expected 369 patients, got {len(df)}"
    assert df["grade"].isin(["HGG", "LGG"]).all(), "All patients must have HGG or LGG grade"


def test_split_no_leakage():
    split_path = Path("C:/Users/Mayan/.gemini/antigravity/scratch/ecte408/data/splits/split_v1.json")
    if not split_path.exists():
        manifest_path = "C:/Users/Mayan/.gemini/antigravity/scratch/ecte408/data/manifest.csv"
        df = pd.read_csv(manifest_path)
        splits = create_stratified_split(df, output_json=str(split_path))
    else:
        splits = load_splits(str(split_path))

    train_p = set(splits["train"])
    val_p = set(splits["val"])
    test_p = set(splits["test"])

    # Strict disjointness
    assert len(train_p.intersection(val_p)) == 0, "Patient overlap between Train and Val!"
    assert len(train_p.intersection(test_p)) == 0, "Patient overlap between Train and Test!"
    assert len(val_p.intersection(test_p)) == 0, "Patient overlap between Val and Test!"

    total_patients = len(train_p) + len(val_p) + len(test_p)
    assert total_patients == 369, f"Expected 369 total unique patients, got {total_patients}"

    # Verify split counts approximately 70/10/20
    assert 255 <= len(train_p) <= 260, f"Train count {len(train_p)} out of expected ~258"
    assert 35 <= len(val_p) <= 40, f"Val count {len(val_p)} out of expected ~37"
    assert 70 <= len(test_p) <= 76, f"Test count {len(test_p)} out of expected ~74"
