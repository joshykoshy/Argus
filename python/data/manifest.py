"""
Manifest builder for BraTS 2020 dataset.
Scans raw patient directories, resolves all 4 modalities and segmentation,
merges grade annotations, and verifies dataset completeness.
"""

import os
import glob
import logging
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_manifest(raw_dir: str, mapping_csv: Optional[str] = None, output_csv: Optional[str] = None) -> pd.DataFrame:
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset directory not found: {raw_dir}")

    # Load grade mapping if available
    grade_map = {}
    if mapping_csv and Path(mapping_csv).exists():
        mapping_df = pd.read_csv(mapping_csv)
        for _, row in mapping_df.iterrows():
            subj_id = str(row["BraTS_2020_subject_ID"]).strip()
            grade = str(row["Grade"]).strip()
            grade_map[subj_id] = grade

    # Find patient directories
    patient_dirs = [p for p in raw_path.iterdir() if p.is_dir() and p.name.startswith("BraTS20_Training_")]
    patient_dirs.sort(key=lambda p: p.name)

    logger.info(f"Found {len(patient_dirs)} patient directories in {raw_dir}")

    records = []
    for p_dir in patient_dirs:
        p_id = p_dir.name
        
        # Modalities search
        t1_files = [f for f in p_dir.glob("*t1.nii*") if not f.name.endswith("t1ce.nii") and not f.name.endswith("t1ce.nii.gz")]
        t1ce_files = list(p_dir.glob("*t1ce.nii*"))
        t2_files = list(p_dir.glob("*t2.nii*"))
        flair_files = list(p_dir.glob("*flair.nii*"))
        seg_files = list(p_dir.glob("*seg.nii*"))

        # Check for non-standard seg naming if empty
        if not seg_files:
            seg_candidates = [
                f for f in p_dir.glob("*.nii*")
                if ("seg" in f.name.lower() or "label" in f.name.lower())
                and not any(m in f.name.lower() for m in ["_t1.", "_t1ce.", "_t2.", "_flair."])
            ]
            if not seg_candidates:
                # Fallback: any remaining .nii file that is not t1, t1ce, t2, flair
                seg_candidates = [
                    f for f in p_dir.glob("*.nii*")
                    if not any(f.name.endswith(suffix) for suffix in ["_t1.nii", "_t1.nii.gz", "_t1ce.nii", "_t1ce.nii.gz", "_t2.nii", "_t2.nii.gz", "_flair.nii", "_flair.nii.gz"])
                ]
            if seg_candidates:
                seg_files = seg_candidates
                logger.warning(f"Patient {p_id} segmentation file has non-standard name: {seg_candidates[0].name}")

        missing = []
        if not t1_files: missing.append("t1")
        if not t1ce_files: missing.append("t1ce")
        if not t2_files: missing.append("t2")
        if not flair_files: missing.append("flair")
        if not seg_files: missing.append("seg")

        if missing:
            logger.error(f"Patient {p_id} is missing modalities: {missing}")
            raise RuntimeError(f"Integrity check failed: {p_id} missing {missing}")

        records.append({
            "patient_id": p_id,
            "grade": grade_map.get(p_id, "Unknown"),
            "t1_path": str(t1_files[0]),
            "t1ce_path": str(t1ce_files[0]),
            "t2_path": str(t2_files[0]),
            "flair_path": str(flair_files[0]),
            "seg_path": str(seg_files[0]),
        })

    manifest_df = pd.DataFrame(records)
    logger.info(f"Successfully built manifest with {len(manifest_df)} patients.")
    logger.info(f"Grade distribution: {manifest_df['grade'].value_counts().to_dict()}")

    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        manifest_df.to_csv(output_csv, index=False)
        logger.info(f"Saved manifest to {output_csv}")

    return manifest_df


if __name__ == "__main__":
    raw_dir = r"C:\Users\Mayan\.cache\kagglehub\datasets\awsaf49\brats20-dataset-training-validation\versions\1\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData"
    mapping_csv = os.path.join(raw_dir, "name_mapping.csv")
    out_csv = r"C:\Users\Mayan\.gemini\antigravity\scratch\ecte408\data\manifest.csv"
    build_manifest(raw_dir, mapping_csv, out_csv)
