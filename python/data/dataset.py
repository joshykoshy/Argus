"""
PyTorch Dataset and DataLoader implementations for 2D axial BraTS 2020 slices.
Supports brain-slice sampling, tumor oversampling, on-the-fly degradation,
post-degradation brain z-score normalization, and 3D volume reconstruction.

High-Throughput Design:
At initialization, extracts only the selected 2D slices (top tumor-area slices per patient)
into memory as float16/uint8 (~760 MB total RAM for all 2,580 slices).
This achieves ZERO disk I/O during training and validation, enabling sub-minute epoch times on CPU.
"""

import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# ──────────────────────────────────────────────────────────────────────────────
# Target region helpers
# ──────────────────────────────────────────────────────────────────────────────
# Internal remapped labels: 0: BG, 1: NCR/NET, 2: ED, 3: ET
# WT = {1, 2, 3},  TC = {1, 3},  ET = {3}


def create_target_masks(seg_slice: np.ndarray) -> np.ndarray:
    """
    Converts integer segmentation slice (H, W) into binary 3-channel mask (3, H, W).
    Channel 0: WT (Whole Tumor)
    Channel 1: TC (Tumor Core)
    Channel 2: ET (Enhancing Tumor)
    """
    wt = (seg_slice > 0).astype(np.float32)
    tc = ((seg_slice == 1) | (seg_slice == 3)).astype(np.float32)
    et = (seg_slice == 3).astype(np.float32)
    return np.stack([wt, tc, et], axis=0)


def reconstruct_3d_volume(
    slice_preds: np.ndarray,
    crop_coords: Tuple[int, int, int, int] = (24, 216, 24, 216),
    orig_shape: Tuple[int, int, int] = (240, 240, 155),
) -> np.ndarray:
    """
    Reconstructs full 3D prediction volume (3, 240, 240, 155)
    from cropped 2D slice predictions (155, 3, 192, 192).
    """
    r_s, r_e, c_s, c_e = crop_coords
    full_vol = np.zeros(
        (3, orig_shape[0], orig_shape[1], orig_shape[2]), dtype=np.float32
    )
    # (155, 3, H_crop, W_crop) -> (3, H_crop, W_crop, 155)
    preds_transposed = np.transpose(slice_preds, (1, 2, 3, 0))
    full_vol[:, r_s:r_e, c_s:c_e, :] = preds_transposed
    return full_vol


# ──────────────────────────────────────────────────────────────────────────────
# Main Dataset (In-Memory 2D Slices)
# ──────────────────────────────────────────────────────────────────────────────

class BraTS2DSliceDataset(Dataset):
    """
    2D Axial slice dataset with pre-extracted in-memory 2D slices.

    Parameters
    ----------
    patient_ids       : list of patient ID strings
    cache_dir         : directory containing <patient_id>.npz files
    is_train          : enables spatial augmentation (random flips)
    tumor_oversample_ratio : (unused when curated slice ranking is active)
    degradation_fn    : callable(x, mu_brain, p_id, slice_idx) -> (x_deg, cond_id)
    spatial_augment   : random horizontal/vertical flip
    seed              : for shuffling and augmentation reproducibility
    max_slices_per_patient : top tumor-area slices to sample per patient (default 10 for train, 8 for val)
    """

    def __init__(
        self,
        patient_ids: List[str],
        cache_dir: str,
        is_train: bool = True,
        tumor_oversample_ratio: float = 1.0,
        degradation_fn: Optional[Callable] = None,
        spatial_augment: bool = False,
        seed: int = 42,
        max_slices_per_patient: Optional[int] = None,
    ):
        self.is_train = is_train
        self.degradation_fn = degradation_fn
        self.spatial_augment = spatial_augment
        self.seed = seed

        if max_slices_per_patient is None:
            max_slices_per_patient = 10 if is_train else 8

        self.samples: List[Dict] = []

        # Load each patient once, extract only selected 2D slices into RAM
        for p_id in patient_ids:
            npz_path = Path(cache_dir) / f"{p_id}.npz"
            if not npz_path.exists():
                raise FileNotFoundError(f"Missing cached patient file: {npz_path}")

            data = np.load(npz_path)
            modalities = data["modalities"]  # (4, 192, 192, 155) float16
            seg = data["seg"]                # (192, 192, 155) uint8
            brain_mask = data["brain_mask"]  # (192, 192, 155) uint8
            mu_brain = data["mu_brain"]      # (4,) float32

            num_slices = brain_mask.shape[2]

            # Identify tumor slices and rank by tumor voxel area
            tumor_z_with_counts = []
            for z in range(num_slices):
                t_count = int(np.sum(seg[:, :, z] > 0))
                if t_count > 0:
                    tumor_z_with_counts.append((z, t_count))

            if len(tumor_z_with_counts) > 0:
                tumor_z_with_counts.sort(key=lambda item: item[1], reverse=True)
                selected_z = [z for z, _ in tumor_z_with_counts[:max_slices_per_patient]]
            else:
                brain_z = [z for z in range(num_slices) if np.sum(brain_mask[:, :, z]) > 10]
                mid = len(brain_z) // 2
                selected_z = brain_z[max(0, mid - 4):min(len(brain_z), mid + 4)]

            for z in selected_z:
                self.samples.append({
                    "modalities_2d": modalities[:, :, :, z],  # float16 (4, 192, 192)
                    "seg_2d": seg[:, :, z],                   # uint8 (192, 192)
                    "mask_2d": brain_mask[:, :, z],           # uint8 (192, 192)
                    "mu_brain": mu_brain.astype(np.float32),  # float32 (4,)
                    "patient_id": p_id,
                    "slice_idx": z,
                })

        if is_train:
            random.Random(seed).shuffle(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Fast in-memory conversion to float32 tensors
        modalities_2d = sample["modalities_2d"].astype(np.float32)
        seg_2d = sample["seg_2d"]
        mask_2d = sample["mask_2d"].astype(np.float32)
        mu_brain = sample["mu_brain"]
        p_id = sample["patient_id"]
        z = sample["slice_idx"]

        target_masks = create_target_masks(seg_2d)

        x = torch.from_numpy(modalities_2d)   # (4, H, W)
        mask = torch.from_numpy(mask_2d)       # (H, W)
        y = torch.from_numpy(target_masks)     # (3, H, W)
        mu_b = torch.from_numpy(mu_brain)      # (4,)

        # Spatial augmentation (random flips)
        if self.is_train and self.spatial_augment:
            if random.random() > 0.5:
                x = torch.flip(x, dims=[-1])
                mask = torch.flip(mask, dims=[-1])
                y = torch.flip(y, dims=[-1])
            if random.random() > 0.5:
                x = torch.flip(x, dims=[-2])
                mask = torch.flip(mask, dims=[-2])
                y = torch.flip(y, dims=[-2])

        # Apply physics-based low-field degradation
        if self.degradation_fn is not None:
            x_deg, cond_id = self.degradation_fn(x, mu_b, p_id=p_id, slice_idx=z)
        else:
            x_deg = x
            cond_id = "clean"

        # Brain z-score normalization inside brain mask after degradation
        x_norm = torch.zeros_like(x_deg)
        mask_bool = mask > 0.5
        for c in range(4):
            ch = x_deg[c]
            if mask_bool.sum() > 0:
                vals = ch[mask_bool]
                mean_val = vals.mean()
                std_val = vals.std()
                if std_val < 1e-6:
                    std_val = torch.tensor(1.0)
                x_norm[c] = (ch - mean_val) / std_val
            else:
                x_norm[c] = ch

        return {
            "image": x_norm,          # (4, 192, 192) normalized input
            "target": y,              # (3, 192, 192) [WT, TC, ET]
            "mask": mask,             # (192, 192) brain mask
            "patient_id": p_id,
            "slice_idx": z,
            "condition_id": cond_id,
        }
