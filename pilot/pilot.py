"""
Degradation-range pilot (no training).

Question: at what degradation does the clean-trained baseline (M0) actually lose
accuracy? The current worst condition (SNR 8, r 0.5) costs M0 only ~0.006 Dice,
too little for any model to show a robustness gain. This pilot runs the ALREADY
TRAINED checkpoints on the validation patients under a harsher, wider grid and
reports Dice per model and condition, so the new training range is chosen from
evidence before anything is retrained.

Grid (36 conditions):
  noise order  'image'  = current pipeline (truncate k-space, then add noise in
                          image space: noise refills the removed frequencies)
               'kspace' = physically correct order (noise only in the acquired
                          k-space band, rescaled so per-pixel sigma is unchanged)
  SNR          8, 5, 3, 2    (linear: sigma = mu_brain / SNR)    + noise-free
  r            1.0, 0.5      (in-plane k-space kept per axis)
  thick        1, 5 voxels   (through-plane partial volume: a 5 mm slice is the
                              average of 5 neighbouring 1 mm slices)

Validation patients only (the test set stays untouched). The same noise
realisation is used for every model at a given (patient, condition), so model
comparisons are paired. Dice only (HD95 is slow and not needed to pick a range).

Reuses the team code unchanged: python.data.preprocess.preprocess_patient,
python.models.factory.build_model, python.physics.degradation.generate_kspace_mask.
"""

import argparse
import hashlib
import json
import os
import struct
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from python.data.preprocess import preprocess_patient
from python.models.factory import build_model
from python.physics.degradation import (
    apply_degradation_single_slice,
    generate_kspace_mask,
)

MODELS = ["M0", "M1", "M4", "M5", "M6"]   # the five distinct models (C1: M2=M1, M3=M5, M7=M4)
SNRS = [8, 5, 3, 2]
RS = [1.0, 0.5]
THICKS = [1, 5]
NOISE_ORDERS = ["image", "kspace"]


def conditions():
    """All pilot conditions as dicts; 'id' is a stable string key."""
    out = [dict(id="clean", noise="none", snr=None, r=1.0, thick=1)]
    for r in RS:
        for t in THICKS:
            if r == 1.0 and t == 1:
                continue
            out.append(dict(id=f"nonoise_r{r}_t{t}", noise="none", snr=None, r=r, thick=t))
    for n in NOISE_ORDERS:
        for s in SNRS:
            for r in RS:
                for t in THICKS:
                    out.append(dict(id=f"{n}_snr{s}_r{r}_t{t}", noise=n, snr=s, r=r, thick=t))
    return out


def seed_for(*parts) -> int:
    """Deterministic 31-bit seed from strings (same scheme as the team's SHA256 seeding)."""
    h = hashlib.sha256("_".join(str(p) for p in parts).encode()).digest()[:4]
    return struct.unpack(">I", h)[0] % (2**31 - 1)


def degrade_volume(x, mu, cond, gen):
    """
    x    : (Z, C, H, W) raw intensities, float32, on device
    mu   : (C,) mean clean brain intensity per channel
    cond : condition dict
    gen  : torch.Generator on x.device
    Returns the degraded magnitude volume, same shape.
    """
    Z, C, H, W = x.shape
    # 1. Through-plane partial volume: moving average over `thick` slices along Z
    #    (replicate-padded so the volume keeps 155 slices on the 1 mm grid).
    if cond["thick"] > 1:
        k = cond["thick"]
        xp = x.permute(1, 2, 3, 0).reshape(1, C * H * W, Z)
        xp = torch.nn.functional.pad(xp, (k // 2, k - 1 - k // 2), mode="replicate")
        xp = torch.nn.functional.avg_pool1d(xp, kernel_size=k, stride=1)
        x = xp.reshape(C, H, W, Z).permute(3, 0, 1, 2).contiguous()

    # 2. In-plane resolution loss: keep the central r fraction of k-space
    #    (same mask as the training pipeline).
    mask = generate_kspace_mask((H, W), cond["r"], device=x.device)
    K = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1)) * mask
    img = torch.fft.ifft2(torch.fft.ifftshift(K, dim=(-2, -1)))   # complex

    if cond["noise"] == "none":
        return img.real

    sigma = (mu / cond["snr"]).view(1, C, 1, 1)
    nr = torch.randn(x.shape, generator=gen, device=x.device)
    ni = torch.randn(x.shape, generator=gen, device=x.device)

    if cond["noise"] == "image":
        # Current pipeline: noise added after truncation, at full bandwidth.
        return torch.sqrt((img.real + nr * sigma) ** 2 + (ni * sigma) ** 2)

    # 'kspace': complex noise restricted to the acquired band, as a real scanner
    # would record it. Band-limiting removes a fraction of the noise power, so
    # rescale by 1/sqrt(kept fraction) to keep the per-pixel sigma = mu / SNR
    # (the SNR label then means the same thing in both noise orders).
    frac = mask.mean()
    N = torch.fft.fftshift(torch.fft.fft2(torch.complex(nr, ni)), dim=(-2, -1)) * mask
    n_bl = torch.fft.ifft2(torch.fft.ifftshift(N, dim=(-2, -1))) / torch.sqrt(frac)
    return torch.abs(img + n_bl * sigma)


def normalise(xd, brain):
    """Per slice, per channel z-score inside the brain mask (python/evaluate.py)."""
    m = brain[:, None].expand_as(xd)                       # (Z, C, H, W) bool
    cnt = m.sum(dim=(-2, -1)).clamp(min=1)
    mean = (xd * m).sum(dim=(-2, -1)) / cnt
    var = ((xd - mean[..., None, None]) ** 2 * m).sum(dim=(-2, -1)) / (cnt - 1).clamp(min=1)
    std = var.sqrt()
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    has = (cnt > 1)[..., None, None]
    return torch.where(has, (xd - mean[..., None, None]) / std[..., None, None], xd)


def dice(pred, gt):
    """3D Dice; BraTS convention for empty truth (1 if both empty, else 0)."""
    p, g = pred.sum().item(), gt.sum().item()
    if g == 0:
        return 1.0 if p == 0 else 0.0
    return 2.0 * (pred & gt).sum().item() / (p + g)


def self_check(device):
    """
    Known-answer checks, run before the pilot:
    1. 'image' noise order reproduces the team's apply_degradation_single_slice
       exactly, given the same noise arrays.
    2. 'kspace' order keeps the per-pixel noise sigma at mu / SNR.
    3. thick = 1 and r = 1 without noise returns the input unchanged.
    """
    g = torch.Generator(device="cpu").manual_seed(0)
    x = torch.rand((3, 4, 64, 64), generator=g) + 0.1
    mu = x.mean(dim=(0, 2, 3))
    # 1. parity with the team implementation
    cond = dict(noise="image", snr=8, r=0.5, thick=1)
    gd = torch.Generator(device="cpu").manual_seed(1)
    ours = degrade_volume(x, mu, cond, gd)
    gd = torch.Generator(device="cpu").manual_seed(1)
    nr = torch.randn(x.shape, generator=gd)
    ni = torch.randn(x.shape, generator=gd)
    worst = 0.0
    for z in range(x.shape[0]):
        shared = torch.stack([nr[z], ni[z]], dim=1)          # (C, 2, H, W)
        ref = apply_degradation_single_slice(x[z], mu, 8.0, 0.5, shared_noise=shared)
        worst = max(worst, (ref - ours[z]).abs().max().item())
    assert worst < 1e-4, f"parity with team degradation failed: {worst}"
    # 2. kspace noise level on a flat zero image: magnitude of complex Gaussian
    #    noise with per-component sigma s has mean s*sqrt(pi/2) (Rayleigh).
    z0 = torch.zeros((8, 1, 128, 128))
    m1 = torch.ones(1)
    out = degrade_volume(z0, m1, dict(noise="kspace", snr=4, r=0.5, thick=1),
                         torch.Generator(device="cpu").manual_seed(2))
    s_hat = out.mean().item() / np.sqrt(np.pi / 2)
    assert abs(s_hat - 0.25) / 0.25 < 0.03, f"kspace sigma {s_hat} != 0.25"
    # 3. identity
    same = degrade_volume(x, mu, dict(noise="none", snr=None, r=1.0, thick=1), None)
    assert (same - x).abs().max().item() < 1e-5, "identity failed"
    print(f"SELF-CHECK PASSED (parity max diff {worst:.2e}, kspace sigma {s_hat:.4f} vs 0.25)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True, help="MICCAI_BraTS2020_TrainingData folder")
    ap.add_argument("--out_csv", required=True, help="results CSV (appended; resumable)")
    ap.add_argument("--splits", default="data/splits/split_v1.json")
    ap.add_argument("--max_patients", type=int, default=0, help="0 = all validation patients")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--models", default=",".join(MODELS), help="comma-separated model ids")
    ap.add_argument("--ckpt_root", default="results",
                    help="checkpoints at <root>/<model>/seed_0/best_model.pt (v1: results; v2: the train_v2 folder)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    self_check(device)

    val_ids = json.load(open(args.splits))["val"]
    if args.max_patients:
        val_ids = val_ids[: args.max_patients]
    mapping = pd.read_csv(Path(args.raw_dir) / "name_mapping.csv")
    grade = dict(zip(mapping["BraTS_2020_subject_ID"], mapping["Grade"]))

    models = {}
    for m in args.models.split(","):
        net = build_model(m, d0=0.20).to(device)
        # weights_only=False: the team's checkpoints store more than tensors, which
        # PyTorch >= 2.6 refuses to unpickle by default. They are our own files.
        ckpt = torch.load(f"{args.ckpt_root}/{m}/seed_0/best_model.pt", map_location=device, weights_only=False)
        net.load_state_dict(ckpt["model_state_dict"])
        models[m] = net.eval()

    conds = conditions()
    done = set()
    if os.path.exists(args.out_csv):
        prev = pd.read_csv(args.out_csv)
        done = set(zip(prev.patient_id, prev.condition_id))
        print(f"Resuming: {len(done)} (patient, condition) pairs already done")

    t0 = time.time()
    for i, pid in enumerate(val_ids):
        if all((pid, c["id"]) in done for c in conds):
            continue
        pdir = Path(args.raw_dir) / pid
        files = {p.name.split("_")[-1].split(".")[0]: str(p) for p in pdir.glob("*.nii*")}
        seg = files.get("seg") or next(str(p) for p in pdir.glob("*.nii*")
                                       if not any(k in p.name for k in ["_t1", "_t2", "_flair"]))
        row = pd.Series(dict(patient_id=pid, grade=grade.get(pid, "Unknown"),
                             t1_path=files["t1"], t1ce_path=files["t1ce"],
                             t2_path=files["t2"], flair_path=files["flair"], seg_path=seg))
        d = preprocess_patient(row)
        x = torch.from_numpy(d["modalities"].astype(np.float32)).permute(3, 0, 1, 2).to(device)
        brain = torch.from_numpy(d["brain_mask"]).permute(2, 0, 1).bool().to(device)
        mu = torch.from_numpy(d["mu_brain"]).to(device)
        seg = torch.from_numpy(d["seg"]).permute(2, 0, 1).to(device)       # (Z, H, W)
        gts = [seg > 0, (seg == 1) | (seg == 3), seg == 3]

        rows = []
        for c in conds:
            if (pid, c["id"]) in done:
                continue
            gen = torch.Generator(device=device).manual_seed(seed_for(pid, c["id"]))
            with torch.no_grad():
                xn = normalise(degrade_volume(x, mu, c, gen), brain)
                for m, net in models.items():
                    out = torch.cat([net(xn[s:s + args.batch]) for s in range(0, xn.shape[0], args.batch)])
                    pb = out > 0.5                                       # (Z, 3, H, W)
                    ds = [dice(pb[:, k], gts[k]) for k in range(3)]
                    rows.append(dict(model=m, patient_id=pid, grade=row.grade, condition_id=c["id"],
                                     noise=c["noise"], snr=c["snr"] or 0, r=c["r"], thick=c["thick"],
                                     dice_wt=ds[0], dice_tc=ds[1], dice_et=ds[2], dice_mean=float(np.mean(ds))))
        pd.DataFrame(rows).to_csv(args.out_csv, mode="a", header=not os.path.exists(args.out_csv), index=False)
        print(f"[{i + 1}/{len(val_ids)}] {pid}: {len(rows)} rows ({time.time() - t0:.0f} s elapsed)", flush=True)
    print("PILOT DONE")


if __name__ == "__main__":
    main()
