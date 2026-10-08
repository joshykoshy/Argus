"""
Why is the pilot's clean 3D Dice ~0.2 when training reported ~0.75?

Hypothesis: training validation scored only the 8 largest-tumor slices per
patient (python/train.py + dataset.py, max_slices_per_patient = 8 for val), and
the models were trained only on tumor slices (DEV-003), so on the ~100+
tumor-free slices of a full volume they predict tumor that is not there. Those
false positives would not exist in the training metric but dominate 3D Dice.

For each validation patient (clean scans, models M0 and M4), this scores the
SAME predictions three ways:
  train_metric   per-slice Dice averaged over the 8 largest-tumor slices
                 (exactly what the training logs report)
  dice_3d        whole-volume 3D Dice (what the pilot and the test evaluation use)
  dice_3d_tumor_slices  3D Dice using only slices that contain tumor
and counts predicted whole-tumor voxels on slices with NO tumor.
If train_metric ~0.75 and dice_3d_tumor_slices is high while dice_3d is low,
the pilot is correct and the models produce false positives on tumor-free
slices. Also saves one picture of such a false positive.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from pilot.pilot import degrade_volume, normalise, dice
from python.data.preprocess import preprocess_patient
from python.models.factory import build_model

ap = argparse.ArgumentParser()
ap.add_argument("--raw_dir", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--max_patients", type=int, default=10)
args = ap.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
val_ids = json.load(open("data/splits/split_v1.json"))["val"][: args.max_patients]
models = {}
for m in ["M0", "M4"]:
    net = build_model(m, d0=0.20).to(device)
    ck = torch.load(f"results/{m}/seed_0/best_model.pt", map_location=device, weights_only=False)
    net.load_state_dict(ck["model_state_dict"])
    models[m] = net.eval()

clean = dict(noise="none", snr=None, r=1.0, thick=1)
rows, shown = [], False
for pid in val_ids:
    pdir = Path(args.raw_dir) / pid
    f = {p.name.split("_")[-1].split(".")[0]: str(p) for p in pdir.glob("*.nii*")}
    d = preprocess_patient(pd.Series(dict(patient_id=pid, grade="", t1_path=f["t1"], t1ce_path=f["t1ce"],
                                          t2_path=f["t2"], flair_path=f["flair"], seg_path=f["seg"])))
    x = torch.from_numpy(d["modalities"].astype(np.float32)).permute(3, 0, 1, 2).to(device)
    brain = torch.from_numpy(d["brain_mask"]).permute(2, 0, 1).bool().to(device)
    seg = torch.from_numpy(d["seg"]).permute(2, 0, 1).to(device)
    gts = [seg > 0, (seg == 1) | (seg == 3), seg == 3]
    xn = normalise(degrade_volume(x, torch.from_numpy(d["mu_brain"]).to(device), clean, None), brain)

    area = gts[0].sum(dim=(1, 2))
    tumor_z = torch.nonzero(area > 0).flatten()
    top8 = torch.argsort(area, descending=True)[:8]
    free_z = torch.nonzero(area == 0).flatten()
    for m, net in models.items():
        with torch.no_grad():
            pb = torch.cat([net(xn[s:s + 32]) for s in range(0, 155, 32)]) > 0.5     # (Z, 3, H, W)
        d3 = np.mean([dice(pb[:, k], gts[k]) for k in range(3)])
        d3t = np.mean([dice(pb[tumor_z, k], gts[k][tumor_z]) for k in range(3)])
        tm = np.mean([[dice(pb[z, k], gts[k][z]) for k in range(3)] for z in top8.tolist()])
        fp_free = pb[free_z, 0].sum().item()
        fp_slices = int((pb[free_z, 0].sum(dim=(1, 2)) > 50).sum().item())
        rows.append(dict(patient_id=pid, model=m, train_metric=tm, dice_3d=d3, dice_3d_tumor_slices=d3t,
                         true_wt_voxels=int(gts[0].sum().item()), pred_wt_on_tumor_free_slices=fp_free,
                         tumor_free_slices=len(free_z), tumor_free_slices_with_false_tumor=fp_slices))
        if m == "M0" and not shown and fp_slices > 0:
            z = free_z[torch.argmax(pb[free_z, 0].sum(dim=(1, 2)))].item()
            zt = top8[0].item()
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            for a, zz, ttl in [(ax[0], zt, f"slice {zt}: largest tumor"), (ax[1], z, f"slice {z}: NO tumor in truth")]:
                a.imshow(xn[zz, 3].cpu().T, cmap="gray", vmin=-2, vmax=4)
                a.contour(gts[0][zz].cpu().float().T, [0.5], colors="cyan", linewidths=1.5)
                a.contour(pb[zz, 0].cpu().float().T, [0.5], colors="red", linewidths=1.2)
                a.set_title(ttl)
                a.axis("off")
            fig.suptitle(f"{pid}, M0, clean FLAIR: cyan = expert whole tumor, red = model prediction")
            fig.tight_layout()
            fig.savefig(out / "diagnose_false_positives.png", dpi=130)
            shown = True
    print(pid, "done", flush=True)

R = pd.DataFrame(rows)
R.to_csv(out / "diagnose.csv", index=False)
pd.set_option("display.width", 200)
print(R.groupby("model")[["train_metric", "dice_3d", "dice_3d_tumor_slices",
                          "pred_wt_on_tumor_free_slices", "true_wt_voxels",
                          "tumor_free_slices_with_false_tumor", "tumor_free_slices"]].mean().round(3).to_string())
