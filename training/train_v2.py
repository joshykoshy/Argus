"""
Training v2: fixes the problems the pilot diagnostic found, and trains every
model of the study under one protocol (docs/DECISIONS.md DEC-005, DEC-007).

What changed from python/train.py, and why
  slices      ALL brain slices of every training patient, tumor-free ones
              included. The v1 models saw only the 10 largest-tumor slices and
              drew tumor on 100 % of tumor-free slices (finding C11).
  validation  full-volume 3D Dice on all validation patients (all 155 slices),
              under three conditions (clean, SNR 5, worst), plus false-tumor
              volume on tumor-free slices. v1 scored the 8 largest-tumor slices
              only (~0.75 reported while true 3D Dice was ~0.2).
  loss        BCE + soft Dice summed over the whole BATCH per channel (v1's
              per-slice Dice is unstable on slices with no tumor).
  length      20 epochs, AdamW 3e-4, 1-epoch warm-up + cosine, mixed precision,
              checkpoint every epoch so a Colab disconnect resumes.
  degradation M0 trains on clean scans; M1, M4, M5, M6 draw one of the 13
              DEC-005 conditions uniformly per slice: k-space noise inside the
              acquired band (fixes C4), SNR {8, 5, 3} x r {1.0, 0.5} x slice
              thickness {1, 5 mm}, plus clean. A 5 mm slice is the mean of 5
              neighbouring 1 mm slices, so the dataset returns a 5-slice stack.
Unchanged: architectures (python.models.factory.build_model), preprocessing and
crop (python.data.preprocess), per-slice brain z-score normalisation, flips.
The degradation itself is pilot.pilot.degrade_volume, whose 'image' order is
verified against the team's implementation (pilot self-check).
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from pilot.pilot import degrade_volume, dice, normalise, seed_for
from python.models.factory import build_model

CLEAN = dict(id="clean", noise="none", snr=None, r=1.0, thick=1)
DEC005 = [CLEAN] + [dict(id=f"kspace_snr{s}_r{r}_t{t}", noise="kspace", snr=s, r=r, thick=t)
                    for s in (8, 5, 3) for r in (1.0, 0.5) for t in (1, 5)]
VAL_CONDS = [CLEAN,
             dict(id="kspace_snr5_r1.0_t1", noise="kspace", snr=5, r=1.0, thick=1),
             dict(id="kspace_snr3_r0.5_t5", noise="kspace", snr=3, r=0.5, thick=5)]   # worst (H1)


class SliceSet(Dataset):
    """Every brain slice of the given patients, as a 5-slice stack centred on it."""

    def __init__(self, work, ids, flips):
        self.work, self.flips, self.cache, self.index = Path(work), flips, {}, []
        for pid in ids:
            m = np.load(self.work / f"{pid}_m.npy", mmap_mode="r")
            self.index += [(pid, z) for z in np.nonzero(m.reshape(m.shape[0], -1).sum(1) > 100)[0]]

    def _arr(self, pid, kind):
        key = (pid, kind)
        if key not in self.cache:                # opened lazily: one handle per worker
            self.cache[key] = np.load(self.work / f"{pid}_{kind}.npy", mmap_mode="r")
        return self.cache[key]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        pid, z = self.index[i]
        xs = self._arr(pid, "x")
        zz = np.clip(np.arange(z - 2, z + 3), 0, xs.shape[0] - 1)     # replicate at the ends
        x5 = torch.from_numpy(np.array(xs[zz], dtype=np.float32))      # (5, C, H, W)
        seg = torch.from_numpy(np.array(self._arr(pid, "y")[z]))
        m = torch.from_numpy(np.array(self._arr(pid, "m")[z]))
        mu = torch.from_numpy(np.array(self._arr(pid, "mu")))
        y = torch.stack([seg > 0, (seg == 1) | (seg == 3), seg == 3]).float()   # WT, TC, ET
        if self.flips:
            for dim in (-1, -2):
                if random.random() < 0.5:
                    x5, y, m = x5.flip(dim), y.flip(dim), m.flip(dim)
        return x5, y, m, mu


def degrade_batch(x5, mu, gen):
    """One random DEC-005 condition per slice. x5: (B, 5, C, H, W) -> (B, C, H, W)."""
    out = []
    for i in range(x5.shape[0]):
        c = random.choice(DEC005)
        x = x5[i].mean(0) if c["thick"] == 5 else x5[i, 2]
        if c["noise"] == "none":
            out.append(x)
        else:
            out.append(degrade_volume(x[None], mu[i], dict(c, thick=1), gen)[0])
    return torch.stack(out)


def loss_fn(p, t, eps=1.0):
    """BCE + soft Dice, Dice summed over the whole batch per channel."""
    bce = torch.nn.functional.binary_cross_entropy(p, t)
    inter = (p * t).sum(dim=(0, 2, 3))
    denom = p.sum(dim=(0, 2, 3)) + t.sum(dim=(0, 2, 3))
    return bce + (1 - (2 * inter + eps) / (denom + eps)).mean()


@torch.no_grad()
def validate(model, work, ids, device):
    """
    Full-volume 3D Dice (mean of WT, TC, ET) per validation condition, averaged
    over patients; noise seeded per (patient, condition) so every model and every
    epoch sees the same degraded volumes. Also false tumor (cm^3) on the
    tumor-free slices of the clean scans.
    """
    model.eval()
    scores = {c["id"]: [] for c in VAL_CONDS}
    fp = []
    for pid in ids:
        x = torch.from_numpy(np.load(Path(work) / f"{pid}_x.npy").astype(np.float32)).to(device)
        m = torch.from_numpy(np.load(Path(work) / f"{pid}_m.npy")).to(device)
        seg = torch.from_numpy(np.load(Path(work) / f"{pid}_y.npy")).to(device)
        mu = torch.from_numpy(np.load(Path(work) / f"{pid}_mu.npy")).to(device)
        gts = [seg > 0, (seg == 1) | (seg == 3), seg == 3]
        for c in VAL_CONDS:
            gen = torch.Generator(device=device).manual_seed(seed_for(pid, c["id"]))
            xn = normalise(degrade_volume(x, mu, c, gen), m)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                p = torch.cat([model(xn[s:s + 32]) for s in range(0, xn.shape[0], 32)]).float() > 0.5
            scores[c["id"]].append(np.mean([dice(p[:, k], gts[k]) for k in range(3)]))
            if c is CLEAN:
                fp.append(p[gts[0].sum(dim=(1, 2)) == 0, 0].sum().item() / 1000)   # 1 mm^3 voxels
    model.train()
    return {k: float(np.mean(v)) for k, v in scores.items()}, float(np.mean(fp))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="M0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--work", required=True, help="folder written by training.prepare")
    ap.add_argument("--out", required=True, help="checkpoints + log (put it on Drive)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--slices_per_epoch", type=int, default=12000, help="random brain slices per epoch (0 = all)")
    ap.add_argument("--val_every", type=int, default=4)
    ap.add_argument("--val_patients", type=int, default=0, help="0 = all validation patients")
    ap.add_argument("--augment", choices=["auto", "yes", "no"], default="auto",
                    help="degradation augmentation; auto = every model except M0")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    augment = args.augment == "yes" or (args.augment == "auto" and args.model != "M0")

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out) / args.model / f"seed_{args.seed}"
    out.mkdir(parents=True, exist_ok=True)

    split = json.load(open("data/splits/split_v1.json"))
    val_ids = split["val"][: args.val_patients] if args.val_patients else split["val"]
    data = SliceSet(args.work, split["train"], flips=True)
    print(f"{args.model} seed {args.seed}: {len(data)} brain slices from {len(split['train'])} patients, "
          f"degradation augmentation {'ON (13 DEC-005 conditions)' if augment else 'OFF (clean)'}, device {device}",
          flush=True)

    model = build_model(args.model, d0=0.20).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.2, total_iters=1)
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs - 1, 1), eta_min=1e-6)
    sched = torch.optim.lr_scheduler.SequentialLR(opt, [warm, cos], milestones=[1])
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    start, best = 1, -1.0

    last = out / "last.pt"
    if last.exists():                           # resume after a disconnect (or skip if finished)
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start, best = ck["epoch"] + 1, ck["best"]
        print(f"Resuming from epoch {start} (best so far {best:.4f})", flush=True)

    g = torch.Generator().manual_seed(args.seed)
    noise_gen = torch.Generator(device=device).manual_seed(10_000 + args.seed)
    log = out / "log.csv"
    for epoch in range(start, args.epochs + 1):
        t0 = time.time()
        n = args.slices_per_epoch or len(data)
        sampler = torch.utils.data.RandomSampler(data, num_samples=min(n, len(data)), generator=g)
        loader = DataLoader(data, batch_size=args.batch, sampler=sampler, num_workers=args.workers,
                            pin_memory=True, drop_last=True)
        model.train()
        tot, nb = 0.0, 0
        for x5, y, m, mu in loader:
            x5, y, m, mu = (t.to(device, non_blocking=True) for t in (x5, y, m, mu))
            with torch.no_grad():
                x = degrade_batch(x5, mu, noise_gen) if augment else x5[:, 2]
                xn = normalise(x, m)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                p = model(xn)
            loss = loss_fn(p.float().clamp(1e-6, 1 - 1e-6), y)   # loss in fp32 (BCE is not autocast-safe)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            tot, nb = tot + loss.item(), nb + 1
            if nb % 100 == 0:
                print(f"  epoch {epoch} batch {nb}/{len(loader)} loss {tot / nb:.4f} "
                      f"({nb / (time.time() - t0):.1f} batch/s)", flush=True)
        sched.step()

        row = dict(epoch=epoch, train_loss=round(tot / max(nb, 1), 5), lr=opt.param_groups[0]["lr"],
                   epoch_s=round(time.time() - t0), **{f"val_{c['id']}": "" for c in VAL_CONDS},
                   val_mean="", val_false_tumor_cm3="")
        if epoch % args.val_every == 0 or epoch == args.epochs:
            s, fp = validate(model, args.work, val_ids, device)
            mean = float(np.mean(list(s.values())))
            row.update({f"val_{k}": round(v, 4) for k, v in s.items()},
                       val_mean=round(mean, 4), val_false_tumor_cm3=round(fp, 1))
            if mean > best:
                best = mean
                torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val": s},
                           out / "best_model.pt")
        new = not log.exists()
        with open(log, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(), "best": best}, last)
        print(f"epoch {epoch}: loss {row['train_loss']}, val mean 3D Dice {row['val_mean']} "
              f"(clean / SNR5 / worst: {row['val_clean']} / {row['val_kspace_snr5_r1.0_t1']} / "
              f"{row['val_kspace_snr3_r0.5_t5']}), false tumor {row['val_false_tumor_cm3']} cm3, "
              f"{row['epoch_s']} s", flush=True)
    print(f"TRAINING DONE {args.model} seed {args.seed}. Best mean 3D Dice {best:.4f} -> {out / 'best_model.pt'}")


if __name__ == "__main__":
    main()
