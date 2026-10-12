"""
Export Python reference outputs of pilot.pilot.degrade_volume for the MATLAB
parity test (matlab/reference/test_degrade_volume_parity.m calls this).

    python matlab/reference/export_degrade_parity.py <out_dir>

Writes raw little-endian float32 arrays in torch layout (Z, C, H, W), C order:
  x.f32, nr.f32, ni.f32   input volume and the exact N(0,1) noise the torch
                          generator produces for it (same seed, same draw order)
  mu.f32                  per-channel mean brain intensity
  <cond_id>.f32           degrade_volume output for each of the 13 DEC-005 conditions
  meta.txt                Z C H W and the condition ids
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pilot.pilot import degrade_volume            # noqa: E402
from training.train_v2 import DEC005              # noqa: E402

Z, C, H, W = 9, 2, 192, 192
SEED = 1234


def main(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    g = np.random.default_rng(0)
    # Smooth positive "brain" in a disc, zero background, like a skull-stripped scan
    yy, xx = np.mgrid[:H, :W]
    disc = ((yy - H / 2) ** 2 + (xx - W / 2) ** 2) < (0.4 * H) ** 2
    x = (300 + 200 * g.random((Z, C, H, W))) * disc
    x = torch.from_numpy(x.astype(np.float32))
    mu = torch.tensor([x[:, c][x[:, c] > 0].mean().item() for c in range(C)])

    gen = torch.Generator().manual_seed(SEED)     # replay the draws degrade_volume makes
    nr = torch.randn(x.shape, generator=gen)
    ni = torch.randn(x.shape, generator=gen)
    for name, a in [("x", x), ("nr", nr), ("ni", ni), ("mu", mu)]:
        a.numpy().astype("<f4").tofile(out / f"{name}.f32")
    for c in DEC005:
        y = degrade_volume(x, mu, c, torch.Generator().manual_seed(SEED))
        y.numpy().astype("<f4").tofile(out / f"{c['id']}.f32")
    (out / "meta.txt").write_text(f"{Z} {C} {H} {W}\n" + "\n".join(
        f"{c['id']} {c['snr'] or 0} {c['r']} {c['thick']}" for c in DEC005) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
