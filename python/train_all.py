"""
Master Multi-Model Sequential Training Runner for ECTE408 Study.
Trains all capacity-matched models (M0 through M7) on Seed 0 (and seeds 1, 2 for primary models)
with deterministic resumability and automated progress tracking.
"""

import sys
import time
from pathlib import Path
import pandas as pd

from python.train import train_model


def run_all_training(
    models=None,
    seeds=None,
    epochs=5,
    batch_size=32,
    lr=3e-4,
    d0=0.20,
    cache_dir="data/cache",
    splits_file="data/splits/split_v1.json",
    results_dir="results",
):
    if models is None:
        # All 8 capacity-matched models
        models = ["M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7"]
    if seeds is None:
        seeds = [0]

    print("=" * 60, flush=True)
    print("STARTING MASTER ECTE408 SEQUENTIAL TRAINING SUITE", flush=True)
    print(f"Models: {models}", flush=True)
    print(f"Seeds:  {seeds}", flush=True)
    print(f"Epochs per model: {epochs}", flush=True)
    print(f"Optimal D0: {d0}", flush=True)
    print("=" * 60, flush=True)

    summary_records = []
    suite_start = time.time()

    for m_id in models:
        for seed in seeds:
            t0 = time.time()
            print(f"\n>>> Starting {m_id} (Seed {seed}) at {time.strftime('%H:%M:%S')}...", flush=True)
            res = train_model(
                model_id=m_id,
                seed=seed,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                d0=d0,
                cache_dir=cache_dir,
                splits_file=splits_file,
                results_dir=results_dir,
            )
            elapsed = time.time() - t0
            print(f">>> Finished {m_id} (Seed {seed}) in {elapsed/60:.1f} minutes | Status: {res.get('status')} | Best Score: {res.get('best_val_score', 0.0):.4f}", flush=True)

            summary_records.append({
                "model": m_id,
                "seed": seed,
                "status": res.get("status"),
                "best_val_score": res.get("best_val_score", 0.0),
                "runtime_minutes": round(elapsed / 60.0, 1),
                "checkpoint": res.get("best_ckpt"),
            })

    total_time = time.time() - suite_start
    print("\n" + "=" * 60, flush=True)
    print(f"ALL TRAINING COMPLETED IN {total_time/60:.1f} MINUTES ({total_time/3600:.2f} HOURS)", flush=True)
    print("=" * 60, flush=True)

    summary_df = pd.DataFrame(summary_records)
    summary_path = Path(results_dir) / "training_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}", flush=True)
    print(summary_df.to_string(index=False), flush=True)
    return summary_df


if __name__ == "__main__":
    run_all_training()
