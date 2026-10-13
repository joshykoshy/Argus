# Test-set results and DEC-006 verdicts

Generated from `results/raw_metrics_v2.csv` with `python -m python.analysis.stats` (tables in `tables/`).
Data: 74 test patients x 13 DEC-005 conditions x 5 models (M0, M1, M4, M5, M6) x 3 seeds = 14,430 rows,
full-volume 3D Dice (mean of WT, TC, ET). Test set run once, as pre-registered.

## Bottom line

1. **Degradation augmentation is the effect.** Every augmented model (M1, M4, M5, M6) beats the clean-trained
   M0 by a wide margin at the worst condition (SNR 3, r 0.5, 5 mm): M4 minus M0 = +0.216 Dice
   (95 % CI 0.195 to 0.237), far above the seed noise.
2. **None of the three pre-registered hypotheses is supported.** The two-stream design (M4) does not help
   over the simpler M1, and it does not beat early fusion (M6).
3. **H1 and H3 point the wrong way, but only by about 0.004 Dice**, which is smaller than twice the
   seed-to-seed SD (0.014). Per DEC-006 that is "within training noise" whatever the p-value says.
   The honest reading is "no difference we can detect", with a slight lean towards M1 and M6 being at
   least as good as M4.

## Verdicts (Holm over H1-H3, alpha 0.05)

| | Hypothesis | Mean diff (95 % CI) | Raw p | Holm p | Verdict |
|---|---|---|---|---|---|
| H1 (primary) | M4 > M1 at worst | -0.0037 (-0.0086, +0.0004) | 0.0216 | 0.043 | Not supported (opposite direction, within noise) |
| H2 | M4's clean-to-worst drop < M1's | +0.0041 (-0.0008, +0.0096) | 0.704 | 0.704 | Not supported (within noise) |
| H3 | M4 > M6 at worst | -0.0044 (-0.0098, +0.0001) | 0.0006 | 0.0017 | Not supported (opposite direction, within noise) |

Positive mean diff means the first-named model (M4) is better. For H2 the mean and median disagree
(median -0.0008), so there is no consistent drop advantage either.

## Mean 3D Dice (seed- and patient-averaged)

| Model | Clean | SNR 5, r 1.0, 1 mm | Worst (SNR 3, r 0.5, 5 mm) | Drop clean to worst |
|---|---|---|---|---|
| M0 (clean-trained) | 0.786 | 0.674 | 0.549 | 0.237 |
| M1 | 0.830 | 0.822 | 0.768 | 0.061 |
| M4 | 0.822 | 0.816 | 0.765 | 0.057 |
| M5 | 0.819 | 0.811 | 0.763 | 0.055 |
| M6 | 0.826 | 0.820 | 0.769 | 0.057 |

The four augmented models are within about 0.012 Dice of each other at every condition shown (0.006 at the worst).

## Secondary comparisons (reported, outside the verdict)

| Comparison | Condition | Mean diff | Note |
|---|---|---|---|
| M4 vs M0 | worst | +0.216 | Real: well beyond seed noise |
| M4 vs M0 | clean | +0.036 | Within training noise: M0's clean seed SD is 0.048 (seeds disagree a lot) |
| M5 vs M4 | worst | -0.001 | Within training noise, p = 0.43 |
| M5 vs M4 | clean | -0.003 | Within training noise (Holm p = 0.002 is not meaningful at this size) |

## Other observations

- **False tumour on tumour-free slices** (mean `false_tumor_cm3`): M0 at the worst condition predicts about
  19.8 cm3 of tumour where there is none, versus about 0.4 cm3 for every augmented model. This is the same
  failure mode as finding C11, now triggered by noise rather than by training on tumour-only slices.
- **HD95** (median WT, mm): M0 jumps from 5.7 (clean) to 105 (worst); augmented models stay at 3.3 to 4.9 across the three conditions.
- **M0's clean Dice (0.786) matches its Step C validation value (0.798, mean of 3 seeds)**; the 0.84
  was the separate Step A run. Its seed SD (0.048) is about ten times that of the augmented models. Worth a sentence in the paper: the clean-trained model is
  also the least stable across training seeds.

## Sensitivity analysis: brain-masked predictions (post hoc)

Much of M0's false tumour under noise lies outside the brain. Zeroing predictions outside the brain mask and
repeating M4 vs M0 on all seeds (`evaluation/evaluate_masked.py`, `tables/masked_secondary_tests.csv`): worst
condition +0.126 Dice (95 % CI 0.112 to 0.142), beyond training noise (2 x seed SD 0.040); SNR 5 +0.059 and
clean +0.036, both within training noise. The augmentation effect at the worst condition is real but about
40 % smaller than the unmasked +0.216.

## Caveats

- Three seeds only, so the seed SD itself is a rough estimate.
- Differences of 0.004 Dice are tiny relative to what could matter clinically, even where p is small.
- Tests are paired Wilcoxon on seed-averaged per-patient Dice; the 74 patients are the unit of replication,
  not the seeds.

## Files

- `tables/hypotheses.csv`, `tables/secondary_tests.csv`: full test output
- `tables/dice_curves.csv`: 13-condition curves with 95 % bootstrap CIs (seed 42)
- `tables/headline_metrics.csv`: Dice, HD95 and false-tumour volume at clean / mid / worst
- `tables/seed_noise_floor.csv`: seed SD per model, condition and metric
