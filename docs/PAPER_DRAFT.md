# Paper draft

Status: **full draft for the team** (roadmap Stage 5), rewritten 2026-10-12 for
the redesigned study (DEC-005, DEC-006, DEC-007) and the test-set results in
`results/raw_metrics_v2.csv` / `docs/RESULTS_STATS.md`. Every number is
traceable to a file. Square-bracket notes:

- `[MATLAB: ...]` - a number or figure from Stage 3 (morphometrics on the
  real predictions); the slot says which script produces it.
- `[TEAM: ...]` - a decision or check the team still has to make.
- References: verified 2026-10-12 (`docs/paper/references_check.md`, BibTeX in
  `docs/paper/references.bib`). Entries marked "metadata only" there still need
  their content checked before submission.

Title: *Degradation-aware training, not frequency-band decomposition, makes
brain tumor segmentation robust to simulated low-field MRI*. IEEE-format
version: `docs/paper/main.tex` / `main.pdf`.

---

## Abstract (draft; finalise last)

**Purpose.** Low-field MRI could widen access to brain imaging, but its scans
are noisier and blurrier than the high-field data segmentation networks are
trained on. We tested whether splitting each image into low- and
high-spatial-frequency bands, processed in separate network streams, makes
brain tumor segmentation more robust to this degradation than conventional
training on degraded data.

**Methods.** On BraTS 2020 (258 training, 37 validation, 74 test patients) we
simulated low-field acquisition in k-space: band-limited complex noise at SNR
8, 5 and 3, halved in-plane resolution and 5 mm slices, giving 13 conditions
including clean. Five capacity-matched 2D U-Nets (about 4.2-4.4 M parameters)
were each trained with three seeds: a clean-trained baseline (M0), a U-Net
trained on degraded data (M1), a dual-stream frequency-band network (M4), a
low-band-only ablation (M5) and an early-fusion variant (M6). Three hypotheses
comparing M4 with M1 and M6 were pre-registered before training and tested on
full-volume 3D Dice with paired Wilcoxon tests and Holm correction.

**Results.** Training on degraded data was the decisive factor: at the
worst condition (SNR 3, 0.5 in-plane resolution, 5 mm slices) the augmented
models reached a mean Dice of 0.763-0.769, against 0.549 for the clean-trained
baseline (M4 minus M0 = +0.216, 95 % CI 0.195-0.237), and they removed the
baseline's spurious tumor on tumor-free slices (0.4 vs 19.8 cm3), about a
quarter of which lay outside the brain. None of the
pre-registered hypotheses was supported: the dual-stream network did not beat
the plain augmented U-Net (difference -0.004) or early fusion (-0.004), and
did not lose less accuracy from clean to worst. All differences between
augmented models were below twice the seed-to-seed SD.
Tumor-volume agreement followed the same pattern: at the worst condition the
augmented models kept a median absolute whole-tumor volume error of 7.0-7.8 %
(ICC 0.93-0.94, unchanged from clean), while the clean-trained baseline's
error rose to 25.8 % (ICC 0.74), with a near-zero mean bias hiding 95 % limits
of agreement of +/-73 cm3.

**Conclusion.** For segmentation under simulated low-field degradation,
what the network is trained on matters; how the input is split by frequency
does not.

---

## 1. Introduction

### 1.1 Low-field MRI and why segmentation robustness matters

Magnetic resonance imaging is the reference modality for brain tumor
diagnosis, treatment planning and follow-up, but conventional high-field
scanners (1.5-3 T) are expensive to buy, site and run, which concentrates
them in large hospitals in high-income settings [Arnold 2023; Ogbole 2018; Anazodo 2023]. In West Africa, for
example, a survey found 84 MRI units for about 373 million people, most of Nigeria's being low-field systems
[Ogbole 2018].
Low-field systems (defined as 0.25-1 T by Marques et al.; recent portable scanners
operate near 0.064 T [Iglesias 2023]) cost a fraction as much, need little shielding, and can be brought
to the bedside [Marques 2019; Sheth 2021]. The price is image quality:
signal grows roughly with field strength, so a low-field image has a lower
signal-to-noise ratio (SNR), and scanners compensate by acquiring larger
voxels, i.e. lower spatial resolution [Marques 2019].

Automated tumor segmentation networks, typically U-Net variants
[Ronneberger 2015], are trained and benchmarked almost entirely on
high-field data such as the BraTS challenge [Menze 2015; Bakas 2017].
If they are to be useful on low-field scanners, their accuracy must hold up
when SNR and resolution fall. A drop in overlap scores is only part of the
concern: tumor volume and shape measurements derived from the segmentation
are what inform clinical decisions (for example response assessment), so a
model that loses little Dice but systematically mis-sizes tumors would still
fail clinically.

### 1.2 Why decompose by spatial frequency

Noise and resolution loss do not affect all spatial frequencies equally. In
brain MRI, most image energy is at low spatial frequencies (smooth tissue
contrast, the overall shape of the tumor), while fine detail and edges sit at
high frequencies (Section 3.6 measures 92-97 % of clean-image power below
the cutoff used here). Thermal noise, in contrast, is spread evenly across
the acquired frequencies. Lowering SNR therefore hurts the high-frequency
band proportionally far more than the low band, and resolution loss removes
the highest frequencies outright.

This motivates the hypothesis tested here: separating each image into a
low-frequency and a high-frequency band *before* the network, and letting the
network process them in separate streams, might let it rely on the robust
low band when the high band becomes unreliable.

The fair test of that idea is not against a network trained on clean scans:
any model shown degraded scans during training will beat one that never saw
them. The question is whether the band-split architecture adds anything over
the same training data fed to a conventional network.

### 1.3 Contributions

1. A controlled comparison of five capacity-matched 2D U-Net configurations,
   three training seeds each, under 13 simulated low-field conditions
   (clean + 3 SNR levels x 2 in-plane resolutions x 2 slice thicknesses) with
   noise simulated inside the acquired k-space band.
2. Three hypotheses, the degradation range and the training protocol fixed
   in a dated decision log before any model was trained (DEC-005 to DEC-007),
   with the comparison that isolates architecture from training data (M4 vs
   M1) as the primary test, and a pre-specified rule that differences below
   twice the seed-to-seed SD are reported as training noise.
3. A clear negative result: frequency-band decomposition, as a dual-stream or
   early-fusion input, gives no measurable benefit over a plain U-Net trained
   on the same degraded data; degradation-aware training accounts for the
   whole robustness gain.
4. A morphometric validation layer: tumor volume, surface, sphericity,
   elongation and fragmentation for every prediction, with agreement
   analysis (Bland-Altman, ICC) that was itself validated on synthetically
   damaged masks with known errors.
5. An independent MATLAB re-implementation of the degradation and
   decomposition physics, numerically matched to the Python code
   (all 13 degradation conditions within 1.1e-6 of the mean brain intensity).

---

## 2. Related work

Full text: `docs/paper/related_work.md` (references verified); a condensed
version is merged into `docs/paper/main.tex`. It covers low-field and portable
MRI, deep-learning tumor segmentation, robustness to noise, resolution and
domain shift, frequency-domain and multi-band networks, and volumetric
agreement.

---

## 3. Methods

### 3.1 Dataset and splits

We used the BraTS 2020 training set [Menze 2015; Bakas 2017; Bakas
2018]: 369 pre-operative glioma patients (293 high-grade, HGG; 76
low-grade, LGG), each with four co-registered, skull-stripped MRI sequences
(T1, contrast-enhanced T1 [T1ce], T2, FLAIR) resampled to 1 mm isotropic
voxels on a 240 x 240 x 155 grid, and expert labels for necrotic/non-enhancing
core (label 1), edema (2) and enhancing tumor (4). Evaluation follows the
three BraTS regions: whole tumor (WT = 1 + 2 + 4), tumor core (TC = 1 + 4) and
enhancing tumor (ET = 4).

Patients were split once, stratified by grade with seed 42, into training
(258; 205 HGG / 53 LGG), validation (37; 29 / 8) and test (74; 59 / 15) sets
(`data/splits/split_v1.json`); the split was fixed before any model was
trained and never changed. The BraTS 2020 validation set was not used
(no public labels). All development (choice of degradation range, debugging
of the evaluation code) used the validation patients only; the test set was
evaluated once, after all 15 models were trained.

Each axial slice was cropped in-plane to 192 x 192 (rows 24-215, columns
28-219, zero-based). This crop retains at least 99.9 % of brain voxels for
every patient (checked in preprocessing) and loses 18 tumor voxels in total
across all 369 patients (one patient; 0.00005 % of tumor volume;
`matlab/check_crop_retention.m`); none of the 74 test patients is affected.
The label maps written by the evaluation pipeline (3.4) were checked against
the raw BraTS segmentations: identical grid, voxel spacing and labels for all
74 test patients, 0 differing voxels (`matlab/check_pred_grid.m`).

Ground-truth tumor sizes in the test set (median, range): WT 81.3 cm^3
(16.6-230.9), TC 28.6 cm^3 (0.8-159.9), ET 13.2 cm^3 (0-79.0); ET is absent
in 5 test patients, all LGG (Section 3.8).

### 3.2 Simulated low-field degradation

Degradation is applied to raw (un-normalised) intensities of each sequence
in three steps (`pilot/pilot.py::degrade_volume`).

**Slice thickness.** A thick slice is simulated by partial-volume averaging:
a t mm slice is the mean of t consecutive 1 mm axial slices (centred, with
edge slices replicated), kept on the 1 mm grid so every model sees 155
slices. t is 1 (unchanged) or 5 mm.

**In-plane resolution loss by k-space truncation.** Each axial slice x is
Fourier transformed and centred, K = fftshift(FFT2(x)). A rectangular mask
keeps the central fraction r of k-space along each axis (half-widths
round(r H / 2) and round(r W / 2) around the centre) and zeroes the rest.
With r = 0.5 the effective in-plane resolution is halved (2 mm). Because the
transform is applied to the magnitude image rather than raw complex scanner
data, this is a "pseudo-k-space" simulation, a common simplification.

**Thermal noise inside the acquired band.** A real scanner records noise
only at the k-space locations it samples. We therefore draw complex Gaussian
noise n = n_re + i n_im (n_re, n_im ~ N(0, 1)), restrict it to the same
k-space mask, and rescale it by 1 / sqrt(f), where f is the fraction of
k-space kept, so that its per-pixel standard deviation stays sigma. The
degraded image is the magnitude

  x_deg = | IFFT2(ifftshift(K . M_r)) + sigma . n_band |,

which yields Rician-distributed magnitudes in tissue and Rayleigh-distributed
background [Gudbjartsson 1995]. The noise level is set relative to the mean
clean intensity of the brain for that patient and sequence,
sigma = mu_brain / SNR, with mu_brain computed over all brain voxels of the
cropped volume. **SNR is a linear ratio** (SNR = 8 means the noise SD is 1/8
of the mean brain signal), not decibels.

**Conditions (DEC-005, fixed before training).** SNR in {8, 5, 3} x
r in {1.0, 0.5} x t in {1, 5} mm, plus clean, gives 13 conditions; the
pre-specified worst condition is SNR 3, r 0.5, 5 mm. The range was chosen in
a pilot on the 37 validation patients with a separately trained clean
baseline: every noisy condition cost it at least 0.17 Dice, so a robustness
gain would be detectable, and the worst kept condition cost 0.34. SNR 2
(up to 0.50 lost) was excluded as near-unusable.

**Deterministic test noise.** For validation and test, the noise generator
for each (patient, condition) is seeded with the first 4 bytes of the SHA-256
hash of the two identifiers, so every model sees exactly the same degraded
volume and all model comparisons are paired.

### 3.3 Frequency-band decomposition

Each (degraded, brain-z-score-normalised) slice is split into two bands with
complementary Gaussian filters in the centred 2D Fourier domain
(`python/physics/decomposition.py`; MATLAB `decompose_slice.m`):

  D(u, v) = sqrt((u / (H/2))^2 + (v / (W/2))^2)     (normalised radius, 1 = Nyquist along an axis)
  H_lp = exp(-D^2 / (2 D0^2)),   H_hp = 1 - H_lp
  x_low = Re(IFFT2(F . H_lp)),   x_high = Re(IFFT2(F . H_hp))

with D0 = 0.20. A Gaussian rather than an ideal (hard circular) cutoff avoids
ringing artefacts. Because H_lp + H_hp = 1 at every frequency, the bands sum
exactly to the input (maximum reconstruction error 5.1e-7, float32
rounding), so the decomposition discards no information; it only rearranges
it. D0 = 0.20 was fixed as a design choice motivated by the spectral
analysis (Section 3.6: 92-97 % of clean-image power lies below it); an early
search over D0 in {0.10, 0.15, 0.20, 0.30} on three validation patients was
too weak to select it empirically, and other cutoffs were not tested.

### 3.4 Models

All models are 2D U-Nets with five resolution levels, instance normalisation
and LeakyReLU(0.2), taking the four sequences as input and outputting three
sigmoid channels (WT, TC, ET). Parameter counts are matched within 4.5 %
(`tables/model_complexity.csv`). The decomposition (3.3) is part of the model
for M4-M6, so all five receive the same input.

| Model | Design | Training data | Params |
|---|---|---|---|
| M0 | U-Net | clean | 4.37 M |
| M1 | U-Net | degraded (augmented) | 4.37 M |
| M4 | **dual-stream U-Net**: separate encoders for the low and high band, fused at every scale into a shared decoder | degraded | 4.17 M |
| M5 | U-Net on the low band only (ablation) | degraded | 4.37 M |
| M6 | U-Net on low + high band as 8 input channels (early fusion) | degraded | 4.37 M |

Each comparison isolates one factor: M1 vs M0 is the effect of training on
degraded data; M4 vs M1 is the effect of the band-split architecture with
training data held fixed; M4 vs M6 is two streams vs one stream given the
same two bands; M5 vs M4 is what the high band contributes. Three further
configurations in the original plan (non-local-means denoising + U-Net,
low-pass + CLAHE + U-Net, and dual-stream with an enhanced high band) were
dropped because, as implemented, they reduced to exact duplicates of M1, M5
and M4.

**Training (DEC-007).** Every model was trained for 20 epochs on all brain
slices of the 258 training patients, tumor-free slices included, drawing
12,000 random slices per epoch, with batch size 32, AdamW (learning rate
3e-4, 1-epoch linear warm-up, cosine decay), mixed precision, random flips,
and a loss of binary cross-entropy plus soft Dice computed over the whole
batch per channel. Inputs are z-scored per slice and sequence inside the
brain mask after degradation. M1, M4, M5 and M6 draw one of the 13 DEC-005
conditions uniformly at random for each training slice; M0 sees clean slices
only. Each configuration was trained with seeds 0, 1 and 2 (15 runs; Colab
and Kaggle T4 GPUs).

**Model selection.** After every epoch, each run was evaluated on the full
volumes (all 155 slices) of the 37 validation patients under three
conditions (clean; SNR 5, r 1.0, 1 mm; worst); the checkpoint with the
highest mean 3D Dice over the three was kept. The same rule was used for
every model, including M0.

**Inference.** Each test volume is degraded once per condition, normalised,
and passed through every model in batches of 32 slices; channels are
thresholded at 0.5 and stacked into a 3D prediction. For the morphometric
analysis, seed-0 predictions at three conditions (clean; SNR 5, r 1.0, 1 mm;
worst) were written as label maps (ET over TC over WT) padded back to the
original 240 x 240 x 155 grid and affine.

**A note on the earlier protocol.** A first round of training used only the
ten largest-tumor slices per patient and was validated on tumor-bearing
slices. On full volumes those models predicted tumor on every tumor-free
slice (true 3D Dice about 0.2 against about 0.75 reported during training),
which also made the degradation appear harmless. All results here come from
the corrected protocol; the earlier models are not used.

### 3.5 Hypotheses and statistics (pre-registered, DEC-006)

All tests use full-volume 3D Dice averaged over WT, TC and ET, averaged per
patient over the three seeds, on the 74 test patients (the patient is the
unit of replication).

- **H1 (primary).** At the worst condition, M4 > M1.
- **H2.** M4's clean-to-worst Dice drop is smaller than M1's.
- **H3.** At the worst condition, M4 > M6 (two streams vs early fusion).

Each is a two-sided paired Wilcoxon signed-rank test [Wilcoxon 1945] with
Holm correction over the three [Holm 1979], alpha 0.05. A hypothesis is
supported only if the difference is significant, in the predicted
direction, and at least twice the seed-to-seed SD; a difference smaller than
that is reported as "within training noise" whatever its p-value. The seed
SD is the standard deviation across the three seeds of a model's
patient-averaged Dice at that condition (for H2, of its clean-to-worst
drop), taking the larger of the two models compared.

Secondary comparisons, reported but outside the verdict: M4 vs M0 and M5 vs
M4, at clean and worst (Holm over the four). Effect sizes are mean and
median paired differences with 95 % percentile bootstrap intervals over
patients (10,000 resamples, seed 42); the same bootstrap gives the
per-condition Dice curves.

Segmentation metrics: Dice per region (1 when reference and prediction are
both empty, 0 when only one is); 95th-percentile Hausdorff distance (HD95,
mm) at clean, SNR 5 and worst only, with the BraTS convention (0 when both
masks are empty, 373.13 mm when exactly one is); and false-tumor volume, the
predicted WT volume (cm^3) on slices containing no true tumor.
(`evaluation/evaluate_v2.py`, `python/analysis/stats.py`.)

### 3.6 Implementation verification and spectral analysis

**Parity.** The degradation of 3.2 was re-implemented independently in
MATLAB (`matlab/reference/degrade_volume.m`) and compared with the Python
function used for training and evaluation (`pilot.pilot.degrade_volume`) on
the same 9-slice, 2-channel, 192 x 192 volume and the same noise arrays,
for all 13 conditions (`test_degrade_volume_parity.m`). The largest
difference was 4.3e-4 intensity units, 1.1e-6 of the mean brain intensity,
i.e. float32 rounding (`results/degrade_volume_parity.csv`). The
decomposition (`decompose_slice.m`) matched the PyTorch code on five random
slices for four cutoffs D0 in {0.10, 0.15, 0.20, 0.30} with a maximum
absolute difference of 4.8e-7 for intensities in [0.1, 1]
(`tests/test_degradation.py::test_matlab_python_parity`, tolerance 1e-4). Rerunning the MATLAB side on a second
machine and MATLAB version (R2026b vs R2025a) reproduced the stored outputs
exactly (`matlab/env_check.m`).

**Where the image energy is.** On the axial slice with the largest tumor
area for 12 test patients (FLAIR and T1ce; `matlab/spectral_check.m`), the
fraction of clean-image spectral power at normalised radius D < 0.20 was
95.1 % for FLAIR (minimum 92.5 %) and 91.9 % for T1ce (minimum 88.4 %),
excluding the zero-frequency term (96.9 % and 95.2 % including it). Under the
Gaussian split the low band carries 97.8 % (FLAIR) and 96.6 % (T1ce) of the
band energy. This is independent of the degradation.

**How the degradation hits each band.** For each DEC-005 condition we
computed a band SNR: clean-image power in the band divided by the power of
the degradation error (degraded minus clean slice: noise plus lost in-plane
and through-plane resolution) in the same band, in dB (`degrade_volume.m` on
a 5-slice slab around each slice, noise seed 42).

| SNR | r 1.0, 1 mm | r 1.0, 5 mm | r 0.5, 1 mm | r 0.5, 5 mm |
|---|---|---|---|---|
| | *FLAIR, low / high band (dB)* | | | |
| 8 | 16.4 / 2.6 | 16.3 / 1.7 | 16.1 / 1.3 | 16.0 / 0.9 |
| 5 | 12.3 / -1.4 | 12.3 / -1.8 | 12.1 / -1.8 | 12.0 / -2.0 |
| 3 | 7.8 / -5.7 | 7.8 / -5.8 | 7.6 / -5.6 | 7.6 / -5.7 |
| | *T1ce, low / high band (dB)* | | | |
| 8 | 15.1 / 3.4 | 14.7 / 1.1 | 14.8 / 1.9 | 14.4 / 0.7 |
| 5 | 11.0 / -0.6 | 10.9 / -1.7 | 10.8 / -1.1 | 10.6 / -1.7 |
| 3 | 6.5 / -4.9 | 6.5 / -5.4 | 6.3 / -4.9 | 6.3 / -5.2 |

(`results/spectral_check.csv`, Figure `figures/spectral_band_snr.png`.)

The low band keeps more anatomy than error at every condition (6-16 dB),
while the high band falls below 0 dB, more error than anatomy, from SNR 5
on, reaching about -5 to -6 dB at SNR 3. Noise sets the band SNR almost
entirely; halving resolution and 5 mm slices lower the high band by at most
about 2 dB at SNR 8 and change little at SNR 3, where noise already
dominates. The premise of band separation therefore holds strongly in this
range: at the noise levels where the clean-trained model fails, most of the
high band is degradation. Caveat: the high-band anatomical power includes
the sharp brain/background edge of skull-stripped images, which inflates
high-band SNR relative to the tumor interior.

### 3.7 Morphometric biomarkers

For every segmentation (ground truth or prediction, identical code:
`matlab/tumor_biomarkers.m`) and each region (WT, TC, ET) we computed:
volume (voxel count x voxel volume from the NIfTI header), surface area,
sphericity psi = pi^(1/3) (6V)^(2/3) / A [Wadell 1935], elongation
(ratio of the second to the first principal axis length), the number of
26-connected components, and the fraction of volume outside the largest
component. Empty regions are recorded as volume 0 with undefined (NaN) shape
measures.

Surface area on a voxel grid needs care: counting exposed voxel faces
overestimates the true area of a smooth surface by 45-55 % regardless of
size (validated on digital spheres, `figures/surface_area_validation.png`).
We used MATLAB's `regionprops3` SurfaceArea, which was within 1.7 % of the
analytic area for spheres of radius >= 8 mm, and cross-checked it with a
marching-cubes mesh [Lorensen 1987] after Gaussian smoothing
(sigma = 0.5 voxel; +2-4 % on spheres). Because `regionprops3` reports the
surface of only one component when a single label spans several
disconnected components, surface area was computed per connected component
and summed (regression-tested). The two estimators then agreed closely on
the 74 test tumors (Spearman rho 0.992-0.996 for sphericity).

### 3.8 Agreement analysis

Agreement between predicted and ground-truth biomarkers is reported per
model and condition as (i) percentage volume error, (ii) Bland-Altman bias
and 95 % limits of agreement [Bland 1986], and (iii) the intraclass
correlation ICC(A,1) (two-way, absolute agreement, single measure;
[McGraw 1996], equivalent to ICC(2,1) of [Shrout 1979]),
alongside the consistency form ICC(C,1). Confidence intervals are
percentile bootstrap intervals over patients (2000 resamples, seed 42).
The implementation (`matlab/agreement_stats.m`) reproduces the worked
example of Shrout & Fleiss (ICC(2,1) = 0.29, ICC(3,1) = 0.71).

The full pipeline was validated before use on real predictions by applying
known synthetic errors to the 74 test ground truths (dilation/erosion by 1
and 2 mm, a 2-voxel shift, deletion of an 8 mm ball, three spurious 3 mm
blobs; `matlab/agreement_validation.m`). Exactly known volume changes were
recovered to the voxel; dilation-induced volume changes matched a
surface-area (Steiner) prediction calibrated on a sphere to within 1-3 %
(median ratio 1.006-1.032).

### 3.9 Qualitative case selection

Five test patients were selected for qualitative display before any
predictions were inspected, using only ground-truth WT volume, ET presence
and grade: the HGG patients (with ET) closest to the 10th, 50th and 90th
percentile of HGG WT volume, the LGG patient with ET closest to that
subgroup's median, and the LGG patient without ET closest to that subgroup's
median (ties: lowest ID). Selected: BraTS20_Training_008, _167, _236, _292,
_281 (`results/diagnostic_selection.csv`).

---

## 4. Results

All numbers: 74 test patients, mean over patients of the seed-averaged
per-patient value, from `results/raw_metrics_v2.csv` (14,430 rows) via
`python -m python.analysis.stats` (`tables/`).

### 4.1 Accuracy across conditions

**Table 2.** Mean 3D Dice (WT/TC/ET average) at all 13 conditions
(`tables/dice_curves.csv`; 95 % bootstrap CIs there).

| SNR | r | Slice (mm) | M0 | M1 | M4 | M5 | M6 |
|---|---|---|---|---|---|---|---|
| clean | 1.0 | 1 | 0.786 | 0.830 | 0.822 | 0.819 | 0.826 |
| 8 | 1.0 | 1 | 0.687 | 0.825 | 0.820 | 0.813 | 0.823 |
| 8 | 1.0 | 5 | 0.680 | 0.817 | 0.813 | 0.807 | 0.815 |
| 8 | 0.5 | 1 | 0.659 | 0.817 | 0.812 | 0.810 | 0.816 |
| 8 | 0.5 | 5 | 0.649 | 0.808 | 0.805 | 0.802 | 0.808 |
| 5 | 1.0 | 1 | 0.674 | 0.822 | 0.816 | 0.811 | 0.820 |
| 5 | 1.0 | 5 | 0.660 | 0.814 | 0.809 | 0.804 | 0.812 |
| 5 | 0.5 | 1 | 0.630 | 0.805 | 0.800 | 0.798 | 0.805 |
| 5 | 0.5 | 5 | 0.618 | 0.796 | 0.792 | 0.791 | 0.796 |
| 3 | 1.0 | 1 | 0.635 | 0.814 | 0.806 | 0.802 | 0.812 |
| 3 | 1.0 | 5 | 0.623 | 0.804 | 0.796 | 0.793 | 0.802 |
| 3 | 0.5 | 1 | 0.558 | 0.779 | 0.774 | 0.773 | 0.779 |
| 3 | 0.5 | 5 | **0.549** | **0.768** | **0.765** | **0.763** | **0.769** |

Figure: `figures/dice_curves.png` (`docs/paper/fig_dice_curves.py`).

The clean-trained M0 lost accuracy under every degradation, and noise was
the dominant factor: SNR 8 alone cost it 0.099 Dice, going from SNR 8 to
SNR 3 a further 0.052, and halving in-plane resolution with 5 mm slices at
SNR 3 another 0.086 (clean to worst: 0.237). The four augmented models were
almost insensitive to noise (M1: 0.005 at SNR 8, a further 0.011 at SNR 3);
most of their remaining loss came from resolution at the highest noise
level (0.046 for M1 between SNR 3 full-resolution and the worst condition;
clean to worst: 0.055-0.061).

The four augmented models were within 0.012 Dice of each other at every
condition, and within 0.006 at the worst. The plain augmented U-Net (M1)
was numerically the best or tied for best at every condition.

**Per region (worst condition).** WT was most robust (M1 0.863, M4 0.863, M0
0.702), ET least (M1 0.658, M4 0.653, M0 0.437), with TC in between (M1
0.784, M4 0.777, M0 0.508) (`tables/headline_metrics.csv`).

### 4.2 Pre-registered hypotheses

**Table 3.** DEC-006 tests at the worst condition (`tables/hypotheses.csv`).
Positive difference = M4 better.

| | Hypothesis | Mean diff (95 % CI) | Median diff | Raw p | Holm p | 2 x seed SD | Verdict |
|---|---|---|---|---|---|---|---|
| H1 | M4 > M1 | -0.0037 (-0.0086, +0.0004) | -0.0027 | 0.022 | 0.043 | 0.014 | Not supported |
| H2 | M4 drop < M1 drop | +0.0041 (-0.0008, +0.0096) | -0.0008 | 0.70 | 0.70 | 0.005 | Not supported |
| H3 | M4 > M6 | -0.0044 (-0.0098, +0.0001) | -0.0037 | 0.0006 | 0.0017 | 0.014 | Not supported |

None of the three hypotheses was supported. For H1 and H3 the Wilcoxon test
was significant after correction, but in the opposite direction to the
prediction (M1 and M6 slightly better than M4), and the differences,
about 0.004 Dice, are less than a third of the pre-specified training-noise
threshold; we read them as "no detectable difference". For H2 the mean
favours M4 and the median favours M1, and neither is significant.

### 4.3 Secondary comparisons

**Table 4.** (`tables/secondary_tests.csv`; Holm over the four.)

| Comparison | Condition | Mean diff (95 % CI) | Holm p | Reading |
|---|---|---|---|---|
| M4 - M0 | worst | +0.216 (0.195, 0.237) | 3e-13 | Real: 9x the training-noise threshold |
| M4 - M0 | clean | +0.036 (0.029, 0.043) | 8e-12 | Within training noise (M0 seed SD 0.048) |
| M5 - M4 | worst | -0.001 (-0.006, +0.003) | 0.43 | Within training noise |
| M5 - M4 | clean | -0.003 (-0.008, +0.003) | 0.002 | Within training noise |

The whole robustness gain of M4 over M0 is therefore attributable to the
degraded training data, which M1 shares (Table 2). Discarding the high band
entirely (M5) cost nothing measurable.

### 4.4 False tumor, boundary error and stability

**Table 5.** At clean / SNR 5 (r 1.0, 1 mm) / worst (`tables/headline_metrics.csv`).

| Model | False tumor, cm^3 (mean) | HD95 WT, mm (median) |
|---|---|---|
| M0 | 0.47 / 13.1 / 19.8 | 5.7 / 100.1 / 105.4 |
| M1 | 0.07 / 0.31 / 0.41 | 3.4 / 3.9 / 4.4 |
| M4 | 0.12 / 0.41 / 0.47 | 3.8 / 4.1 / 4.9 |
| M5 | 0.10 / 0.43 / 0.41 | 3.4 / 4.0 / 4.3 |
| M6 | 0.07 / 0.40 / 0.37 | 3.3 / 4.1 / 4.2 |

Under noise, M0 predicted 13-20 cm^3 of tumor on slices with none, on
average, far from the real tumor: its median WT HD95 jumped from 5.7 mm to
about 100 mm, while every augmented model stayed below 5 mm. Overlap scores
understate this failure, since a few spurious islands far from the tumor
cost little Dice but make the segmentation clinically unusable.

**Where M0's false tumor lies.** The 3D renderings (4.6) showed that much of
M0's spurious tumor is outside the brain: in the background around the head
and on slices above and below it. Measured on the seed-0 predictions
(`results/pred_outside_brain.csv`, brain = FLAIR > 0), 24 % of M0's predicted
WT volume at the worst condition (23.4 cm^3 per patient) and 16 % at SNR 5
lay outside the brain, against under 1 % for every augmented model
(0.2-0.6 cm^3). No post-processing was pre-specified, so the results above
use the raw predictions. As a sensitivity analysis, zeroing predictions
outside the brain mask:

| Seed 0, WT | M0 SNR 5 | M0 worst | Augmented, worst |
|---|---|---|---|
| Dice, raw -> masked | 0.784 -> 0.883 | 0.647 -> 0.763 | 0.861-0.864 -> 0.863-0.866 |
| Median abs. volume error, raw -> masked | 19.5 % -> 6.3 % | 25.8 % -> 12.1 % | 7.0-7.8 % -> 7.0-7.8 % |

A brain mask, standard in any clinical pipeline, would remove most of M0's
deficit at SNR 5 and about half of it at the worst condition; the augmented
models remain clearly better there (WT Dice +0.10, half the volume error).
The pre-registered comparisons between augmented models are unaffected.
Repeated on brain-masked predictions of all three seeds (inference rerun on CPU
with the same checkpoints, `evaluation/evaluate_masked.py`, post hoc;
`tables/masked_secondary_tests.csv`):

| M4 - M0, masked | Mean diff (95 % CI) | Holm p | 2 x seed SD | Reading |
|---|---|---|---|---|
| worst | +0.126 (0.112, 0.142) | 2e-13 | 0.040 | Real (unmasked: +0.216) |
| SNR 5 | +0.059 (0.050, 0.068) | 2e-13 | 0.099 | Within training noise |
| clean | +0.036 (0.029, 0.043) | 2e-12 | 0.096 | Within training noise |

Masking cut M0's mean false tumor at the worst condition from 19.7 to
3.2 cm^3. About 40 % of the unmasked worst-condition gain is tumor drawn
outside the brain; the remaining +0.13 is robust. The CPU rerun draws
different (statistically identical) noise from the GPU run, so individual
raw scores differ slightly from `raw_metrics_v2.csv` (clean scans agree to
1e-4); the model averages agree (M0 worst 0.549 in both).

M0 was also the least reproducible model: the seed-to-seed SD of its clean
Dice was 0.048, against 0.0004-0.005 for the augmented models
(`tables/seed_noise_floor.csv`). Its clean test Dice (0.786) is consistent
with its validation value from the same runs (0.798, mean of three seeds).

### 4.5 Morphometric agreement

Seed-0 predictions of all five models at clean, SNR 5 (r 1.0, 1 mm) and
worst, 74 test patients each (`results/volumetrics_pred.csv`,
`results/agreement_stage3.csv`; Figures `figures/agreement_stage3_*.png`).

**Table 6.** Whole-tumor volume agreement, clean / worst. Bias and 95 %
limits of agreement (LoA) in cm^3; error = median absolute volume error.

| Model | Error (%) | Bias (LoA) clean | Bias (LoA) worst | ICC(A,1) clean | ICC(A,1) worst (95 % CI) |
|---|---|---|---|---|---|
| M0 | 5.8 / 25.8 | -3.0 (-34, +28) | -0.2 (-74, +73) | 0.957 | 0.737 (0.55-0.89) |
| M1 | 6.5 / 7.1 | -4.7 (-35, +26) | -5.8 (-42, +30) | 0.956 | 0.938 (0.87-0.98) |
| M4 | 6.8 / 7.0 | -4.6 (-38, +29) | -4.8 (-40, +31) | 0.948 | 0.942 (0.88-0.98) |
| M5 | 6.3 / 7.8 | -4.2 (-38, +30) | -5.3 (-45, +34) | 0.947 | 0.928 (0.84-0.98) |
| M6 | 6.8 / 7.2 | -4.3 (-35, +27) | -5.1 (-41, +31) | 0.956 | 0.940 (0.87-0.98) |

**The augmented models measured tumor volume about as well at the worst
condition as on clean scans.** Their median absolute volume error at the worst
condition was 7.0-7.8 % for WT, 12.7-14.5 % for TC and 17.1-18.5 % for ET
(clean: 6.3-6.8 %, 7.8-9.8 %, 11.4-13.9 %), and their limits of agreement
widened by only a few cm^3. They slightly underestimated WT at every
condition (bias -4 to -6 cm^3; for M1 the bias CI excludes zero). The four
were indistinguishable from each other, as in Dice.

**The clean-trained model's volumes became unreliable, and its mean bias
hid it.** At the worst condition M0's median WT volume error rose from 5.8 %
to 25.8 %, TC to 65.9 % and ET to 37.5 %, ICC(A,1) fell to 0.74 (WT), 0.58
(TC) and 0.73 (ET), and the WT limits of agreement widened from about +/-30
to +/-73 cm^3. Yet its WT mean bias was -0.2 cm^3: spurious tumor added
elsewhere and true tumor missed cancel out on average. The per-patient
errors run both ways (Figure, Bland-Altman). A bias-only summary would have
called M0 the most accurate model; the limits of agreement and ICC do not.
At SNR 5 the same model overestimated instead (WT bias +12.1 cm^3, median
error +17.8 %).

**Fragmentation.** M0's noisy predictions split into very many pieces (median
extra connected components over ground truth: 1,116 at SNR 5 and 1,911 at the
worst condition for WT), the scattered false tumor of 4.4. The augmented
models also fragmented somewhat under noise (median 43-69 extra WT pieces at
the worst condition, 0-2 on clean scans), but these are small specks: the
volume they carry does not show in the volume error.

**ET where there is none.** All five test patients without ET (all LGG)
received a non-empty ET prediction from every model at every condition (M1
on clean scans: 4 of 5). No model missed a region that was present.

### 4.6 Qualitative cases

Figures `figures/diagnostic_<case>_<id>.png` show, for the five cases of
3.9, the clean and worst-condition FLAIR, the ground truth, the M0 and M4
predictions (seed 0) on the degraded slice, and the band decomposition;
`figures/render3d_stage3_<case>_<model>.png` show the 3D truth, prediction and
WT surface-error map (signed distance of each predicted surface point to the
true boundary; renderer validated on known 2 mm dilation/erosion, 3.8).

**Table 7.** Worst condition, seed 0 (`results/render3d_stage3_surface_error.csv`).

| Case | Mean 3D Dice M0 / M4 | Mean abs. surface distance (mm) M0 / M4 | Surface > 2 mm off, M0 / M4 |
|---|---|---|---|
| 1 HGG small | 0.13 / 0.56 | 78.3 / 15.7 | 89 % / 33 % |
| 2 HGG typical | 0.46 / 0.78 | 51.9 / 4.0 | 86 % / 31 % |
| 3 HGG large | 0.03 / 0.23 | 54.0 / 7.6 | 97 % / 69 % |
| 4 LGG with ET | 0.25 / 0.77 | 63.5 / 7.0 | 91 % / 23 % |
| 5 LGG no ET | 0.31 / 0.44 | 35.5 / 4.2 | 85 % / 48 % |

In every case M0's predicted surface is dominated by scattered islands, many
outside the brain, which put its mean surface distance at 35-78 mm. M4's
surface follows the true boundary (mean 4-8 mm) apart from a few small
distant specks, which explain the larger mean in case 1 (small tumor, so
the specks weigh more). Two cases are hard for every model: in case 3 (the
largest tumor) all models under-segment the core even on clean scans (TC
Dice about 0.3-0.4, seed-averaged) and at the worst condition M4 recovers
half the WT volume (84 of 170 cm^3); in case 5 the mean Dice is pulled down
by ET predicted in a patient without ET (ET Dice 0 by the BraTS convention),
while its WT Dice is 0.84.

---

## 5. Discussion

**Degradation-aware training is the effect; the architecture is not.**
Training on simulated low-field data raised worst-condition Dice from 0.55 to
0.76-0.77 and removed the spurious tumor that the clean-trained network drew
on noisy scans. Given that training data, splitting the input into frequency
bands, whether into two streams (M4) or as extra channels (M6), gave nothing
measurable, and dropping the high band (M5) cost nothing either. The study
was designed so that this could be detected: the hypotheses compared M4
with M1, the configuration that shares its training data, rather than with
the clean baseline, against which any augmented model looks good.

**Why the decomposition did not help.** The split is a fixed, linear,
invertible filter (the bands sum to the input; 3.3). A convolutional
network can learn equivalent filtering in its first layers, so the split
offers no information the plain U-Net lacks; it can only change what is
easy to learn. Once the network has seen degraded scans in training, it
already learns to ignore unreliable fine detail, which is exactly the job
the band split was meant to do. That M5, which never sees the high band,
matches M4 suggests the networks lean on low-frequency content anyway,
consistent with the low band carrying 92-97 % of the image power and with
the high band being mostly degradation from SNR 5 on (3.6). The premise of
the design was right; the network simply did not need the split to act on
it.

**Statistical versus practical significance.** H1 and H3 were significant
in the wrong direction at a difference of about 0.004 Dice. With 74 paired
patients, a small but consistent shift is detectable; whether it is a
property of the architecture or of these three training runs is not, since
the seed-to-seed spread is larger. The pre-registered noise rule exists for
this case, and we do not interpret those differences beyond "M4 is not
better".

**Clinical reading.** A segmentation that mislabels distant tissue as tumor
inflates volume and fragments the lesion even when overlap looks
acceptable; M0's false-tumor volume and HD95 under noise show this
directly, and the volume analysis (4.5) shows its consequence. Part of
that failure, tumor drawn outside the brain, would be removed by the brain
masking any clinical pipeline applies (4.4); the rest, inside the brain,
would not: M0's
per-patient WT volume could be off by up to about 73 cm^3 at the worst
condition, while the augmented models stayed within the same roughly
+/-35 cm^3 limits they had on clean scans. For scale, expert raters agree on
preoperative glioma volume with ICC(A,1) of 0.92 (grade II-III) to 0.99
(glioblastoma enhancing tumor), and novices on grade II-III glioma with 0.73
[Visser 2019]. The augmented models' worst-condition WT ICC (0.93-0.94) lies
at the lower end of the expert range and M0's (0.74) at novice level,
although that study rated different regions on clinical scans. The augmented
models' ET error (about 18 %) is larger, as expected for a thin shell
(below). Two practical warnings: a near-zero
mean bias can coexist with large per-patient error, so agreement must be
judged from the limits of agreement and ICC; and every model predicted ET in
patients who have none, so ET presence from these models should not be
trusted in LGG.

**Points from the morphometric validation (hold regardless of model).**

- **ET is intrinsically fragile.** A 1 mm boundary error changes ET volume
  by a median 48 % but WT by 17 % (synthetic validation), because ET is a
  thin, small shell with a high surface-to-volume ratio. ET was indeed the
  least robust region for every model (4.1).
- **Overlap, volume and false positives are different questions.** In the
  synthetic validation a 2 mm shift kept ICC = 1.000 while ET Dice fell to
  0.71, and three spurious blobs kept Dice at 0.997 and ICC at 1.000; only the
  component count detected them. Report all three kinds of metric.
- **ICC can flatter a biased method, and bias can flatter a noisy one.** A
  systematic +17 % volume error still gave ICC(A,1) = 0.957 (synthetic
  validation), and M0's -0.2 cm^3 WT bias at the worst condition came with
  +/-73 cm^3 limits of agreement (4.5). Report bias, limits and ICC together.

**Implication.** For deploying segmentation on low-field scanners, effort
is better spent on realistic degradation during training than on
frequency-specific architectures of this kind.

## 6. Limitations

- **Simulated degradation on high-field data.** The noise, resolution and
  slice-thickness model operates on magnitude images (pseudo-k-space) and
  omits low-field contrast changes, B0/B1 inhomogeneity and other artefacts.
  No real low-field scans with tumor labels were available to validate on.
- **In-distribution degradation.** Augmented models were trained on the same
  13 conditions they were tested on, so their robustness is a best case;
  harsher or different degradation (e.g. SNR 2) was not tested.
- **2D networks of about 4 M parameters, 20 epochs, one dataset.** A
  frequency split might behave differently in 3D or much larger networks.
- **Three seeds.** The seed SD that defines the training-noise threshold is
  itself a rough estimate.
- **One decomposition.** A single Gaussian cutoff (D0 = 0.20) was tested,
  chosen by design rather than tuned.
- **Morphometrics on seed 0 at three conditions** only; the other ten
  conditions and seeds 1-2 have Dice and volumes (`raw_metrics_v2.csv`) but
  no shape measures or agreement analysis.
- **Pre-registration is internal** (a dated decision log in the repository),
  not a public registry.
- MATLAB figure noise realisations differ from the Python test set (same
  statistics).
- **No post-processing.** Predictions were evaluated raw, as pre-specified;
  brain masking (4.4) or connected-component filtering would reduce false
  tumor, mostly for M0.

## 7. Conclusion

Under simulated low-field degradation, training on degraded scans made a
U-Net far more robust (+0.22 Dice at the worst condition, with spurious
tumor eliminated), while frequency-band decomposition, as a dual-stream or
early-fusion design, added nothing measurable over a conventional U-Net
trained on the same data. All three pre-registered hypotheses for the
band-split design were not supported. With degradation-aware training,
tumor volumes stayed about as reliable at the worst condition as on clean
scans (median whole-tumor error about 7 %); without it, per-patient volumes
became unreliable even though their average error was close to zero.

---

## References

Verified 2026-10-12 against Crossref / PubMed / arXiv: details in `docs/paper/references_check.md`,
BibTeX in `docs/paper/references.bib`. The related-work references are in that file and are not repeated here.

- Anazodo UC, Ng JJ, et al. A framework for advancing sustainable magnetic resonance
  imaging access in Africa. *NMR in Biomedicine* 2023;36(3):e4846.
- Arnold TC, Freeman CW, Litt B, Stein JM. Low-field MRI: clinical promise and
  challenges. *Journal of Magnetic Resonance Imaging* 2023;57(1):25-44.
- Bakas S, et al. Advancing The Cancer Genome Atlas glioma MRI collections
  with expert segmentation labels and radiomic features. *Scientific Data*
  2017.
- Bakas S, et al. Identifying the best machine learning algorithms for brain
  tumor segmentation, progression assessment, and overall survival
  prediction in the BRATS challenge. arXiv:1811.02629, 2018.
- Bland JM, Altman DG. Statistical methods for assessing agreement between
  two methods of clinical measurement. *Lancet* 1986.
- Gudbjartsson H, Patz S. The Rician distribution of noisy MRI data.
  *Magnetic Resonance in Medicine* 1995.
- Holm S. A simple sequentially rejective multiple test procedure.
  *Scandinavian Journal of Statistics* 1979.
- Iglesias JE, Schleicher R, Laguna S, Billot B, et al. Quantitative brain morphometry of
  portable low-field-strength MRI using super-resolution machine learning.
  *Radiology* 2023;306(3):e220522.
- Lorensen WE, Cline HE. Marching cubes: a high resolution 3D surface
  construction algorithm. *SIGGRAPH* 1987.
- Marques JP, Simonis FFJ, Webb AG. Low-field MRI: an MR physics perspective.
  *Journal of Magnetic Resonance Imaging* 2019.
- McGraw KO, Wong SP. Forming inferences about some intraclass correlation
  coefficients. *Psychological Methods* 1996.
- Menze BH, et al. The multimodal brain tumor image segmentation benchmark
  (BRATS). *IEEE Transactions on Medical Imaging* 2015.
- Ogbole GI, Adeyomoye AO, Badu-Peprah A, Mensah Y, Nzeh DA. Survey of magnetic
  resonance imaging availability in West Africa. *Pan African Medical Journal*
  2018;30:240.
- Visser M, Muller DMJ, van Duijn RJM, et al. Inter-rater agreement in glioma
  segmentations on longitudinal MRI. *NeuroImage: Clinical* 2019;22:101727.
- Ronneberger O, Fischer P, Brox T. U-Net: convolutional networks for
  biomedical image segmentation. *MICCAI* 2015.
- Sheth KN, et al. Assessment of brain injury using portable, low-field
  magnetic resonance imaging at the bedside of critically ill patients.
  *JAMA Neurology* 2021.
- Shrout PE, Fleiss JL. Intraclass correlations: uses in assessing rater
  reliability. *Psychological Bulletin* 1979.
- Wadell H. Volume, shape, and roundness of quartz particles. *Journal of
  Geology* 1935.
- Wilcoxon F. Individual comparisons by ranking methods. *Biometrics
  Bulletin* 1945.
