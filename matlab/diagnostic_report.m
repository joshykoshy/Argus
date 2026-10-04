% DIAGNOSTIC_REPORT  Per-patient diagnostic figure for 5 blindly selected test cases.
% ECTE408 Low-Field MRI Brain Tumor Segmentation Study (roadmap step 1.4)
%
% Run from repo root:  matlab -batch "run('matlab/diagnostic_report.m')"
% Needs results/volumetrics_gt.csv (matlab/volumetrics.m).
%
% PATIENT SELECTION RULE (fixed before any prediction exists; uses only the
% ground-truth whole-tumor volume, ET presence and grade - never model output):
%   1-3. HGG with ET present: the patients whose WT volume is closest to the
%        10th, 50th and 90th percentile of that group (small / typical / large)
%   4.   LGG with ET present: closest to that group's median WT volume
%   5.   LGG with NO ET:      closest to that group's median WT volume
%   Ties: lowest patient ID. Written to results/diagnostic_selection.csv.
%
% PANELS (axial slice with the largest true WT area; 192x192 crop = what the
% network sees, python/data/preprocess.py rows 24:216, cols 28:220):
%   1  FLAIR, clean
%   2  FLAIR degraded at SNR 8, r 0.5 (worst condition), via the parity-verified
%      degrade_slice.m. Noise: MATLAB RandStream, seed 42 (the Python test noise
%      uses a SHA256-seeded torch generator that MATLAB cannot replay; the noise
%      is statistically identical, the exact realisation differs).
%   3  ground-truth overlay (WT blue, TC orange, ET red outlines)
%   4  prediction overlay - placeholder until Stage 3 (set USE_PREDICTIONS)
%   5  band decomposition of the degraded input at D0 = 0.20 (decompose_slice.m):
%      low band, high band, and the clean image's high band for comparison
% Intensities follow the network's input pipeline: degrade raw intensities,
% then z-score inside the brain (python/data/dataset.py:181-192), then split.

%% ---- Configuration ----
cd(fileparts(fileparts(mfilename('fullpath'))));
addpath('matlab', 'matlab/reference');
DATA_ROOT  = 'archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData';
GT_CSV     = 'results/volumetrics_gt.csv';
SEL_CSV    = 'results/diagnostic_selection.csv';
OUT_FMT    = 'figures/diagnostic_%d_%s.png';
SNR = 8;  R = 0.5;  D0 = 0.20;  SEED = 42;
ROWS = 25:216;  COLS = 29:220;
USE_PREDICTIONS = false;              % Stage 3 only, when ALL predictions exist
PRED_FMT   = 'results/predictions/%s_%s_snr8_r0.5.nii.gz';
PRED_MODEL = 'M4';
COL = [0.00 0.45 0.70; 0.95 0.60 0.00; 0.85 0.15 0.15];

%% ---- 1. Blind selection from ground truth ----
gt = readtable(GT_CSV, 'TextType', 'string');
wt = gt(gt.region == "WT", :);
et = gt(gt.region == "ET", :);
[~, ia] = ismember(wt.patient_id, et.patient_id);
wt.has_et = et.voxels(ia) > 0;
wt = sortrows(wt, 'patient_id');                 % so ties go to the lowest ID
q = @(v, p) interp1(linspace(0, 100, numel(v)), sort(v), p);
groups = {
    'HGG small (10th pct)',  wt.grade == "HGG" & wt.has_et,  10
    'HGG typical (median)',  wt.grade == "HGG" & wt.has_et,  50
    'HGG large (90th pct)',  wt.grade == "HGG" & wt.has_et,  90
    'LGG with ET (median)',  wt.grade == "LGG" & wt.has_et,  50
    'LGG no ET (median)',    wt.grade == "LGG" & ~wt.has_et, 50
};
sel = table();
for g = 1:size(groups, 1)
    pool = wt(groups{g, 2}, :);
    target = q(pool.volume_cm3, groups{g, 3});
    [~, i] = min(abs(pool.volume_cm3 - target));  % min returns the first = lowest ID
    sel = [sel; table(g, string(groups{g, 1}), pool.patient_id(i), pool.grade(i), ...
        pool.volume_cm3(i), target, height(pool), 'VariableNames', ...
        {'case', 'role', 'patient_id', 'grade', 'wt_volume_cm3', 'target_cm3', 'group_size'})]; %#ok<AGROW>
end
assert(numel(unique(sel.patient_id)) == 5, 'selection picked a patient twice');
writetable(sel, SEL_CSV);
disp(sel);

%% ---- 2. One figure per selected patient ----
set(groot, 'defaultAxesFontSize', 11, 'defaultTextFontSize', 11);
for c = 1:height(sel)
    id = char(sel.patient_id(c));
    lab = niftiread(fullfile(DATA_ROOT, id, [id '_seg.nii']));
    fl  = single(niftiread(fullfile(DATA_ROOT, id, [id '_flair.nii'])));
    lab = lab(ROWS, COLS, :);  fl = fl(ROWS, COLS, :);
    brain3 = fl > 0;
    % mu_brain: mean clean FLAIR over all brain voxels of the cropped volume,
    % exactly as python/data/preprocess.py computes it for noise calibration.
    mu = mean(fl(brain3));
    [~, z] = max(squeeze(sum(sum(lab > 0, 1), 2)));
    x = fl(:, :, z);  bm = brain3(:, :, z);  L = lab(:, :, z);

    rs = RandStream('mt19937ar', 'Seed', SEED);
    nr = randn(rs, size(x), 'single');  ni = randn(rs, size(x), 'single');
    xd = degrade_slice(x, mu, SNR, R, nr, ni);

    zs = @(im) (im - mean(im(bm))) / std(im(bm));    % brain z-score
    xc = zs(x);  xdz = zs(xd);
    [lo, hi]  = decompose_slice(xdz, D0);
    [~, hic]  = decompose_slice(xc, D0);
    % Measured SNR in the slice: brain mean / noise SD from the background
    % (Rayleigh-distributed magnitude: sigma = mean(background) / sqrt(pi/2)).
    bgd = xd(~imdilate(bm, strel('disk', 5)));
    snr_meas = mean(xd(bm)) / (mean(bgd) / sqrt(pi / 2));

    pred = [];
    pf = sprintf(PRED_FMT, PRED_MODEL, id);
    if USE_PREDICTIONS && isfile(pf)
        pred = niftiread(pf);  pred = pred(ROWS, COLS, z);
    end

    f = figure('Visible', 'off', 'Position', [50 50 1500 820]);  theme(f, 'light');
    tl = tiledlayout(f, 2, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
    show = @(im, lims, ttl) imshow_(nexttile, rot90(im), lims, ttl);
    show(xc,  [-2 4], '1  FLAIR, clean');
    show(xdz, [-2 4], sprintf('2  FLAIR, SNR %d, r %.1f', SNR, R));
    ax = show(xc, [-2 4], '3  Ground truth');  outline_(ax, L, COL);
    ax = show(xdz, [-2 4], sprintf('4  Prediction (%s, SNR %d, r %.1f)', PRED_MODEL, SNR, R));
    if isempty(pred)
        text(ax, 96, 96, {'awaiting predictions', '(Stage 3)'}, 'Color', 'w', ...
            'HorizontalAlignment', 'center', 'FontSize', 13, 'FontWeight', 'bold');
    else
        outline_(ax, pred, COL);
    end
    show(lo,  [-2 4],     sprintf('5a  Low band of input 2 (D_0 = %.2f)', D0));
    show(hi,  [-1.5 1.5], '5b  High band of input 2');
    show(hic, [-1.5 1.5], '5c  High band of clean input 1');

    % Text card
    ax = nexttile;  axis(ax, 'off');
    v = gt(gt.patient_id == id, :);
    card = {
        sprintf('\\bf%s\\rm', strrep(id, '_', '\_'))
        sprintf('Case %d: %s', c, sel.role(c))
        sprintf('Grade %s, axial slice %d / 155', sel.grade(c), z)
        ''
        '\bfGround-truth biomarkers\rm'
        bio_(v, 1, 'WT')
        bio_(v, 2, 'TC')
        bio_(v, 3, 'ET')
        ''
        '\bfDegradation of panel 2\rm'
        sprintf('noise \\sigma = \\mu_{brain} / %d', SNR)
        sprintf('measured slice SNR %.1f', snr_meas)
        sprintf('k-space kept: central %d%% per axis', round(100 * R))
    };
    text(ax, 0, 1, card, 'VerticalAlignment', 'top', 'FontSize', 11, 'Interpreter', 'tex');
    h = gobjects(3, 1);  hold(ax, 'on');
    for k = 1:3, h(k) = plot(ax, NaN, NaN, '-', 'Color', COL(k, :), 'LineWidth', 2); end
    legend(ax, h, {'WT', 'TC', 'ET'}, 'Location', 'southwest', 'Orientation', 'horizontal');
    out = sprintf(OUT_FMT, c, id);
    exportgraphics(f, out, 'Resolution', 150);  close(f);
    fprintf('Saved %s (slice %d, measured SNR %.1f)\n', out, z, snr_meas);
end

function s = bio_(v, k, name)
if v.voxels(k) == 0
    s = sprintf('%s  absent', name);
else
    s = sprintf('%s %6.1f cm^3   sphericity %.2f', name, v.volume_cm3(k), v.sphericity(k));
end
end

function ax = imshow_(ax, im, lims, ttl)
imshow(im, lims, 'Parent', ax);
title(ax, ttl, 'FontWeight', 'normal');
end

function outline_(ax, L, col)
% WT / TC / ET outlines; label 3 or 4 = ET (raw and Python conventions).
hold(ax, 'on');
m = {L > 0, L == 1 | L == 3 | L == 4, L == 3 | L == 4};
for k = 1:3
    if any(m{k}(:))
        contour(ax, rot90(double(m{k})), [0.5 0.5], 'Color', col(k, :), 'LineWidth', 1.6);
    end
end
end
