% AGREEMENT_VALIDATION  Roadmap step 1.2: validate the agreement analysis on
% synthetically damaged ground-truth masks.
%
% Run from repo root:  matlab -batch "run('matlab/agreement_validation.m')"
% Runtime ~5-8 min (74 patients x 7 damage types, full biomarkers each).
%
% Logic: damage every test patient's true outline in a KNOWN way, measure the
% damaged outline with the same tumor_biomarkers.m used for predictions, then
% run the same agreement statistics (agreement_stats.m) that Stage 3 will run
% on real predictions. Because the damage is known, the right answer is known:
%   - blobs / delete:  volume change is known to the voxel  -> must match exactly
%   - shift:           volume unchanged, overlap falls      -> bias 0, Dice < 1
%   - dilate / erode:  volume change ~ surface area x offset (Steiner's formula,
%                      calibrated on a sphere)               -> must match closely
% If the agreement code reports these, it can be trusted on real predictions.
%
% Outputs
%   results/agreement_damaged_biomarkers.csv   per patient x damage x region
%   results/agreement_validation.csv           summary per damage x region
%   figures/agreement_damage_examples.png      what each damage looks like
%   figures/agreement_bland_altman_wt.png      Bland-Altman per damage (WT)
%   figures/agreement_boundary_error.png       surface-area prediction + size effect

%% ---- Configuration ----
cd(fileparts(fileparts(mfilename('fullpath'))));
addpath('matlab');
DATA_ROOT  = 'archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData';
SPLIT_FILE = 'data/splits/split_v1.json';
GT_CSV     = 'results/volumetrics_gt.csv';
OUT_ROWS   = 'results/agreement_damaged_biomarkers.csv';
OUT_SUM    = 'results/agreement_validation.csv';
SEED       = 42;
RECOMPUTE  = false;   % false: reuse OUT_ROWS if it exists (figures-only rerun)
% {name, type, param}
DAMAGE = {
    'dilate 1 mm',   'dilate', 1
    'dilate 2 mm',   'dilate', 2
    'erode 1 mm',    'erode',  1
    'erode 2 mm',    'erode',  2
    'shift 2 vox',   'shift',  2
    'delete r=8 mm', 'delete', 8
    '3 blobs r=3',   'blobs',  [3 3]
};
REG = ["WT", "TC", "ET"];
set(groot, 'defaultAxesFontSize', 11, 'defaultTextFontSize', 11, ...
    'defaultAxesTickDir', 'out', 'defaultAxesBox', 'off');
COL = [0.00 0.45 0.70; 0.90 0.60 0.00; 0.80 0.20 0.20];

ids = jsondecode(fileread(SPLIT_FILE)).test;
gt  = readtable(GT_CSV, 'TextType', 'string');

%% ---- Calibration: volume change per mm^2 of surface, on a big sphere ----
% Steiner's formula: growing a smooth body outward by d adds ~ A*d volume.
% On a voxel grid the effective offset differs from d, so measure it on a
% digital sphere (R = 30) and use that factor to PREDICT each tumor's change.
[x, y, z] = ndgrid(-40:40);
sph = uint8(2 * (x.^2 + y.^2 + z.^2 <= 30^2));
A0 = tumor_biomarkers(sph, [1 1 1]).surface_area_mm2(1);
kcal = nan(size(DAMAGE, 1), 1);
for s = 1:4                                   % dilate/erode only
    dmg = damage_mask(sph, DAMAGE{s, 2}, DAMAGE{s, 3});
    kcal(s) = (nnz(dmg) - nnz(sph)) / A0;     % mm^3 per mm^2 = effective offset (mm)
end
fprintf('Effective offset per mm^2 of surface: %s mm\n', mat2str(kcal(1:4)', 3));

%% ---- Damage every patient ----
if RECOMPUTE || ~isfile(OUT_ROWS)
rows = cell(numel(ids), 1);
tic;
for p = 1:numel(ids)
    info = niftiinfo(fullfile(DATA_ROOT, ids{p}, [ids{p} '_seg.nii']));
    lab  = niftiread(info);
    brain = niftiread(fullfile(DATA_ROOT, ids{p}, [ids{p} '_flair.nii'])) > 0;
    sp = info.PixelDimensions(1:3);
    ref = {lab > 0, lab == 1 | lab == 4, lab == 4};
    rs = RandStream('mt19937ar', 'Seed', SEED + p);     % per-patient, reproducible
    pr = cell(size(DAMAGE, 1), 1);
    for s = 1:size(DAMAGE, 1)
        [dmg, dinfo] = damage_mask(lab, DAMAGE{s, 2}, DAMAGE{s, 3}, rs, brain);
        T = tumor_biomarkers(dmg, sp);
        hyp = {dmg > 0, dmg == 1 | dmg == 4, dmg == 4};
        dice = zeros(3, 1);
        for k = 1:3
            den = nnz(ref{k}) + nnz(hyp{k});
            dice(k) = 2 * nnz(ref{k} & hyp{k}) / den;     % 0/0 -> NaN when both empty
        end
        T.dice = dice;
        T.expected_dvox = (dinfo.added - dinfo.removed)';
        T.new_pieces = repmat(dinfo.new_pieces, 3, 1);
        T = [table(repmat(string(DAMAGE{s, 1}), 3, 1), repmat(string(ids{p}), 3, 1), ...
                   'VariableNames', {'damage', 'patient_id'}), T]; %#ok<AGROW>
        pr{s} = T;
    end
    rows{p} = vertcat(pr{:});
    if mod(p, 10) == 0 || p == numel(ids)
        fprintf('  %d/%d patients (%.0f s)\n', p, numel(ids), toc);
    end
end
D = vertcat(rows{:});
D.region = string(D.region);
writetable(D, OUT_ROWS);
fprintf('Saved %s (%d rows)\n', OUT_ROWS, height(D));
else
    D = readtable(OUT_ROWS, 'TextType', 'string');
    fprintf('Reusing %s (set RECOMPUTE = true to redo)\n', OUT_ROWS);
end

%% ---- Summary: agreement statistics per damage x region ----
S = table();
for s = 1:size(DAMAGE, 1)
    for k = 1:3
        g = gt(gt.region == REG(k), :);
        d = D(D.damage == DAMAGE{s, 1} & D.region == REG(k), :);
        [~, ia] = ismember(d.patient_id, g.patient_id);
        g = g(ia, :);                                 % align patients
        A = agreement_stats(g.volume_cm3, d.volume_cm3);
        B = agreement_stats(g.sphericity, d.sphericity, 200);
        dv = d.voxels - g.voxels;                     % measured change, voxels
        % Expected change: exact for delete/blobs, Steiner prediction otherwise.
        if s <= 4
            expv = kcal(s) * g.surface_area_mm2;
        elseif strcmp(DAMAGE{s, 2}, 'shift')
            expv = zeros(size(dv));
        else
            expv = d.expected_dvox;
        end
        has = g.voxels > 0;
        pc = d.n_components - g.n_components;
        r = table(string(DAMAGE{s, 1}), REG(k), A.n, ...
            A.bias, A.bias_ci(1), A.bias_ci(2), A.loa_low, A.loa_high, ...
            A.pct_err_median, A.pct_abs_err_median, ...
            A.icc_a1, A.icc_a1_ci(1), A.icc_a1_ci(2), A.icc_c1, ...
            mean(d.dice(has), 'omitnan'), ...
            median(dv(has) ./ expv(has), 'omitnan'), ...
            max(abs(dv - expv)), ...
            median(pc), B.bias, ...
            'VariableNames', {'damage', 'region', 'n', ...
            'bias_cm3', 'bias_ci_lo', 'bias_ci_hi', 'loa_low_cm3', 'loa_high_cm3', ...
            'pct_err_median', 'pct_abs_err_median', ...
            'icc_a1', 'icc_a1_ci_lo', 'icc_a1_ci_hi', 'icc_c1', ...
            'mean_dice', 'measured_over_expected_median', 'max_abs_dev_from_expected_vox', ...
            'median_extra_pieces', 'sphericity_bias'});
        S = [S; r]; %#ok<AGROW>
    end
end
writetable(S, OUT_SUM);
fprintf('Saved %s\n', OUT_SUM);
disp(S(:, {'damage', 'region', 'bias_cm3', 'pct_err_median', 'icc_a1', 'icc_c1', ...
    'mean_dice', 'measured_over_expected_median', 'max_abs_dev_from_expected_vox', ...
    'median_extra_pieces', 'sphericity_bias'}));

%% ---- Figure 1: what each damage looks like (one patient, one slice) ----
% First test patient whose core is clearly bigger than its enhancing tumor
% (so all three outlines are visible separately) and ET >= 5 cm^3.
for p = 1:numel(ids)
    v = gt.volume_cm3(gt.patient_id == ids{p});    % WT, TC, ET
    if v(3) >= 5 && v(2) > 1.5 * v(3), break; end
end
lab = niftiread(fullfile(DATA_ROOT, ids{p}, [ids{p} '_seg.nii']));
fl  = double(niftiread(fullfile(DATA_ROOT, ids{p}, [ids{p} '_flair.nii'])));
brain = fl > 0;
[~, zz] = max(squeeze(sum(sum(lab > 0, 1), 2)));
rs = RandStream('mt19937ar', 'Seed', SEED + p);
f = figure('Visible', 'off', 'Position', [50 50 1000 700]);  theme(f, 'light');
tl = tiledlayout(f, 2, 4, 'TileSpacing', 'tight', 'Padding', 'compact');
[ri, ci] = find(any(lab > 0, 3));                  % zoom box around tumor
box = {max(min(ri) - 15, 1):min(max(ri) + 15, 240), max(min(ci) - 15, 1):min(max(ci) + 15, 240)};
for s = 0:size(DAMAGE, 1)
    nexttile;
    if s == 0
        dmg = lab;  ttl = 'original (truth)';
    else
        [dmg, ~] = damage_mask(lab, DAMAGE{s, 2}, DAMAGE{s, 3}, rs, brain);
        ttl = DAMAGE{s, 1};
        if strcmp(DAMAGE{s, 2}, 'blobs')           % show the slice through a blob
            extra = (dmg > 0) & ~(lab > 0);
            [~, zb] = max(squeeze(sum(sum(extra, 1), 2)));
            ttl = sprintf('%s (slice %d)', ttl, zb);
        end
    end
    zs = zz;  if s > 0 && strcmp(DAMAGE{s, 2}, 'blobs'), zs = zb; end
    b = fl(:, :, zs);  bm = brain(:, :, zs);
    b = b / prctile_(b(bm), 99.5);                 % normalise on this slice's brain
    if strcmp(ttl, 'original (truth)') || ~contains(ttl, 'blobs')
        b = b(box{1}, box{2});  L = dmg(box{1}, box{2}, zs);  T0 = lab(box{1}, box{2}, zs);
    else
        L = dmg(:, :, zs);  T0 = lab(:, :, zs);
    end
    imshow(rot90(min(b, 1)), []);  hold on;
    for k = 1:3
        m = {T0 > 0, T0 == 1 | T0 == 4, T0 == 4};
        h = {L > 0, L == 1 | L == 4, L == 4};
        if s > 0 && any(m{k}(:))
            contour(rot90(double(m{k})), [0.5 0.5], ':', 'Color', [1 1 1], 'LineWidth', 1);
        end
        if any(h{k}(:))
            contour(rot90(double(h{k})), [0.5 0.5], '-', 'Color', COL(k, :), 'LineWidth', 1.6);
        end
    end
    title(ttl, 'FontWeight', 'normal');
end
title(tl, {sprintf('Synthetic damage, %s (FLAIR, axial)', strrep(ids{p}, '_', '\_')), ...
    'colour = damaged (blue WT, orange TC, red ET), white dotted = original'}, 'FontSize', 11);
exportgraphics(f, 'figures/agreement_damage_examples.png', 'Resolution', 200);  close(f);

%% ---- Figure 2: Bland-Altman per damage type, WT volume ----
f = figure('Visible', 'off', 'Position', [50 50 1600 760]);  theme(f, 'light');
tl = tiledlayout(f, 2, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
g = gt(gt.region == "WT", :);
for s = 1:size(DAMAGE, 1)
    nexttile;  hold on;
    d = D(D.damage == DAMAGE{s, 1} & D.region == "WT", :);
    [~, ia] = ismember(d.patient_id, g.patient_id);
    ref = g.volume_cm3(ia);  tst = d.volume_cm3;
    A = agreement_stats(ref, tst, 200);
    scatter((ref + tst) / 2, tst - ref, 18, COL(1, :), 'filled', 'MarkerFaceAlpha', 0.7);
    yline(A.bias, '-', sprintf('bias %+.2f', A.bias), 'Color', [0.2 0.2 0.2], ...
        'LabelHorizontalAlignment', 'left', 'FontSize', 9);
    yline(A.loa_low, '--', 'Color', [0.5 0.5 0.5]);
    yline(A.loa_high, '--', 'Color', [0.5 0.5 0.5]);
    yline(0, ':', 'Color', [0.7 0.7 0.7]);
    title(sprintf('%s  (ICC_A = %.3f)', DAMAGE{s, 1}, A.icc_a1), 'FontWeight', 'normal');
    xlabel('Mean of truth and damaged (cm^3)');
    ylabel('Damaged - truth (cm^3)');
end
% Last tile: ICC absolute vs consistency for every damage, WT.
nexttile;
w = S(S.region == "WT", :);
bar(categorical(w.damage, w.damage), [w.icc_a1 w.icc_c1]);
ylim([min([w.icc_a1; w.icc_c1]) - 0.02, 1.005]);
legend({'ICC(A,1) absolute', 'ICC(C,1) consistency'}, 'Location', 'southwest', 'FontSize', 9);
ylabel('ICC');  title('Agreement index per damage', 'FontWeight', 'normal');
title(tl, ['Bland-Altman, whole-tumor volume, 74 test patients. ' ...
    'Solid = bias, dashed = 95% limits of agreement']);
exportgraphics(f, 'figures/agreement_bland_altman_wt.png', 'Resolution', 200);  close(f);

%% ---- Figure 3: boundary error -> volume error, and why small/irregular regions suffer ----
f = figure('Visible', 'off', 'Position', [50 50 1300 480]);  theme(f, 'light');
tl = tiledlayout(f, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
nexttile;  hold on;
for k = 1:3
    g = gt(gt.region == REG(k) & gt.voxels > 0, :);
    d = D(D.damage == "dilate 1 mm" & D.region == REG(k), :);
    [~, ia] = ismember(g.patient_id, d.patient_id);
    meas = (d.voxels(ia) - g.voxels) / 1000;
    pred = kcal(1) * g.surface_area_mm2 / 1000;
    scatter(pred, meas, 20, COL(k, :), 'filled', 'MarkerFaceAlpha', 0.7, 'DisplayName', REG(k));
end
lim = xlim;  plot(lim, lim, 'k:', 'DisplayName', 'identity');
xlabel('Predicted added volume = k \times surface area (cm^3)');
ylabel('Measured added volume (cm^3)');
legend('Location', 'northwest');
title('(a) 1 mm dilation: volume added follows surface area', 'FontWeight', 'normal');
nexttile;  hold on;
for k = 1:3
    g = gt(gt.region == REG(k) & gt.voxels > 0, :);
    d = D(D.damage == "dilate 1 mm" & D.region == REG(k), :);
    [~, ia] = ismember(g.patient_id, d.patient_id);
    pe = 100 * (d.voxels(ia) - g.voxels) ./ g.voxels;
    scatter(g.volume_cm3, pe, 20, COL(k, :), 'filled', 'MarkerFaceAlpha', 0.7, 'DisplayName', REG(k));
end
set(gca, 'XScale', 'log', 'YScale', 'log');  grid on;
xlabel('True region volume (cm^3, log)');
ylabel('Volume error from a 1 mm boundary error (%, log)');
legend('Location', 'northeast');
title('(b) The same 1 mm error is a far larger % for small/irregular regions', ...
    'FontWeight', 'normal');
exportgraphics(f, 'figures/agreement_boundary_error.png', 'Resolution', 200);  close(f);
fprintf('Figures saved.\n');

function q = prctile_(x, p)
x = sort(x(:));  q = interp1(linspace(0, 100, numel(x)), x, p);
end
