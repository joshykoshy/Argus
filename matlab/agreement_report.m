% AGREEMENT_REPORT  Stage 3: predicted vs ground-truth biomarkers, per model,
% condition and region.
%
% Run from repo root, after volumetrics.m has produced BOTH
%   results/volumetrics_gt.csv   (SOURCE = 'gt')
%   results/volumetrics_pred.csv (SOURCE = 'pred', all models complete)
%     matlab -batch "run('matlab/agreement_report.m')"
%
% Uses agreement_stats.m, validated on synthetic damage in Step 1.2.
% PRED_CSV, OUT_DIR, FIG_DIR, TAG and COMPARE can be pre-set in the workspace
% (test_stage3_pipeline.m does this on fake predictions).
%
% Outputs
%   <OUT_DIR>/agreement_<TAG>.csv            one row per model x condition x region
%   <FIG_DIR>/agreement_<TAG>_volume_error.png  median volume error vs condition
%   <FIG_DIR>/agreement_<TAG>_bland_altman.png  COMPARE models, clean vs worst, WT and ET

cd(fileparts(fileparts(mfilename('fullpath'))));
addpath('matlab');
GT_CSV = 'results/volumetrics_gt.csv';
if ~exist('PRED_CSV', 'var'), PRED_CSV = 'results/volumetrics_pred.csv'; end
if ~exist('OUT_DIR', 'var'),  OUT_DIR  = 'results'; end
if ~exist('FIG_DIR', 'var'),  FIG_DIR  = 'figures'; end
if ~exist('TAG', 'var'),      TAG      = 'stage3'; end
if ~exist('COMPARE', 'var'),  COMPARE  = {'M0', 'M4'}; end   % baseline vs proposed
CONDS = ["clean", "snr12_r0.75", "snr8_r0.5"];               % mild -> worst
REG   = ["WT", "TC", "ET"];

gt = readtable(GT_CSV, 'TextType', 'string');
pr = readtable(PRED_CSV, 'TextType', 'string');
models = unique(pr.source, 'stable');

%% ---- Agreement table ----
S = table();
for m = models'
    for c = CONDS
        for k = REG
            g = gt(gt.region == k, :);
            p = pr(pr.source == m & pr.condition == c & pr.region == k, :);
            if isempty(p), continue; end
            [ok, ia] = ismember(p.patient_id, g.patient_id);
            assert(all(ok), 'prediction patients missing from ground truth');
            g = g(ia, :);
            A = agreement_stats(g.volume_cm3, p.volume_cm3);
            B = agreement_stats(g.sphericity, p.sphericity, 200);
            both = g.voxels > 0 & p.voxels > 0;
            S = [S; table(m, c, k, A.n, ...
                A.bias, A.bias_ci(1), A.bias_ci(2), A.loa_low, A.loa_high, ...
                A.pct_err_median, A.pct_abs_err_median, ...
                A.icc_a1, A.icc_a1_ci(1), A.icc_a1_ci(2), A.icc_c1, ...
                B.bias, median(p.n_components(both) - g.n_components(both)), ...
                median(p.frac_outside_largest(both) - g.frac_outside_largest(both)), ...
                sum(g.voxels == 0 & p.voxels > 0), sum(g.voxels > 0 & p.voxels == 0), ...
                'VariableNames', {'model', 'condition', 'region', 'n', ...
                'vol_bias_cm3', 'vol_bias_ci_lo', 'vol_bias_ci_hi', 'loa_low_cm3', 'loa_high_cm3', ...
                'vol_pct_err_median', 'vol_pct_abs_err_median', ...
                'icc_a1', 'icc_a1_ci_lo', 'icc_a1_ci_hi', 'icc_c1', ...
                'sphericity_bias', 'extra_pieces_median', 'extra_frac_outside_largest_median', ...
                'n_false_region', 'n_missed_region'})]; %#ok<AGROW>
        end
    end
end
out = fullfile(OUT_DIR, sprintf('agreement_%s.csv', TAG));
writetable(S, out);
fprintf('Saved %s (%d rows)\n', out, height(S));
disp(S(:, {'model', 'condition', 'region', 'vol_bias_cm3', 'vol_pct_abs_err_median', ...
    'icc_a1', 'extra_pieces_median', 'n_false_region', 'n_missed_region'}));

%% ---- Figure 1: how wrong does the volume get as the scan gets worse? ----
set(groot, 'defaultAxesFontSize', 11, 'defaultAxesTickDir', 'out', 'defaultAxesBox', 'off');
f = figure('Visible', 'off', 'Position', [50 50 1500 440]);  theme(f, 'light');
tl = tiledlayout(f, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
cm = lines(numel(models));
for k = 1:3
    ax = nexttile;  hold(ax, 'on');
    for i = 1:numel(models)
        med = nan(1, 3);  lo = med;  hi = med;
        for c = 1:3
            g = gt(gt.region == REG(k) & gt.voxels > 0, :);
            p = pr(pr.source == models(i) & pr.condition == CONDS(c) & pr.region == REG(k), :);
            [ok, ia] = ismember(g.patient_id, p.patient_id);
            if ~any(ok), continue; end
            e = 100 * (p.volume_cm3(ia(ok)) - g.volume_cm3(ok)) ./ g.volume_cm3(ok);
            q = interp1(linspace(0, 100, numel(e)), sort(e), [25 50 75]);
            lo(c) = q(1);  med(c) = q(2);  hi(c) = q(3);
        end
        x = (1:3) + 0.06 * (i - (numel(models) + 1) / 2);
        errorbar(ax, x, med, med - lo, hi - med, '-o', 'Color', cm(i, :), ...
            'MarkerFaceColor', cm(i, :), 'LineWidth', 1.5, 'DisplayName', models(i));
    end
    yline(ax, 0, 'k:', 'HandleVisibility', 'off');
    xticks(ax, 1:3);  xticklabels(ax, {'clean', 'SNR 12, r 0.75', 'SNR 8, r 0.5'});
    xlim(ax, [0.6 3.4]);
    ylabel(ax, 'Volume error (%), median and IQR');
    title(ax, REG(k), 'FontWeight', 'normal');
    if k == 3, legend(ax, 'Location', 'eastoutside'); end
end
title(tl, 'Predicted vs true tumor volume as scan quality drops (patients with the region present)');
exportgraphics(f, fullfile(FIG_DIR, sprintf('agreement_%s_volume_error.png', TAG)), 'Resolution', 200);
close(f);

%% ---- Figure 2: Bland-Altman, COMPARE models, clean vs worst, WT and ET ----
f = figure('Visible', 'off', 'Position', [50 50 1600 760]);  theme(f, 'light');
tl = tiledlayout(f, 2, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
for k = ["WT", "ET"]
    for m = string(COMPARE)
        for c = CONDS([1 3])
            ax = nexttile;  hold(ax, 'on');
            g = gt(gt.region == k, :);
            p = pr(pr.source == m & pr.condition == c & pr.region == k, :);
            if isempty(p), axis(ax, 'off'); continue; end
            [~, ia] = ismember(p.patient_id, g.patient_id);
            ref = g.volume_cm3(ia);  tst = p.volume_cm3;
            A = agreement_stats(ref, tst, 200);
            scatter(ax, (ref + tst) / 2, tst - ref, 16, 'filled', 'MarkerFaceAlpha', 0.7);
            yline(ax, A.bias, '-', sprintf('bias %+.2f', A.bias), 'FontSize', 9, ...
                'LabelHorizontalAlignment', 'left');
            yline(ax, [A.loa_low A.loa_high], '--', 'Color', [0.5 0.5 0.5]);
            yline(ax, 0, ':', 'Color', [0.7 0.7 0.7]);
            title(ax, sprintf('%s %s, %s (ICC_A %.3f)', m, k, strrep(c, '_', ' '), A.icc_a1), ...
                'FontWeight', 'normal');
            xlabel(ax, 'Mean of truth and prediction (cm^3)');
            ylabel(ax, 'Prediction - truth (cm^3)');
        end
    end
end
title(tl, 'Bland-Altman: predicted vs true volume. Solid = bias, dashed = 95% limits of agreement');
exportgraphics(f, fullfile(FIG_DIR, sprintf('agreement_%s_bland_altman.png', TAG)), 'Resolution', 200);
close(f);
fprintf('Figures saved to %s\n', FIG_DIR);
