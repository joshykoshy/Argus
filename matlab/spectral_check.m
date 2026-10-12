% SPECTRAL_CHECK  Measure the spectral claims used in the paper, from raw data.
%
% Run from repo root:  matlab -batch "run('matlab/spectral_check.m')"
%
% Why: docs/FINDINGS_SPECTRAL.md states "over 92% of signal power in D < 0.20"
% and "high band drops below 0 dB SNR at SNR 8-12, r <= 0.75". Those sentences
% are hard-coded fprintf text in spectral_analysis.m, not computed values, and
% that script adds unseeded noise. This script computes the numbers.
%
% Same sample as python/analysis/export_for_matlab_spectral.py: the first 12
% test patients, axial slice with the most tumor, 192 x 192 crop, raw
% intensities, mu_brain over the cropped 3D brain mask (any modality > 0).
% Noise: fixed seed 42. Degradation: the DEC-005 grid (SNR 8/5/3 x r 1.0/0.5 x
% slice 1/5 mm, noise in the acquired k-space band) via degrade_volume.m on a
% 5-slice slab centred on the chosen slice (enough for the 5 mm average).
%
% Definitions (D = normalised radial frequency, 1 = Nyquist along an axis):
%   low band = D < D0, high band = D >= D0, D0 = 0.20 (the hard split used for
%   reporting; the model itself uses the smooth Gaussian split, so we also
%   report the Gaussian low-pass energy share).
%   Band SNR = sum over band of |F(clean)|^2 / sum of |F(degraded - clean)|^2,
%   in dB. The "error" image is everything the degradation changed: noise plus
%   lost in-plane and through-plane resolution. 0 dB = error as strong as the anatomy in that band.
%
% Outputs: results/spectral_check.csv, figures/spectral_band_snr.png

cd(fileparts(fileparts(mfilename('fullpath'))));
addpath('matlab/reference');
DATA_ROOT = 'archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData';
ids = jsondecode(fileread('data/splits/split_v1.json')).test;
ids = ids(1:12);
ROWS = 25:216;  COLS = 29:220;  D0 = 0.20;  SEED = 42;
MODS = {'flair', 't1ce'};
SNRS = [8 5 3];  RS = [1 0.5];  THICKS = [1 5];

H = 192;  W = 192;  ch = H / 2;  cw = W / 2;
[V, U] = meshgrid((0:W-1) - cw, (0:H-1) - ch);
D = sqrt((U / ch).^2 + (V / cw).^2);
low = D < D0;  dc = D == 0;
Hlp = exp(-D.^2 / (2 * D0^2));

rs = RandStream('mt19937ar', 'Seed', SEED);
rows = {};
for p = 1:numel(ids)
    id = ids{p};
    seg = niftiread(fullfile(DATA_ROOT, id, [id '_seg.nii']));
    seg = seg(ROWS, COLS, :);
    [~, z] = max(squeeze(sum(sum(seg > 0, 1), 2)));
    vol = cell(1, 4);  names = {'t1', 't1ce', 't2', 'flair'};
    for m = 1:4
        v = single(niftiread(fullfile(DATA_ROOT, id, [id '_' names{m} '.nii'])));
        vol{m} = v(ROWS, COLS, :);
    end
    brain = (abs(vol{1}) + abs(vol{2}) + abs(vol{3}) + abs(vol{4})) > 1e-4;
    for m = 1:numel(MODS)
        v = vol{strcmp(names, MODS{m})};
        mu = mean(v(brain));
        x = v(:, :, z);
        zs = max(1, z - 2):min(size(v, 3), z + 2);  zc = find(zs == z);
        slab = v(:, :, zs);
        P = abs(fftshift(fft2(x))).^2;
        frac_low_all  = sum(P(low)) / sum(P(:));
        frac_low_nodc = sum(P(low & ~dc)) / sum(P(~dc));
        frac_gauss    = sum(P(:) .* Hlp(:).^2) / sum(P(:) .* (Hlp(:).^2 + (1 - Hlp(:)).^2));
        nr = randn(rs, size(slab), 'single');  ni = randn(rs, size(slab), 'single');
        for s = SNRS
            for r = RS
                for t = THICKS
                    xd = degrade_volume(slab, mu, struct('snr', s, 'r', r, 'thick', t), nr, ni);
                    E = abs(fftshift(fft2(xd(:, :, zc) - x))).^2;
                    snr_low  = 10 * log10(sum(P(low))  / sum(E(low)));
                    snr_high = 10 * log10(sum(P(~low)) / sum(E(~low)));
                    rows(end+1, :) = {id, MODS{m}, s, r, t, frac_low_all, frac_low_nodc, ...
                        frac_gauss, snr_low, snr_high}; %#ok<SAGROW>
                end
            end
        end
    end
end
T = cell2table(rows, 'VariableNames', {'patient_id', 'modality', 'snr', 'r', 'thick', ...
    'clean_power_frac_low', 'clean_power_frac_low_no_dc', 'clean_energy_frac_gauss_low', ...
    'band_snr_low_db', 'band_snr_high_db'});
writetable(T, 'results/spectral_check.csv');

%% Summary
for m = 1:numel(MODS)
    t = T(strcmp(T.modality, MODS{m}) & T.snr == SNRS(1) & T.r == 1 & T.thick == 1, :);
    fprintf('%s clean power in D<%.2f: %.1f%% incl. DC (min %.1f%%), %.1f%% excl. DC (min %.1f%%); Gaussian low band energy share %.1f%%\n', ...
        upper(MODS{m}), D0, 100 * mean(t.clean_power_frac_low), 100 * min(t.clean_power_frac_low), ...
        100 * mean(t.clean_power_frac_low_no_dc), 100 * min(t.clean_power_frac_low_no_dc), ...
        100 * mean(t.clean_energy_frac_gauss_low));
end
fprintf('\nMean band SNR (dB) over 12 patients: low band / high band\n');
fprintf('%-10s', 'SNR');  fprintf('%18s', 'r1.0 1mm', 'r1.0 5mm', 'r0.5 1mm', 'r0.5 5mm');  fprintf('\n');
for m = 1:numel(MODS)
    fprintf('%s\n', upper(MODS{m}));
    for s = SNRS
        fprintf('%-10d', s);
        for r = RS
            for th = THICKS
                t = T(strcmp(T.modality, MODS{m}) & T.snr == s & T.r == r & T.thick == th, :);
                fprintf('%8.1f / %6.1f  ', mean(t.band_snr_low_db), mean(t.band_snr_high_db));
            end
        end
        fprintf('\n');
    end
end

%% Figure: band SNR for every degraded condition, FLAIR and T1ce
set(groot, 'defaultAxesFontSize', 11);
f = figure('Visible', 'off', 'Position', [50 50 1200 440]);  theme(f, 'light');
tl = tiledlayout(f, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
mk = {'o', 's', '^', 'd'};  cl = [0.00 0.45 0.70; 0.35 0.70 0.90; 0.80 0.20 0.20; 0.90 0.60 0.00];
combos = [1 1; 1 5; 0.5 1; 0.5 5];             % [r thick]
for m = 1:numel(MODS)
    ax = nexttile;  hold on;
    for ci = 1:size(combos, 1)
        lo = zeros(size(SNRS));  hi = lo;
        for si = 1:numel(SNRS)
            t = T(strcmp(T.modality, MODS{m}) & T.snr == SNRS(si) & T.r == combos(ci, 1) & ...
                T.thick == combos(ci, 2), :);
            lo(si) = mean(t.band_snr_low_db);  hi(si) = mean(t.band_snr_high_db);
        end
        lbl = sprintf('r = %.2g, %d mm', combos(ci, 1), combos(ci, 2));
        plot(SNRS, lo, ['-' mk{ci}], 'Color', cl(ci, :), 'LineWidth', 1.6, ...
            'MarkerFaceColor', cl(ci, :), 'DisplayName', ['low band, ' lbl]);
        plot(SNRS, hi, ['--' mk{ci}], 'Color', cl(ci, :), 'LineWidth', 1.6, ...
            'DisplayName', ['high band, ' lbl]);
    end
    yline(0, 'k:', 'error = anatomy', 'LabelHorizontalAlignment', 'left', 'HandleVisibility', 'off');
    set(ax, 'XDir', 'reverse');  xticks(fliplr(SNRS));  xlim([2.5 8.5]);  ylim('auto');
    xlabel('Nominal SNR (linear, \sigma = \mu_{brain} / SNR)');
    ylabel('Band SNR (dB): anatomy / degradation error');
    title(sprintf('%s, 12 test slices, D_0 = %.2f', upper(MODS{m}), D0), 'FontWeight', 'normal');
    if m == 2, legend('Location', 'eastoutside', 'FontSize', 9); end
    grid on;
end
exportgraphics(f, 'figures/spectral_band_snr.png', 'Resolution', 200);  close(f);
fprintf('Saved results/spectral_check.csv and figures/spectral_band_snr.png\n');
