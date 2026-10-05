% FIGURE_GALLERY  Picture-book overview of the 5 selected test patients.
%
% Run from repo root:  matlab -batch "run('matlab/figure_gallery.m')"
% Needs results/diagnostic_selection.csv (matlab/diagnostic_report.m).
%
% Columns = the 5 patients chosen blind in Step 1.4. Rows:
%   1  FLAIR scan as a normal hospital scanner gives it
%   2  the same slice as a cheap low-field scanner might (SNR 8, r 0.5)
%   3  the expert's tumor outline (blue = whole tumor incl. swelling,
%      orange = tumor core, red = actively growing part)
%   4  zoom on the tumor, clean vs low-field, side by side
% Same crop, degradation and seed as diagnostic_report.m.

cd(fileparts(fileparts(mfilename('fullpath'))));
addpath('matlab/reference');
DATA_ROOT = 'archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData';
ROWS = 25:216;  COLS = 29:220;  SNR = 8;  R = 0.5;  SEED = 42;
COL = [0.00 0.45 0.70; 0.95 0.60 0.00; 0.85 0.15 0.15];
sel = readtable('results/diagnostic_selection.csv', 'TextType', 'string');

f = figure('Visible', 'off', 'Position', [20 20 1700 1450]);  theme(f, 'light');
tl = tiledlayout(f, 4, height(sel), 'TileSpacing', 'tight', 'Padding', 'compact');
rowname = {'Normal scanner', 'Cheap low-field scanner', 'Expert tumor outline', 'Zoom: normal | low-field'};
for c = 1:height(sel)
    id = char(sel.patient_id(c));
    lab = niftiread(fullfile(DATA_ROOT, id, [id '_seg.nii']));
    fl  = single(niftiread(fullfile(DATA_ROOT, id, [id '_flair.nii'])));
    lab = lab(ROWS, COLS, :);  fl = fl(ROWS, COLS, :);
    mu = mean(fl(fl > 0));
    [~, z] = max(squeeze(sum(sum(lab > 0, 1), 2)));
    x = fl(:, :, z);  bm = x > 0;  L = lab(:, :, z);
    rs = RandStream('mt19937ar', 'Seed', SEED);
    nr = randn(rs, size(x), 'single');  ni = randn(rs, size(x), 'single');
    xd = degrade_slice(x, mu, SNR, R, nr, ni);
    zs = @(im) (im - mean(im(bm))) / std(im(bm));
    a = rot90(zs(x));  b = rot90(zs(xd));  Lr = rot90(L);

    ax = nexttile(tl, c);
    imshow(a, [-2 4], 'Parent', ax);
    title(ax, {sprintf('%s', sel.role(c)), strrep(id, 'BraTS20_Training_', 'patient ')}, ...
        'FontSize', 12, 'FontWeight', 'normal');
    ax = nexttile(tl, height(sel) + c);  imshow(b, [-2 4], 'Parent', ax);
    ax = nexttile(tl, 2 * height(sel) + c);  imshow(a, [-2 4], 'Parent', ax);  hold(ax, 'on');
    m = {Lr > 0, Lr == 1 | Lr == 4, Lr == 4};
    for k = 1:3
        if any(m{k}(:)), contour(ax, double(m{k}), [0.5 0.5], 'Color', COL(k, :), 'LineWidth', 1.8); end
    end
    % Zoom: tumor bounding box + margin, clean and degraded side by side
    [ri, ci] = find(Lr > 0);
    r1 = max(min(ri) - 8, 1);  r2 = min(max(ri) + 8, 192);
    c1 = max(min(ci) - 8, 1);  c2 = min(max(ci) + 8, 192);
    ax = nexttile(tl, 3 * height(sel) + c);
    imshow([a(r1:r2, c1:c2), 4 * ones(r2 - r1 + 1, 2), b(r1:r2, c1:c2)], [-2 4], 'Parent', ax);
end
for r = 1:4
    ax = nexttile(tl, (r - 1) * height(sel) + 1);
    ylabel(ax, rowname{r}, 'FontSize', 13, 'FontWeight', 'bold');  ax.YLabel.Visible = 'on';
end
title(tl, {'The 5 example patients (FLAIR, axial slice with the most tumor)', ...
    'blue = whole tumor incl. swelling, orange = tumor core, red = actively growing tumor'}, ...
    'FontSize', 14);
exportgraphics(f, 'figures/gallery_cases.png', 'Resolution', 130);  close(f);
fprintf('Saved figures/gallery_cases.png\n');
