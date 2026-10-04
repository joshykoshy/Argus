% FIGURES_VOLUMETRICS  Figures for roadmap step 1.1 (method validation + GT cohort).
%
% Run from repo root:  matlab -batch "run('matlab/figures_volumetrics.m')"
% Needs results/volumetrics_gt.csv (run matlab/volumetrics.m first).
%
% Fig 1  figures/surface_area_validation.png
%        Digital spheres of radius 2-25 voxels, true area 4*pi*R^2 known.
%        Compares three surface-area estimators:
%          raw voxel faces  (the "staircase" count, what a naive method gives)
%          regionprops3     (primary method in tumor_biomarkers.m)
%          mesh, sigma=0.5  (cross-check in tumor_biomarkers.m)
% Fig 2  figures/volumetrics_gt_cohort.png
%        Ground-truth biomarker distributions over the 74 test patients.

cd(fileparts(fileparts(mfilename('fullpath'))));
addpath('matlab');
IN_CSV = 'results/volumetrics_gt.csv';
FIG1   = 'figures/surface_area_validation.png';
FIG2   = 'figures/volumetrics_gt_cohort.png';
set(groot, 'defaultAxesFontSize', 12, 'defaultTextFontSize', 12, ...
    'defaultAxesTickDir', 'out', 'defaultAxesBox', 'off');
C = struct('WT', [0.00 0.45 0.70], 'TC', [0.90 0.60 0.00], 'ET', [0.80 0.20 0.20]);

%% ---- Fig 1: surface-area estimator validation on spheres ----
R = 2:25;
ratio = zeros(numel(R), 3);
for i = 1:numel(R)
    [x, y, z] = ndgrid(-R(i)-3:R(i)+3);
    m = x.^2 + y.^2 + z.^2 <= R(i)^2;
    A = 4 * pi * R(i)^2;
    % Raw face count: every boundary between a tumor and a non-tumor voxel
    % is one 1x1 mm face. diff() along each axis finds those boundaries.
    faces = nnz(diff(m, 1, 1)) + nnz(diff(m, 1, 2)) + nnz(diff(m, 1, 3));
    T = tumor_biomarkers(uint8(2 * m), [1 1 1]);
    ratio(i, :) = [faces, T.surface_area_mm2(1), T.surface_area_mesh_mm2(1)] / A;
end
f = figure('Visible', 'off', 'Position', [100 100 760 460]);
theme(f, 'light');   % R2025a+ defaults to dark figures
hold on;
plot(R, 100 * (ratio(:, 1) - 1), 's-', 'LineWidth', 1.5, 'Color', [0.5 0.5 0.5]);
plot(R, 100 * (ratio(:, 2) - 1), 'o-', 'LineWidth', 1.8, 'Color', C.WT);
plot(R, 100 * (ratio(:, 3) - 1), '^-', 'LineWidth', 1.5, 'Color', C.TC);
yline(0, 'k:');
xlabel('Sphere radius (mm, 1 mm voxels)');
ylabel('Surface area error vs 4\piR^2 (%)');
legend({'Raw voxel faces (staircase)', 'regionprops3 (primary)', ...
        'Mesh, Gaussian \sigma = 0.5 (cross-check)'}, 'Location', 'east');
title('Surface-area estimators on digital spheres');
grid on;
exportgraphics(f, FIG1, 'Resolution', 200);  close(f);
fprintf('Saved %s\n', FIG1);
fprintf('Raw faces error: %.0f%%..%.0f%%; regionprops3 at R>=8: max |err| %.1f%%; mesh at R>=8: max |err| %.1f%%\n', ...
    100 * (min(ratio(:, 1)) - 1), 100 * (max(ratio(:, 1)) - 1), ...
    100 * max(abs(ratio(R >= 8, 2) - 1)), 100 * max(abs(ratio(R >= 8, 3) - 1)));

%% ---- Fig 2: ground-truth cohort ----
T = readtable(IN_CSV, 'TextType', 'string');
regs = ["WT", "TC", "ET"];
f = figure('Visible', 'off', 'Position', [100 100 1500 440]);
theme(f, 'light');
tl = tiledlayout(f, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

% (a) volume by region and grade. ET-empty patients plotted at 0.
nexttile; hold on;
for k = 1:3
    for g = ["HGG", "LGG"]
        v = T.volume_cm3(T.region == regs(k) & T.grade == g);
        xpos = k + 0.18 * (2 * (g == "LGG") - 1);
        boxchart(repmat(xpos, size(v)), v, 'BoxWidth', 0.3, ...
            'BoxFaceColor', C.(regs(k)), 'MarkerStyle', 'none', ...
            'BoxFaceAlpha', 0.25 + 0.35 * (g == "HGG"));
        rng(1);  % fixed jitter
        scatter(xpos + 0.06 * randn(size(v)), v, 14, C.(regs(k)), 'filled', ...
            'MarkerFaceAlpha', 0.6);
    end
end
xticks([0.82 1.18 1.82 2.18 2.82 3.18]);
xticklabels({'WT HGG', 'WT LGG', 'TC HGG', 'TC LGG', 'ET HGG', 'ET LGG'});
ylabel('Volume (cm^3)');
title('(a) Volume by region and grade');

% (b) sphericity: regionprops3 vs mesh, every non-empty region
nexttile; hold on;
for k = 1:3
    r = T(T.region == regs(k) & T.voxels > 0, :);
    scatter(r.sphericity, r.sphericity_mesh, 22, C.(regs(k)), 'filled', ...
        'MarkerFaceAlpha', 0.7, 'DisplayName', regs(k));
end
plot([0 1], [0 1], 'k:', 'DisplayName', 'identity');
axis([0 1 0 1]); axis square;
xlabel('Sphericity, regionprops3');
ylabel('Sphericity, mesh (\sigma = 0.5)');
legend('Location', 'southeast');
title('(b) Sphericity: primary vs mesh');

% (c) fragmentation: share of volume outside the largest piece
nexttile; hold on;
for k = 1:3
    r = T(T.region == regs(k) & T.voxels > 0, :);
    v = sort(100 * r.frac_outside_largest);
    stairs(v, (1:numel(v)) / numel(v), 'LineWidth', 1.8, 'Color', C.(regs(k)), ...
        'DisplayName', regs(k));
end
set(gca, 'XScale', 'log');  xlim([1e-3 100]);
xlabel('Volume outside largest component (%, log scale)');
ylabel('Cumulative fraction of patients');
legend('Location', 'southeast');
title('(c) Expert labels are slightly fragmented');
grid on;
exportgraphics(f, FIG2, 'Resolution', 200);  close(f);
fprintf('Saved %s\n', FIG2);

% Numbers quoted in the lab notebook
for k = 1:3
    r = T(T.region == regs(k) & T.voxels > 0, :);
    % Spearman = Pearson on ranks (no Statistics Toolbox here; continuous
    % values, so ties are not a concern).
    [~, ia] = sort(r.sphericity);       ra(ia) = 1:height(r);
    [~, ib] = sort(r.sphericity_mesh);  rb(ib) = 1:height(r);
    cc = corrcoef(ra, rb);  rho = cc(1, 2);  clear ra rb
    fprintf('%s: Spearman(rp, mesh) = %.3f, median mesh/rp sphericity ratio = %.3f\n', ...
        regs(k), rho, median(r.sphericity_mesh ./ r.sphericity));
end
