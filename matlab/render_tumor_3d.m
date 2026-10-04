function E = render_tumor_3d(flair, truth, pred, spacing, out_png, ttl, stl_prefix)
% RENDER_TUMOR_3D  3D truth-vs-prediction figure with a surface error heatmap.
%
%   E = render_tumor_3d(flair, truth, pred, spacing, out_png, ttl)
%   E = render_tumor_3d(..., stl_prefix)     also write STL meshes
%
%   flair   - FLAIR volume (skull-stripped; brain = voxels > 0)
%   truth   - ground-truth label volume (1 NCR, 2 edema, 3/4 ET)
%   pred    - predicted (or synthetically damaged) label volume, same size
%   spacing - voxel size in mm [dx dy dz] from the NIfTI header
%   out_png - output figure path ('' = do not save)
%   ttl     - figure title
%   stl_prefix - optional; writes <prefix>_{truth,pred}_{WT,TC,ET}.stl
%
%   Figure: 2 rows (lateral and top views) x 3 columns:
%     1. truth       translucent brain + WT (blue, see-through), TC (orange), ET (red)
%     2. prediction  same rendering
%     3. error map   predicted WT surface, every vertex coloured by its SIGNED
%                    distance (mm) to the true WT boundary:
%                    red   = prediction sticks OUT of the truth (over-segmentation)
%                    blue  = prediction lies INSIDE the truth (under-segmentation)
%                    white = on the true boundary
%
%   E (struct) - per-vertex error summary for the predicted WT surface:
%     mean_signed_mm, mean_abs_mm, p95_abs_mm, frac_over_2mm
%
%   Meshes use the same light smoothing (Gaussian sigma = 0.5 voxel) and
%   0.5 iso-level as the surface cross-check in tumor_biomarkers.m.

if nargin < 6, ttl = ''; end
if nargin < 7, stl_prefix = ''; end
sp = double(spacing(1:3));
% isosurface returns vertices as (x = column, y = row, z = slice), in voxels.
% Convert to mm with the matching spacing order.
tomm = @(v) (v - 1) .* sp([2 1 3]);

regions = {@(L) L > 0, @(L) L == 1 | L == 3 | L == 4, @(L) L == 3 | L == 4};
names = {'WT', 'TC', 'ET'};
col = [0.00 0.45 0.70; 0.95 0.60 0.00; 0.85 0.15 0.15];
alph = [0.20 0.35 1.00];               % WT, TC see-through so ET shows inside

% Brain surface: downsample 2x for speed (it is context only), smooth, mesh.
b = imgaussfilt3(double(flair(1:2:end, 1:2:end, 1:2:end) > 0), 1.5);
brain = isosurface(b, 0.5);
brain.vertices = (brain.vertices - 1) * 2 + 1;     % back to full-res voxel units
brain = reducepatch(brain, 0.25);
brain.vertices = tomm(brain.vertices);

mesh = @(m) isosurface(imgaussfilt3(double(m), 0.5), 0.5);
T = cell(2, 3);  vols = {truth, pred};
for v = 1:2
    for k = 1:3
        T{v, k} = mesh(regions{k}(vols{v}));
        T{v, k}.vertices = tomm(T{v, k}.vertices);
        if ~isempty(stl_prefix) && ~isempty(T{v, k}.faces)
            who = {'truth', 'pred'};
            stlwrite(triangulation(double(T{v, k}.faces), T{v, k}.vertices), ...
                sprintf('%s_%s_%s.stl', stl_prefix, who{v}, names{k}));
        end
    end
end

% Signed distance to the true WT boundary, in mm:
%   outside the truth: + distance to the nearest truth voxel
%   inside  the truth: - distance to the nearest non-truth voxel
% bwdist works in voxels; spacing is isotropic for BraTS (checked upstream).
wt = regions{1}(truth);
sdf = (bwdist(wt) - bwdist(~wt)) * sp(1);
P = T{2, 1};
% Sample the signed distance at every predicted-surface vertex. interp3 takes
% (x = column, y = row, z = slice) in voxel units, the isosurface convention.
vox = P.vertices ./ sp([2 1 3]) + 1;
d = interp3(sdf, vox(:, 1), vox(:, 2), vox(:, 3), 'linear', NaN);
% The surface sits half-way between inside (-) and outside (+) voxels, so
% sdf on a perfect match reads ~0 (it is 0 or +-1 on voxel centres, and the
% 0.5 iso-level interpolates to ~0). No offset correction is applied.
E.mean_signed_mm = mean(d, 'omitnan');
E.mean_abs_mm = mean(abs(d), 'omitnan');
E.p95_abs_mm = interp1(linspace(0, 100, nnz(~isnan(d))), sort(abs(d(~isnan(d)))), 95);
E.frac_over_2mm = mean(abs(d(~isnan(d))) > 2);

if isempty(out_png), return; end

% ---- Figure ----
f = figure('Visible', 'off', 'Position', [50 50 1500 900], 'Color', 'w');
theme(f, 'light');
tl = tiledlayout(f, 2, 3, 'TileSpacing', 'none', 'Padding', 'compact');
% BraTS arrays: dim 1 = left-right, dim 2 = front-back (isosurface x = dim 2,
% y = dim 1). az 0 looks along y = from the side; el 90 = from above.
views = {[0 10], [0 90]};
vname = {'side view', 'top view'};
lim = 5;                                % colour range for the error map, mm
cmap = diverging_map(256);
for r = 1:2
    for c = 1:3
        ax = nexttile;  hold(ax, 'on');
        patch(ax, brain, 'FaceColor', [0.75 0.75 0.75], 'EdgeColor', 'none', ...
            'FaceAlpha', 0.10);
        if c < 3
            for k = 1:3
                if isempty(T{c, k}.faces), continue; end
                patch(ax, T{c, k}, 'FaceColor', col(k, :), 'EdgeColor', 'none', ...
                    'FaceAlpha', alph(k));
            end
            if r == 1
                lbl = {'Ground truth', 'Prediction'};
                title(ax, lbl{c}, 'FontSize', 13);
            end
        else
            if ~isempty(P.faces)
                patch(ax, 'Faces', P.faces, 'Vertices', P.vertices, ...
                    'FaceVertexCData', d, 'FaceColor', 'interp', 'EdgeColor', 'none');
            end
            colormap(ax, cmap);  clim(ax, [-lim lim]);
            if r == 1
                title(ax, sprintf('WT surface error (mean |d| %.2f mm)', E.mean_abs_mm), ...
                    'FontSize', 13);
            else
                cb = colorbar(ax, 'southoutside');
                cb.Label.String = 'Signed distance to true boundary (mm): - inside, + outside';
                cb.FontSize = 11;
            end
        end
        axis(ax, 'equal', 'off', 'vis3d');
        view(ax, views{r});
        camlight(ax, 'headlight');  lighting(ax, 'gouraud');  material(ax, 'dull');
        if c == 1
            text(ax, 0, 0.5, vname{r}, 'Units', 'normalized', 'Rotation', 90, ...
                'HorizontalAlignment', 'center', 'FontSize', 12);
        end
    end
end
if ~isempty(ttl)
    % Figure-level title above the layout (a tiledlayout title collides with
    % the panel titles when tiles have no spacing).
    tl.OuterPosition = [0 0 1 0.95];
    annotation(f, 'textbox', [0 0.95 1 0.05], 'String', ttl, 'EdgeColor', 'none', ...
        'HorizontalAlignment', 'center', 'FontSize', 14, 'Interpreter', 'none');
end
% Legend for region colours (dummy patches, off-screen)
ax = nexttile(tl, 1);
h = gobjects(3, 1);
for k = 1:3
    h(k) = patch(ax, NaN, NaN, NaN, 'FaceColor', col(k, :), 'EdgeColor', 'none');
end
legend(ax, h, {'WT whole tumor', 'TC tumor core', 'ET enhancing'}, ...
    'Location', 'southwest', 'FontSize', 10);
exportgraphics(f, out_png, 'Resolution', 150);
close(f);
end

function m = diverging_map(n)
% Blue -> white -> red, perceptually balanced enough for a signed error.
x = linspace(-1, 1, n)';
blue = [0.15 0.35 0.75];  red = [0.80 0.15 0.15];
m = zeros(n, 3);
neg = x < 0;
m(neg, :)  = 1 - (-x(neg)) .* (1 - blue);
m(~neg, :) = 1 - x(~neg) .* (1 - red);
end
