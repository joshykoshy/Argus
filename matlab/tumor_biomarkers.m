function T = tumor_biomarkers(labels, spacing)
% TUMOR_BIOMARKERS  Shape biomarkers for WT / TC / ET of one segmentation volume.
%
%   T = tumor_biomarkers(labels, spacing)
%
%   labels  - 3D integer label volume. Accepts both conventions:
%             raw BraTS (1 = NCR/NET, 2 = edema, 4 = ET) and the Python
%             pipeline's remapped predictions (ET = 3). Raw files never use 3,
%             so "ET = 3 or 4" is safe for both.
%   spacing - voxel size in mm [dx dy dz], read from the NIfTI header.
%
%   T       - 3-row table (WT, TC, ET) with:
%     voxels, volume_cm3          count and physical volume
%     surface_area_mm2            regionprops3 SurfaceArea summed over the
%                                 26-connected pieces (primary)
%     surface_area_mesh_mm2       marching-cubes mesh after Gaussian sigma=0.5
%                                 smoothing (cross-check)
%     sphericity, sphericity_mesh pi^(1/3) * (6V)^(2/3) / A, 1 = perfect ball
%     elongation                  2nd / 1st principal axis length, in (0,1];
%                                 1 = round, small = cigar-shaped
%     n_components                separate 26-connected pieces
%     frac_outside_largest        share of volume NOT in the biggest piece
%                                 (fragmented false positives push this up)
%
%   Empty region (ET is often absent in LGG): volume 0, n_components 0,
%   every shape metric NaN. No error.
%
%   Why regionprops3 and not a mesh as primary: in R2026b regionprops3 does
%   not count raw voxel faces (a 10^3 cube gives 524.6, not 600), and on
%   digitised spheres it is within 1% of 4*pi*R^2 for R >= 8 voxels. The mesh
%   is kept as an independent check. See test_tumor_biomarkers.m.
%
%   Ground truth and predictions MUST go through this same function.

assert(numel(spacing) >= 3 && all(abs(spacing(1:3) - spacing(1)) < 1e-6), ...
    'tumor_biomarkers:anisotropic', ...
    'Anisotropic spacing %s: regionprops3 works in voxel units, resample first.', ...
    mat2str(spacing));
% ponytail: isotropic only (all BraTS volumes are 1 mm). Anisotropic data would
% need resampling to isotropic before regionprops3.
s = double(spacing(1));

regions = {'WT', [1 2 3 4]; 'TC', [1 3 4]; 'ET', [3 4]};
rows = cell(3, 1);
for k = 1:3
    rows{k} = measure(ismember(labels, regions{k, 2}), s);
end
T = struct2table([rows{:}]');
T = addvars(T, regions(:, 1), 'Before', 1, 'NewVariableNames', 'region');
end

function r = measure(m, s)
n = nnz(m);
r = struct('voxels', n, 'volume_cm3', n * s^3 / 1000, ...
    'surface_area_mm2', NaN, 'surface_area_mesh_mm2', NaN, ...
    'sphericity', NaN, 'sphericity_mesh', NaN, 'elongation', NaN, ...
    'n_components', 0, 'frac_outside_largest', NaN);
if n == 0, return; end

% Connected components, 26-connectivity (touching at faces, edges or corners
% counts as one piece).
cc = bwconncomp(m, 26);
sizes = cellfun(@numel, cc.PixelIdxList);
r.n_components = cc.NumObjects;
r.frac_outside_largest = 1 - max(sizes) / n;

% Crop to the tumor's bounding box plus a 3-voxel zero border: much faster
% than working on the full 240x240x155 volume, and the border lets the
% smoothing kernel and the mesh close properly.
[i, j, k] = ind2sub(size(m), find(m));
box = padarray(m(min(i):max(i), min(j):max(j), min(k):max(k)), [3 3 3]);

% Surface area: measured per connected piece, then summed. regionprops3's
% SurfaceArea on a single label covering several disconnected pieces returns
% the area of ONE piece only (two equal spheres -> area of one; sphere + a
% far stray voxel -> 3 mm^2). Verified on R2026b; see test 6.
sa = regionprops3(bwconncomp(box, 26), 'SurfaceArea');
V = n * s^3;                                        % mm^3
r.surface_area_mm2 = sum(sa.SurfaceArea) * s^2;
% Principal axes: uint8(box) labels all tumor voxels as region 1, so the axes
% describe the whole tumor (all pieces). This property IS computed over every
% voxel of the label, unlike SurfaceArea.
rp = regionprops3(uint8(box), 'PrincipalAxisLength');
r.sphericity = pi^(1/3) * (6 * V)^(2/3) / r.surface_area_mm2;
ax = sort(rp.PrincipalAxisLength(1, :), 'descend');
r.elongation = ax(2) / ax(1);                       % 0/0 -> NaN for 1 voxel

% Mesh cross-check: blur slightly, take the 0.5 iso-surface (marching cubes),
% sum triangle areas. sigma=0.5 voxel: sigma=1 erases small ET blobs.
fv = isosurface(imgaussfilt3(double(box), 0.5), 0.5);
if ~isempty(fv.faces)
    a = fv.vertices(fv.faces(:, 2), :) - fv.vertices(fv.faces(:, 1), :);
    b = fv.vertices(fv.faces(:, 3), :) - fv.vertices(fv.faces(:, 1), :);
    r.surface_area_mesh_mm2 = sum(vecnorm(cross(a, b, 2), 2, 2)) / 2 * s^2;
    r.sphericity_mesh = pi^(1/3) * (6 * V)^(2/3) / r.surface_area_mesh_mm2;
end
end
