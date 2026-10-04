% TEST_TUMOR_BIOMARKERS  Checks tumor_biomarkers.m on shapes with known answers.
%
% Run from repo root:  matlab -batch "run('matlab/test_tumor_biomarkers.m')"
% Every check is an assert; the script errors on the first failure.

cd(fileparts(mfilename('fullpath')));
[x, y, z] = ndgrid(-40:40);

% 1. Sphere R=20 mm at 1 mm voxels, labelled as whole-tumor edema (2).
R = 20;
lab = uint8(2 * (x.^2 + y.^2 + z.^2 <= R^2));
T = tumor_biomarkers(lab, [1 1 1]);
wt = T(strcmp(T.region, 'WT'), :);
fprintf('Sphere R=20: V=%.2f cm3 (true %.2f), SA=%.0f / mesh %.0f (true %.0f), sph=%.3f / mesh %.3f, elong=%.3f\n', ...
    wt.volume_cm3, 4/3*pi*R^3/1000, wt.surface_area_mm2, wt.surface_area_mesh_mm2, ...
    4*pi*R^2, wt.sphericity, wt.sphericity_mesh, wt.elongation);
assert(abs(wt.volume_cm3 / (4/3*pi*R^3/1000) - 1) < 0.01, 'volume');
assert(abs(wt.surface_area_mm2 / (4*pi*R^2) - 1) < 0.02, 'regionprops3 SA');
assert(abs(wt.surface_area_mesh_mm2 / (4*pi*R^2) - 1) < 0.05, 'mesh SA');
assert(abs(wt.sphericity - 1) < 0.03, 'sphericity');
assert(wt.elongation > 0.98, 'elongation of sphere');
assert(wt.n_components == 1 && wt.frac_outside_largest == 0, 'components');

% 2. Empty regions: no 1/3/4 labels -> TC and ET empty -> NaN, not an error.
for reg = {'TC', 'ET'}
    e = T(strcmp(T.region, reg{1}), :);
    assert(e.volume_cm3 == 0 && e.n_components == 0, 'empty volume');
    assert(isnan(e.sphericity) && isnan(e.surface_area_mm2) && isnan(e.elongation), 'empty NaN');
end

% 3. Spacing is honoured: the same physical sphere (R=20 mm) sampled at 2 mm
%    voxels (R=10 voxels) must give the same volume and area in mm.
lab2 = uint8(2 * (x.^2 + y.^2 + z.^2 <= 10^2));
T2 = tumor_biomarkers(lab2, [2 2 2]);
assert(abs(T2.volume_cm3(1) / (4/3*pi*R^3/1000) - 1) < 0.02, '2 mm volume');
assert(abs(T2.surface_area_mm2(1) / (4*pi*R^2) - 1) < 0.03, '2 mm SA');

% 4. Anisotropic spacing is rejected rather than silently wrong.
try
    tumor_biomarkers(lab, [1 1 2]); error('should have thrown');
catch ME
    assert(strcmp(ME.identifier, 'tumor_biomarkers:anisotropic'), ME.message);
end

% 5. Ellipsoid 20 x 10 x 10: elongation = 10/20 = 0.5.
lab = uint8(2 * ((x/20).^2 + (y/10).^2 + (z/10).^2 <= 1));
T = tumor_biomarkers(lab, [1 1 1]);
fprintf('Ellipsoid 20x10x10: elongation=%.3f (true 0.5)\n', T.elongation(1));
assert(abs(T.elongation(1) - 0.5) < 0.02, 'ellipsoid elongation');

% 6. Fragmentation: sphere + one detached 3x3x3 blob (27 voxels).
lab = uint8(2 * (x.^2 + y.^2 + z.^2 <= 15^2));
lab(75:77, 75:77, 75:77) = 2;          % grid index 75 = coordinate +34, outside R=15
T = tumor_biomarkers(lab, [1 1 1]);
assert(T.n_components(1) == 2, 'two components');
assert(abs(T.frac_outside_largest(1) - 27 / T.voxels(1)) < 1e-12, 'fraction outside largest');

% 7. Label conventions: ET as 4 (raw BraTS) and as 3 (Python predictions)
%    give identical results. Core = NCR(1) + ET, WT adds edema(2).
core = x.^2 + y.^2 + z.^2 <= 8^2;
lab4 = uint8(2 * (x.^2 + y.^2 + z.^2 <= 15^2));
lab4(core) = 1;  lab4(core & x > 0) = 4;
lab3 = lab4;  lab3(lab4 == 4) = 3;
T4 = tumor_biomarkers(lab4, [1 1 1]);  T3 = tumor_biomarkers(lab3, [1 1 1]);
assert(isequaln(T4, T3), 'label 3 vs 4');
assert(T4.voxels(2) == nnz(core) && T4.voxels(3) == nnz(core & x > 0), 'TC/ET definitions');

fprintf('ALL TUMOR_BIOMARKERS TESTS PASSED\n');
