% RENDER3D_DEMO  Roadmap step 1.3: test the 3D renderer on damaged masks.
%
% Run from repo root:  matlab -batch "run('matlab/render3d_demo.m')"
%
% Renders one test patient's truth against (a) itself, (b) a 2 mm dilation,
% (c) a 2 mm erosion, (d) a 2-voxel shift, (e) 3 spurious blobs. Because the
% damage is known, the error heatmap's numbers are known too:
%   self      -> mean signed error ~0, mean |error| < 0.5 mm
%   dilate 2  -> mean signed error ~ +2 mm (every boundary point moved out 2 mm)
%   erode 2   -> mean signed error ~ -2 mm
% (Not the 1.83 / 1.65 "volume per surface" factors of Step 1.2: those measure
% voxels gained per mm^2 on a grid, this measures distance, which is exact.)
% The asserts below check this, so the renderer's error map is validated, not
% just pretty. In Stage 3 call render_tumor_3d with real predictions.
%
% Outputs: figures/render3d_<case>.png ; optional STL meshes in results/stl/
% (STL files are large: kept out of git via .git/info/exclude).

cd(fileparts(fileparts(mfilename('fullpath'))));
addpath('matlab');
DATA_ROOT  = 'archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData';
PATIENT    = 'BraTS20_Training_002';   % same patient as the Step 1.2 damage figure
EXPORT_STL = false;
SEED       = 42;

info  = niftiinfo(fullfile(DATA_ROOT, PATIENT, [PATIENT '_seg.nii']));
lab   = niftiread(info);
sp    = info.PixelDimensions(1:3);
flair = niftiread(fullfile(DATA_ROOT, PATIENT, [PATIENT '_flair.nii']));
brain = flair > 0;
rs = RandStream('mt19937ar', 'Seed', SEED);

cases = {
    'self',      'identical (sanity check)', [],       []
    'dilate2',   'dilated 2 mm',             'dilate', 2
    'erode2',    'eroded 2 mm',              'erode',  2
    'shift2',    'shifted 2 voxels',         'shift',  2
    'blobs',     '3 spurious blobs',         'blobs',  [3 3]
};
R = table();
for c = 1:size(cases, 1)
    if isempty(cases{c, 3})
        pred = lab;
    else
        pred = damage_mask(lab, cases{c, 3}, cases{c, 4}, rs, brain);
    end
    stl = '';
    if EXPORT_STL
        if ~isfolder('results/stl'), mkdir('results/stl'); end
        stl = sprintf('results/stl/%s_%s', PATIENT, cases{c, 1});
    end
    tic;
    E = render_tumor_3d(flair, lab, pred, sp, sprintf('figures/render3d_%s.png', cases{c, 1}), ...
        sprintf('%s: truth vs %s', PATIENT, cases{c, 2}), stl);
    fprintf('%-8s mean signed %+.2f mm, mean |d| %.2f mm, p95 |d| %.2f mm, >2 mm %.1f%%  (%.0f s)\n', ...
        cases{c, 1}, E.mean_signed_mm, E.mean_abs_mm, E.p95_abs_mm, 100 * E.frac_over_2mm, toc);
    R = [R; struct2table(E)]; %#ok<AGROW>
end
R.case = cases(:, 1);
writetable(R, 'results/render3d_surface_error.csv');

% Known-answer checks on the error map
assert(abs(R.mean_signed_mm(1)) < 0.2 && R.mean_abs_mm(1) < 0.5, 'self error');
assert(abs(R.mean_signed_mm(2) - 2) < 0.2, 'dilate 2 mm signed error');
assert(abs(R.mean_signed_mm(3) + 2) < 0.2, 'erode 2 mm signed error');
fprintf('RENDER3D CHECKS PASSED\n');
