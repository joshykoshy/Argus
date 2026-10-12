% RENDER3D_STAGE3  Roadmap step 3.5: 3D truth vs prediction, M0 vs M4, worst condition.
%
% Run from repo root:  matlab -batch "run('matlab/render3d_stage3.m')"
% Needs results/diagnostic_selection.csv (diagnostic_report.m) and the seed-0
% predictions in results/predictions/.
%
% For each of the 5 blindly selected cases, renders the ground truth against
% M0 (clean-trained) and M4 (dual-stream) at the DEC-005 worst condition
% (SNR 3, r 0.5, 5 mm) with render_tumor_3d.m, the renderer validated on known
% damage in render3d_demo.m (dilate 2 mm -> +2.0 mm). The error map colours
% every vertex of the predicted WT surface by its signed distance (mm) to the
% true WT boundary: red = over-segmentation, blue = under-segmentation.
%
% Outputs: figures/render3d_stage3_<case>_<model>.png
%          results/render3d_stage3_surface_error.csv

cd(fileparts(fileparts(mfilename('fullpath'))));
addpath('matlab');
DATA_ROOT = 'archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData';
PRED_FMT  = 'results/predictions/%s_%s_kspace_snr3_r0.5_t5.nii.gz';
MODELS    = {'M0', 'M4'};

sel = readtable('results/diagnostic_selection.csv', 'TextType', 'string');
R = table();
for c = 1:height(sel)
    id = char(sel.patient_id(c));
    info  = niftiinfo(fullfile(DATA_ROOT, id, [id '_seg.nii']));
    truth = niftiread(info);
    sp    = info.PixelDimensions(1:3);
    flair = niftiread(fullfile(DATA_ROOT, id, [id '_flair.nii']));
    for m = 1:numel(MODELS)
        pred = niftiread(sprintf(PRED_FMT, MODELS{m}, id));
        tic;
        E = render_tumor_3d(flair, truth, pred, sp, ...
            sprintf('figures/render3d_stage3_%d_%s.png', c, MODELS{m}), ...
            sprintf('Case %d (%s): truth vs %s, SNR 3, r 0.5, 5 mm', c, sel.role(c), MODELS{m}));
        fprintf('case %d %s %s: mean signed %+.2f mm, mean |d| %.2f mm, p95 %.2f mm, >2 mm %.1f%% (%.0f s)\n', ...
            c, id, MODELS{m}, E.mean_signed_mm, E.mean_abs_mm, E.p95_abs_mm, 100 * E.frac_over_2mm, toc);
        E.case = c;  E.patient_id = string(id);  E.model = string(MODELS{m});
        R = [R; struct2table(E)]; %#ok<AGROW>
    end
end
R = movevars(R, {'case', 'patient_id', 'model'}, 'Before', 1);
writetable(R, 'results/render3d_stage3_surface_error.csv');
disp(R);
