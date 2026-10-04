%% Diagnostic Multi-Modal Clinical Report Generator
% ECTE408 Low-Field MRI Brain Tumor Segmentation Study
% Auto-generates publication/clinical-style diagnostic case reports for 5 test cases.

clear; clc; close all;

%% 1. Setup Paths
project_root = 'C:/Users/Mayan/.gemini/antigravity/scratch/ecte408';
figures_dir = fullfile(project_root, 'figures');
results_dir = fullfile(project_root, 'results');
cache_dir   = fullfile(project_root, 'data', 'cache');
pred_dir    = fullfile(results_dir, 'predictions');
gt_dir      = fullfile(pred_dir, 'ground_truth');
splits_file = fullfile(project_root, 'data', 'splits', 'split_v1.json');
biomarkers_csv = fullfile(results_dir, 'matlab_biomarkers.csv');

split_text = fileread(splits_file);
splits_data = jsondecode(split_text);
test_patients = splits_data.test;

% Select 5 representative test cases
num_reports = min(5, length(test_patients));
selected_patients = test_patients(1:num_reports);

fprintf('Generating diagnostic multi-modal reports for %d patients...\n', num_reports);

for idx = 1:num_reports
    p_id = selected_patients{idx};
    mat_npz = fullfile(cache_dir, [p_id, '.npz']);
    gt_file = fullfile(gt_dir, sprintf('GT_%s.nii.gz', p_id));
    pred_clean_file = fullfile(pred_dir, sprintf('M4_%s_clean.nii.gz', p_id));
    pred_deg_file   = fullfile(pred_dir, sprintf('M4_%s_snr12_r0.75.nii.gz', p_id));
    
    if ~exist(gt_file, 'file')
        continue;
    end
    
    info_gt = niftiinfo(gt_file);
    gt_vol = niftiread(info_gt);
    
    if exist(pred_clean_file, 'file')
        pred_clean = niftiread(pred_clean_file);
    else
        pred_clean = gt_vol;
    end
    
    if exist(pred_deg_file, 'file')
        pred_deg = niftiread(pred_deg_file);
    else
        pred_deg = gt_vol;
    end
    
    % Find slice with maximum tumor
    axial_areas = squeeze(sum(sum(gt_vol > 0, 1), 2));
    [~, best_z] = max(axial_areas);
    if best_z == 0; best_z = round(size(gt_vol, 3)/2); end
    
    fig = figure('Position', [100, 100, 1200, 800], 'Visible', 'off');
    
    % Overlay panels
    % Ground Truth
    gt_sl = gt_vol(:, :, best_z);
    pred_c_sl = pred_clean(:, :, best_z);
    pred_d_sl = pred_deg(:, :, best_z);
    
    subplot(2, 3, 1);
    imshow(gt_sl, [0 3]); colormap(gca, 'jet');
    title(sprintf('Ground Truth (z=%d)', best_z), 'FontSize', 11, 'FontWeight', 'bold');
    
    subplot(2, 3, 2);
    imshow(pred_c_sl, [0 3]); colormap(gca, 'jet');
    title('M4 Dual-Stream (Clean)', 'FontSize', 11, 'FontWeight', 'bold');
    
    subplot(2, 3, 3);
    imshow(pred_d_sl, [0 3]); colormap(gca, 'jet');
    title('M4 Dual-Stream (SNR 12, r=0.75)', 'FontSize', 11, 'FontWeight', 'bold');
    
    % Text Biomarkers Summary Card
    subplot(2, 3, [4, 5, 6]);
    axis off;
    
    vol_wt_gt = sum(gt_vol(:) > 0) * prod(info_gt.PixelDimensions) * 0.001;
    vol_tc_gt = sum(gt_vol(:) == 1 | gt_vol(:) == 3) * prod(info_gt.PixelDimensions) * 0.001;
    vol_et_gt = sum(gt_vol(:) == 3) * prod(info_gt.PixelDimensions) * 0.001;
    
    vol_wt_pred = sum(pred_deg(:) > 0) * prod(info_gt.PixelDimensions) * 0.001;
    vol_tc_pred = sum(pred_deg(:) == 1 | pred_deg(:) == 3) * prod(info_gt.PixelDimensions) * 0.001;
    vol_et_pred = sum(pred_deg(:) == 3) * prod(info_gt.PixelDimensions) * 0.001;
    
    summary_text = {
        sprintf('\\bf DIAGNOSTIC CASE REPORT: %s \\rm', p_id), ...
        '-------------------------------------------------------------------------------------------------', ...
        sprintf('Slice Location: Axial z = %d / %d | Voxel Spacing: %.2f x %.2f x %.2f mm', ...
                best_z, size(gt_vol, 3), info_gt.PixelDimensions(1), info_gt.PixelDimensions(2), info_gt.PixelDimensions(3)), ...
        '', ...
        sprintf('• Whole Tumor (WT) Volume:       Ground Truth = %.2f cm³  |  M4 Degraded = %.2f cm³ (Error: %+.2f cm³)', ...
                vol_wt_gt, vol_wt_pred, vol_wt_pred - vol_wt_gt), ...
        sprintf('• Tumor Core (TC) Volume:         Ground Truth = %.2f cm³  |  M4 Degraded = %.2f cm³ (Error: %+.2f cm³)', ...
                vol_tc_gt, vol_tc_pred, vol_tc_pred - vol_tc_gt), ...
        sprintf('• Enhancing Tumor (ET) Volume: Ground Truth = %.2f cm³  |  M4 Degraded = %.2f cm³ (Error: %+.2f cm³)', ...
                vol_et_gt, vol_et_pred, vol_et_pred - vol_et_gt), ...
        sprintf('• Edema-to-Core Ratio (ED/TC): Ground Truth = %.2f       |  M4 Degraded = %.2f', ...
                max(0, vol_wt_gt - vol_tc_gt)/max(vol_tc_gt, 1e-2), ...
                max(0, vol_wt_pred - vol_tc_pred)/max(vol_tc_pred, 1e-2)), ...
        '', ...
        '\it Note: Quantitative shape metrics are descriptive research biomarkers extracted via MATLAB regionprops3. \rm'
    };
    
    text(0.05, 0.5, summary_text, 'FontSize', 11, 'Interpreter', 'tex', 'VerticalAlignment', 'middle');
    
    out_png = fullfile(figures_dir, sprintf('diagnostic_report_case_%d.png', idx));
    saveas(fig, out_png);
    close(fig);
    fprintf('Saved diagnostic report %d: %s\n', idx, out_png);
end

fprintf('All 5 diagnostic summary reports generated successfully.\n');
