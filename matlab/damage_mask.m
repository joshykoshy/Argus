function [out, info] = damage_mask(lab, type, param, rs, brain)
% DAMAGE_MASK  Apply a known, controlled error to a BraTS label volume.
%
%   [out, info] = damage_mask(lab, type, param, rs, brain)
%
%   Used to validate the agreement analysis: if we know exactly what damage
%   was done, we know what the agreement statistics should report.
%
%   lab   - label volume (1 NCR, 2 edema, 4 or 3 ET)
%   type  - 'dilate'  param = d mm   grow every region by d (Euclidean)
%           'erode'   param = d mm   shrink every region by d
%           'shift'   param = d vox  translate the whole tumor d voxels along dim 1
%           'delete'  param = r mm   remove all tumor inside a ball of radius r
%                                    centred on a random tumor voxel
%           'blobs'   param = [count r]  add `count` ET balls of radius r inside
%                                    the brain, >= 10 voxels from the tumor and
%                                    from each other (spurious false positives)
%   rs    - RandStream (fixed seed per patient) for 'delete' and 'blobs'
%   brain - logical brain mask (needed for 'blobs' only)
%
%   out   - damaged label volume, ET written as 4
%   info  - struct with the EXACT expected change for 'delete' and 'blobs':
%           info.removed / info.added = [WT TC ET] voxel counts,
%           info.new_pieces = blobs added (each is a separate component)
%
%   Every region is damaged separately and the label volume rebuilt, so the
%   nesting ET inside TC inside WT is preserved (dilation, erosion and shift
%   are monotone: A inside B stays A' inside B').
%
%   Spacing: assumes 1 mm isotropic voxels (true for all BraTS volumes).

wt = lab > 0;
tc = lab == 1 | lab == 3 | lab == 4;
et = lab == 3 | lab == 4;
info = struct('removed', [0 0 0], 'added', [0 0 0], 'new_pieces', 0);

switch type
    case 'dilate'
        % bwdist = distance (mm) from each voxel to the nearest region voxel.
        f = @(m) bwdist(m) <= param;
    case 'erode'
        f = @(m) bwdist(~m) > param;
    case 'shift'
        f = @(m) circshift(m, param, 1);
    case 'delete'
        idx = find(wt);
        [ci, cj, ck] = ind2sub(size(wt), idx(randi(rs, numel(idx))));
        [I, J, K] = ndgrid(1:size(wt, 1), 1:size(wt, 2), 1:size(wt, 3));
        ball = (I - ci).^2 + (J - cj).^2 + (K - ck).^2 <= param^2;
        info.removed = [nnz(wt & ball), nnz(tc & ball), nnz(et & ball)];
        f = @(m) m & ~ball;
    case 'blobs'
        n = param(1);  r = param(2);
        % Allowed centres: inside the brain, at least r from the brain edge,
        % and at least 10 + r from the tumor.
        ok = bwdist(~brain) > r & bwdist(wt) > 10 + r;
        blobs = false(size(wt));
        [I, J, K] = ndgrid(1:size(wt, 1), 1:size(wt, 2), 1:size(wt, 3));
        for b = 1:n
            cand = find(ok);
            c = cand(randi(rs, numel(cand)));
            [ci, cj, ck] = ind2sub(size(wt), c);
            d2 = (I - ci).^2 + (J - cj).^2 + (K - ck).^2;
            blobs = blobs | d2 <= r^2;
            ok = ok & d2 > (2 * r + 10)^2;        % keep the next blob separate
        end
        added = nnz(blobs);
        info.added = [added added added];
        info.new_pieces = n;
        f = @(m) m | blobs;
    otherwise
        error('Unknown damage type %s', type);
end

wt = f(wt);  tc = f(tc);  et = f(et);
out = zeros(size(lab), 'uint8');
out(wt) = 2;  out(tc) = 1;  out(et) = 4;
end
