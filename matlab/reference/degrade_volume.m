function y = degrade_volume(x, mu, cond, nr, ni)
% DEGRADE_VOLUME  MATLAB port of pilot/pilot.py::degrade_volume (DEC-005 physics).
%
%   y = degrade_volume(x, mu, cond)          noise from MATLAB's randn
%   y = degrade_volume(x, mu, cond, nr, ni)  noise from given N(0,1) arrays
%
%   x    - raw (un-normalised) volume [H x W x Z x C], single or double
%   mu   - mean clean brain intensity per channel, numel C
%   cond - struct with fields
%            snr   : linear SNR, or [] for no noise (the clean condition)
%            r     : fraction of k-space kept per in-plane axis (1.0 or 0.5)
%            thick : simulated slice thickness in mm (1 or 5)
%   nr, ni - optional real / imaginary N(0,1) noise, same size as x
%
% Three steps, in the order a scanner produces them:
%   1. Slice thickness: a t mm slice is the mean of t neighbouring 1 mm slices
%      (centred, edge slices replicated), kept on the 1 mm grid.
%   2. In-plane resolution: keep the central r fraction of 2D k-space.
%   3. Thermal noise in k-space, INSIDE the acquired band only (a scanner
%      records noise only where it samples). Band-limiting removes part of
%      the noise power, so it is rescaled by 1/sqrt(kept fraction) to keep the
%      per-pixel SD at sigma = mu / SNR. Magnitude of (image + noise) gives
%      Rician tissue and Rayleigh background.
% The older degrade_slice.m adds full-bandwidth noise after truncation (the
% original pipeline, finding C4); it is kept for the original parity test.
% Parity with Python: test_degrade_volume_parity.m.

[H, W, Z, C] = size(x);

% 1. Through-plane partial volume (replicate padding, as torch's pad 'replicate')
t = cond.thick;
if t > 1
    pb = floor(t / 2);  pa = t - 1 - pb;
    idx = [ones(1, pb), 1:Z, Z * ones(1, pa)];
    acc = zeros(size(x), 'like', x);
    for k = 0:t - 1
        acc = acc + x(:, :, idx((1:Z) + k), :);
    end
    x = acc / t;
end

% 2. In-plane k-space truncation (same mask as python generate_kspace_mask)
mask = kspace_mask(H, W, cond.r, x);
sh  = @(A) fftshift(fftshift(A, 1), 2);      % shift the two in-plane axes only
ish = @(A) ifftshift(ifftshift(A, 1), 2);
img = ifft2(ish(sh(fft2(x)) .* mask));        % complex

if isempty(cond.snr)
    y = real(img);
    return;
end

% 3. Band-limited complex noise
if nargin < 5
    nr = randn(size(x), 'like', x);
    ni = randn(size(x), 'like', x);
end
sigma = reshape(mu(:) ./ cond.snr, 1, 1, 1, C);
frac = mean(mask(:));
nbl = ifft2(ish(sh(fft2(complex(nr, ni))) .* mask)) / sqrt(frac);
y = abs(img + nbl .* sigma);
end

function m = kspace_mask(H, W, r, like)
% Central rectangle of half-widths round(r H / 2), round(r W / 2) around the
% centre (floor(H/2) + 1 in MATLAB indexing), symmetric; all ones at r = 1.
m = ones(H, W, 'like', real(like));
if r >= 0.999, return; end
kh = round(H * r / 2);  kw = round(W * r / 2);
ch = floor(H / 2);      cw = floor(W / 2);
m(:) = 0;
m(max(1, ch - kh + 1):min(H, ch + kh + 1), max(1, cw - kw + 1):min(W, cw + kw + 1)) = 1;
end
