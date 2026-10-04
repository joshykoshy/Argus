function [x_low, x_high] = decompose_slice(x, d0)
% DECOMPOSE_SLICE Reference MATLAB implementation of Gaussian frequency decomposition.
%
% Inputs:
%   x    - 2D slice [H x W], single/double precision
%   d0   - Normalized radial cutoff frequency [0 < d0 <= 1.0], default 0.20
%
% Outputs:
%   x_low  - Low-pass band [H x W]
%   x_high - High-pass band [H x W]

if nargin < 2 || isempty(d0)
    d0 = 0.20;
end

[H, W] = size(x);
ch = H / 2;
cw = W / 2;

% Meshgrid matching zero-indexed center (0 to H-1) - ch
[V, U] = meshgrid((0:W-1) - cw, (0:H-1) - ch);
D = sqrt((U / ch).^2 + (V / cw).^2);

% Complementary Gaussian filters
H_lp = exp(-(D.^2) / (2 * (d0^2)));
H_hp = 1.0 - H_lp;

% 2D FFT shifted
F = fftshift(fft2(x));

% Filtering & Inverse FFT
F_lp = F .* H_lp;
F_hp = F .* H_hp;

x_low  = real(ifft2(ifftshift(F_lp)));
x_high = real(ifft2(ifftshift(F_hp)));

end
