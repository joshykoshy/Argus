function [x_deg] = degrade_slice(x, mu_brain, snr, r, n_real, n_imag)
% DEGRADE_SLICE Reference MATLAB implementation of low-field MRI degradation.
%
% Inputs:
%   x        - 2D clean slice [H x W], single/double precision
%   mu_brain - Mean clean brain intensity (scalar)
%   snr      - Target Signal-to-Noise Ratio (scalar, e.g. 30, 20, 12, 8), or [] for clean
%   r        - k-space resolution retention fraction [0 < r <= 1.0]
%   n_real   - Optional pre-generated standard normal real noise array [H x W]
%   n_imag   - Optional pre-generated standard normal imag noise array [H x W]
%
% Output:
%   x_deg    - Degraded magnitude slice [H x W]

[H, W] = size(x);

% 1. Resolution loss via pseudo k-space truncation
if r < 0.999
    % Symmetric center mask preserving Hermitian symmetry
    kh_rad = round(H * r / 2.0);
    kw_rad = round(W * r / 2.0);
    
    ch = floor(H / 2);
    cw = floor(W / 2);
    
    r_start = max(1, ch - kh_rad + 1);
    r_end   = min(H, ch + kh_rad + 1);
    c_start = max(1, cw - kw_rad + 1);
    c_end   = min(W, cw + kw_rad + 1);
    
    mask = zeros(H, W, 'like', x);
    mask(r_start:r_end, c_start:c_end) = 1.0;
    
    K = fftshift(fft2(x));
    K_trunc = K .* mask;
    x_r = real(ifft2(ifftshift(K_trunc)));
else
    x_r = x;
end

% 2. Complex Rician noise addition
if isempty(snr) || snr <= 0
    x_deg = x_r;
    return;
end

sigma = mu_brain / snr;

if nargin >= 6 && ~isempty(n_real) && ~isempty(n_imag)
    nr = n_real * sigma;
    ni = n_imag * sigma;
else
    nr = randn(H, W, 'like', x) * sigma;
    ni = randn(H, W, 'like', x) * sigma;
end

x_deg = sqrt((x_r + nr).^2 + ni.^2);

end
