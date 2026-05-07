%%% Demo for denoising a real-valued signal by the L1-TV model

% create random signal (smoothed pcw constant signal)
rng(12345) % random seed for reproducibility
N = 2000;
lambda = 20 / N;
scale = 100; % scale of signal (change to observe contrast invariance of L1TV)
sigma= scale* 0.3;
innovation =  randCP(randn([N, 1]), lambda );
signal = scale * cumsum(innovation);
% Gaussian smoothing kernel (no Image Processing Toolbox required)
sig_h = 10;
halfwin = floor((N/10 - 1)/2);
t_h = (-halfwin:halfwin).';
h = exp(-t_h.^2 / (2*sig_h^2));
h = h / sum(h);
groundTruth = conv(signal, h, 'same');

% add noise
y = groundTruth + sigma * randl(size(groundTruth));

% perform restoration using L1TV_Real
alpha = sqrt(N)*sigma/scale;
x = L1TV_Real(y, alpha);

% plot the results
figure('Color', 'w')
subplot(1,2,1)
plot(y, '.')
ylim_set = ylim;
title('Real valued data')

subplot(1,2,2)
plot(x, '.')
ylim(ylim_set);
title(sprintf('L1TV restoration, SNR improvement: %.2f dB', deltaSNR(groundTruth, y, x, 'real')))

