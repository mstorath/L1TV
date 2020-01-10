%%% Demo for denoising a circle-valued signal by the L1-TV model

% create random signal (smoothed pcw constant signal)
rng(12345) % random seed for reproducibility
N = 2000;
t = 2*pi;
lambda = 20 / N;
K = 20;
sigma= 0.3;
innovation = randCP((rand([N, 1])-0.5) * t, lambda );
signalUnwrapped = cumsum(innovation);
h = fspecial('Gaussian', [N/10, 1], 10);
smoothed = conv(signalUnwrapped, h, 'same');
groundTruth = wrapAngle(smoothed);

% add noise
y = wrapAngle(groundTruth + sigma* randl(size(groundTruth)));

% perform restoration using L1TV_Circ
alpha = sqrt(N)*sigma;
x = L1TV_Circ(y, alpha);

% plot the results
figure('Color', 'w')
subplot(1,2,1)
plot(y, '.')
ylim([-pi,pi])
title('Data with values on the unit circle')

subplot(1,2,2)
plot(x, '.')
ylim([-pi,pi])
title(sprintf('L1TV restoration, SNR improvement: %.2f dB', deltaSNR(groundTruth, y, x, 'circ')))



