clear; clc; close all;

%% Parameters
N   = 1000;
P_N = 1;                          % normalized noise power

SNRdB = -120:10:10;
SNR   = 10.^(SNRdB/10);

p = [0.01 0.05 0.1 0.25 0.5 1 2];

runs = 10000;
Pfa_target = 0.1;

h = 1;  % AWGN

%% ===== Threshold estimation under H0 (paper step-by-step) =====
TH0 = zeros(runs, length(p));

for k = 1:runs
    v  = sqrt(P_N/2) * (randn(N,1) + 1j*randn(N,1));
    a0 = abs(v);
    for i = 1:length(p)
        TH0(k,i) = mean(a0.^p(i));
    end
end

Z = sort(TH0, 1, 'ascend');
idx = ceil((1 - Pfa_target) * runs);
thresholds = Z(idx, :);     % 1 x length(p)

%% ===== Performance vs SNR =====
Pd_hat  = zeros(length(SNRdB), length(p));
Pfa_hat = zeros(length(SNRdB), length(p));  % sanity check (should be ~0.1)

for sidx = 1:length(SNRdB)
    P = SNR(sidx) * P_N;  % scalar signal power for this SNR

    T0 = zeros(runs, length(p));
    T1 = zeros(runs, length(p));

    for run = 1:runs
        x = sqrt(P/2)   * (randn(N,1) + 1j*randn(N,1));
        v = sqrt(P_N/2) * (randn(N,1) + 1j*randn(N,1));

        y0 = v;
        y1 = h*x + v;

        a0 = abs(y0);
        a1 = abs(y1);

        for i = 1:length(p)
            T0(run,i) = mean(a0.^p(i));
            T1(run,i) = mean(a1.^p(i));
        end
    end

    Pfa_hat(sidx,:) = mean(T0 > thresholds, 1);
    Pd_hat(sidx,:)  = mean(T1 > thresholds, 1);
end
