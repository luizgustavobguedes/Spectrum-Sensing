
%% Performances of Inequality Index Detectors Implemented with Direct-Conversion Receivers
% Dayan Adionel Guimarães and Luiz Gustavo Barros Guedes, 2025.

clear variables; clc; % close all;
% rng default; rng(9); % uncomment for repeated realizations of random variables.
% Change the rng(1) to rng(2, etc) to generate a new pattern of realizations. 

%% System parameters
% Laplace = 0;         % Laplacian noise ON (Laplace = 1); Laplacian noise OFF (Laplace = 0).
m = 6;               % Number of SU receivers.
SNR = -10;           % Average signal-to-noise ratio over all SUs, dB.
runs = 10000;         % Number of events for computing the empirical CDFs.
eta = 2.5;           % Path-loss exponent.
r = 1;               % Coverage radius, m.
d0 = 0.001*r;        % Reference distance for path-loss calculation, m.
P_txPU = 5;          % PU tx power, W.
xPU = 1*r;           % x-coordinate of the PU tx, m. Equal to y-coordinate
n = 250;             % Number of samples per SU.
T = n/10;            % Number of samples per QPSK PU symbol (n/T must be integer).
rho = 0.5;           % Fraction of noise power variations about the mean.
meanK = 1.88;        % Mean of Rice factor (dB) for variable K over the runs and SUs.
sdK = 4.13;          % Standard deviation (dB) of K over the runs and SUs.
randK = 1;           % If randK = 1, K is random; if randK = 0, K = meanK.
% OBS: For urban area: meanK = 1.88, sdK = 4.13. For rural area: meanK = 2.63, sdK = 3.82.
PUsignal = 0;        % PU signal: "0" = iid Gaussian; "1" = niid (T>1) or iid (T=1) QPSK.
Npt = 40;            % Number of points on the ROCs.
Pfa = 0.1;           % Reference Pfa for threshold computation.
Sigma2avg = 1;       % Average noise power.
NU = 0;              % Enable (1) or disable (0) noise uncertainty for ED, AVC, MED.
E = 0.5;             % Inequality aversion parameter of the AID.

%% DCR (Direct-Conversion Receiver) parameters
SDCR = 5;               % Signal-to-DC-offset ratio, dB.
N_q = 8;                % Number of quantization levels of the ADCs (Re and Im).
f_od = 1.2;             % ADC overdrive factor (controls the amount of the ADC input signal that exeeds its dynamic range, controlling the clipping effect)
L = n/10;               % Lenght of the impulse reponse of the low-pass receive filter (defaut L=n/10). 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Enable only the set of parameter values to be varied:
% Parameter = [-20,-15,-10,-5,0,5,10,15,20]; SNR = -9; Flag=1; % For varying meanK.
% Parameter = [2,4,6,8,10,12]; SNR = -10.5; Flag=2; % For varying m.
% Parameter = [2,5,10,15,20,25]; SNR = -12; Flag=2; % For varying m.
% Parameter=[1,1.5,2,2.5,3,3.5,4]; SNR = -10; Flag=3; % For varying eta.
% Parameter = [20,100,200,400,600,800,1000]; SNR = -11.5; Flag=4; % For varying n.
% Parameter = [-20,-17.5,-15,-12.5,-10,-7.5,-5,-2.5,0]; Flag=5; % For varying SNR.
% Parameter = [0,0.2,0.4,0.6,0.8,0.9]; Flag=6; % For varying rho.
% Parameter=[0,r,3*r,5*r,10*r,15*r]; SNR=-11; eta=2.5; Flag=7; % For varying xPU.

%% Enable only the set of DCR parameter values to be varied:
Parameter = [0.5,0.8,1.1,1.4,1.7,2,2.3,2.5]; Flag = 8;  % For varying f_od.
% Parameter = linspace(-15,10,8); Flag = 9;                 % For varying SDCR.
% Parameter = [2,4,8,16,32,64]; Flag = 10;                % For varying N_q
% Parameter = round(linspace(1,n/2,8),0); Flag = 11;      % For varying L.

for loop = 1:length(Parameter)
if Flag==1; meanK = Parameter(loop); end
if Flag==2; m = Parameter(loop); end
if Flag==3; eta = Parameter(loop); end
if Flag==4; n = Parameter(loop); T = n/10; end
if Flag==5; SNR = Parameter(loop); end
if Flag==6; rho = Parameter(loop); end
if Flag==7; xPU = Parameter(loop); end
if Flag==8; f_od = Parameter(loop); end
if Flag==9; SDCR = Parameter(loop); end
if Flag==10; N_q = Parameter(loop); end
if Flag==11; L = Parameter(loop); end
disp( [ 'Parameter value: ', num2str(Parameter(loop))]); % Display the parameter varied

% Pre-allocation of variables
PRx_measured = zeros(runs,1); Pnoise_measured = zeros(runs,1);
Tpride_h0 = zeros(runs,1); Tpride_h1 = zeros(runs,1);
Tgid_h0 = zeros(runs,1); Tgid_h1 = zeros(runs,1);
Ttid_h0 = zeros(runs,1); Ttid_h1 = zeros(runs,1);
Taid_h0 = zeros(runs,1); Taid_h1 = zeros(runs,1);

%% DCR configurations

% L = n/10; % Enable this line only if the variable parameter is n.

% Impulse response of the lowpass filter. Default: moving average filter, hi = 1/sqrt(L)
h = ones(1,L)/sqrt(L);  

% Receive filter as a Toeplitz matrix (obs: input with zero-padding)
F_ma = toeplitz([h zeros(1,n-1)],[h(1) zeros(1,n+L-1)]);     % Moving average filter as a Toeplitz matrix

% Whitening Filter
Convolution = real(conv(h,h)/max(conv(h,h)));            % Self-convolution of MA filter impulse response.
IndexMax = find(Convolution==1);                         % Index of maximum convolution
Correlation = Convolution(1,IndexMax:IndexMax+L-1);      % Keep the right part of convolution
qq = [Correlation zeros(1,n-L)];                         % Resize the correlation vector to dimension (1xn)
for i=1:n                                                %   
    for j=1:n                                            %
        Q(i,j) = qq(abs(i-j)+1);                         % Create the matrix Q(nxn) from the correlation vector
    end                                                  %
end                                                      %
[U,S,V] = svd(Q);                                        % Singular value decomposition of matrix Q
C = chol(Q,'lower');                                     % Lower triangular from Cholensky decomposition of Q
W_f = U*C^(-1);                                          % Whitening filter matrix

for i = 1:runs  % Simulation runs

%% SUs coordinates inside a circular area of radius r
rr = sqrt(rand(m,1)); % Generate m random values between 0 and 1, allocated in a vector. Take the square root of each of these values
theta = 2*pi*rand(m,1); % Calculate the perimeter of each SU
SU = [rr.*cos(theta) rr.*sin(theta)]; 
SU = r*SU;

%% Distances from the PU tx to the SUs
d_pu = zeros(m,1); 
for j = 1:m
    d_pu(j) = norm(SU(j,:) - [xPU,xPU]); % Distances from the SUs to the PU at [xPU,xPU]
end

%% SU received signal powers (from PU tx)
P_rxSU = P_txPU*(d0./d_pu).^eta; % P_rxSU_dBm = 10*log10(P_rxSU/0.001);
if rho > 0
SNRcorrectionFactor = (log((1+rho)/(1-rho)))/(2*rho);
Gamma = sum(P_rxSU)*SNRcorrectionFactor/m; % Average SNR for Sigma2_avg = 1, rho > 0.
else
SNRcorrectionFactor = 1;
Gamma = sum(P_rxSU)/m; % Average SNR for Sigma2_avg = 1, rho = 0.   
end

Sigma2_avg = Gamma/(10^(SNR/10)); % Corrected Sigma2_avg
PRxavg = Sigma2_avg*(10^(SNR/10)); %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Noise variances (m x 1) variable over all sensing rounds
U = unifrnd(-1, 1, m, 1); % Uniform RV in [-1,1].
Sigma2 = (1 + rho*U)*Sigma2_avg; % Noise variances across the SUs.

%% Channel vector (mx1):
a = zeros(m,1);
for row = 1:m
    if randK == 1
    K = 10^(randn(1,1)*sdK+meanK)/10; % Variable K
    else
    K = 10^(meanK/10); % Fixed K
    end
% a(row,1) = (normrnd(sqrt(K/(K+1)),sqrt(1/((K+1))),1,1));
a(row,1) = (normrnd(sqrt(K/(2*(K+1))),sqrt((1-K/(K+1))/2),1,1) + 1j*normrnd(sqrt(K/(2*(K+1))),sqrt((1-K/(K+1))/2),1,1));
end

g = sqrt(P_rxSU/P_txPU); % Distance-dependent channel gains
h = a.*g; % Rescaling the channel vector according to the above gains

%% Gaussian noise matrices (mxn):
W0 = zeros(m,n); W1 = zeros(m,n);
for j = 1:m
W0(j,:) = normrnd(0,sqrt(Sigma2(j)/2),1,n) + 1j*normrnd(0,sqrt(Sigma2(j)/2),1,n);
W1(j,:) = normrnd(0,sqrt(Sigma2(j)/2),1,n) + 1j*normrnd(0,sqrt(Sigma2(j)/2),1,n);
end

%% Laplacean noise matrices (mxn):
L0=zeros(m,n); L1=zeros(m,n);
for j = 1:m
    b = sqrt(Sigma2(j)/2);
    L0(j,:) = random('exp',b,1,n)-random('exp',b,1,n);
    L1(j,:) = random('exp',b,1,n)-random('exp',b,1,n);
end

%% Gaussian or Laplacean noise selection
W0 = W0*(1-Laplace) + L0*Laplace; W1 = W1*(1-Laplace) + L1*Laplace;

W0_Fma = [W0 zeros(m,L)]*F_ma.'; % MA filtering
W0_Fma = W0_Fma(1:m,L:n+L-1);    % Removing the first L-1 samples from filtering
W1_Fma = [W1 zeros(m,L)]*F_ma.'; 
W1_Fma = W1_Fma(1:m,L:n+L-1);

%% PU signal (nx1):
if PUsignal==0 % Cplx iid Gaussian PU signal (1xn)
   S = normrnd(0,1/sqrt(2),1,n) + 1j*normrnd(0,1/sqrt(2),1,n); S = (S'*diag(sqrt(P_txPU)))';
else if PUsignal==1 % QPSK PU signal (1xn) with T samples per symbol
   S = []; for symb = 1:n/T
   S = [S (randi([0,1],1,1)*2-1)*ones(1,T)+1j*(randi([0,1],1,1)*2-1)*ones(1,T)];
    end; S = (S'*diag(sqrt(P_txPU/2)))'; 
end; end

HS_Fma = [h*S zeros(m,L)]*F_ma.'; % MA filtering
HS_Fma = HS_Fma(1:m,L:n+L-1); % Removing the first L-1 samples from filtering

%% Measured SNR in each run and loop
snr(i,loop) = mean(sum(abs(h*S).^2,2)./sum(abs(W0).^2,2)); % Correct
snr2(i,loop) = mean(sum(abs(h*S).^2,2))/mean(sum(abs(W0).^2,2)); % Incorrect
snr2(i,loop) = mean(sum(abs(h*S).^2,2))/mean(sum(abs(W0).^2,2))*SNRcorrectionFactor; % Incorrect corrected

%% Matrix D of residual DC-offset
D0 = (normrnd(0,1,m,1) + 1j*normrnd(0,1,m,1)).*ones(m,n);
D1 = (normrnd(0,1,m,1) + 1j*normrnd(0,1,m,1)).*ones(m,n);
D0 = D0/sqrt(norm(D0,'fro')^2 /(m*n)) * sqrt(PRxavg/(10^(SDCR/10)));
D1 = D1/sqrt(norm(D1,'fro')^2 /(m*n)) * sqrt(PRxavg/(10^(SDCR/10)));

%% Signal, noise and DC-offset power measurements in each run
PRx_measured(i) = sum(sum(abs(HS_Fma).^2))/(m*n);
Pnoise_measured(i) = sum(sum(abs(W0_Fma).^2))/(m*n);
Pdc_measured(i) = sum(sum(abs(D0).^2))/(m*n);

%% Received signal matrices under H0 and H1 (mxn)
X_h0 = W0; X_h1 = h*S + W1;

%% DCR received signal matrices under H0 and H1 (mxn):
X_h0_Fma = W0_Fma + D0;                                     
X_h1_Fma = HS_Fma + W1_Fma + D1;

%% AGC and whitening (modified sequence of operations)
for j=1:m                                                                           
    Gains_0(j) = f_od*sqrt(2*n)/(6*norm(X_h0_Fma(j,1:n)));     
    Gains_1(j) = f_od*sqrt(2*n)/(6*norm(X_h1_Fma(j,1:n)));      
end    
X_h0_Fma = diag(Gains_0) * X_h0_Fma;                                                                                  
X_h1_Fma = diag(Gains_1) * X_h1_Fma;                           
X_h0_in = (W_f * X_h0_Fma.').';  % Whitened samples for H0                                   
X_h1_in = (W_f * X_h1_Fma.').';  % Whitened samples for H1
                  
%% Quantization with clipping
q = 2/(N_q - 1);                                      % Spacing between quatization levels
X_h0_in_real = min(max(real(X_h0_in)/(1/2),-1),1);    % Real fraction of signal matrix under H0 (Clipped)
X_h0_in_imag = min(max(imag(X_h0_in)/(1/2),-1),1);    % Imaginary fraction of signal matrix under H0 (Clipped)
X_h1_in_real = min(max(real(X_h1_in)/(1/2),-1),1);    % Real fraction of signal matrix under H1 (Clipped)
X_h1_in_imag = min(max(imag(X_h1_in)/(1/2),-1),1);    % Imaginary fraction of signal matrix under H1 (Clipped) 
X_h0_real_q = round(X_h0_in_real/q)*q - sign(abs(X_h0_in_real-(round(X_h0_in_real/q)*q + q/2))- q/2)*q/2;
X_h0_imag_q = round(X_h0_in_imag/q)*q - sign(abs(X_h0_in_imag-(round(X_h0_in_imag/q)*q + q/2))- q/2)*q/2;
X_h1_real_q = round(X_h1_in_real/q)*q - sign(abs(X_h1_in_real-(round(X_h1_in_real/q)*q + q/2))- q/2)*q/2;
X_h1_imag_q = round(X_h1_in_imag/q)*q - sign(abs(X_h1_in_imag-(round(X_h1_in_imag/q)*q + q/2))- q/2)*q/2;
X_h0_q = (1/2)*X_h0_real_q + (1/2)*1j*X_h0_imag_q;    % Quantized signal matrix with N_q levels between -1,+1                                      
X_h1_q = (1/2)*X_h1_real_q + (1/2)*1j*X_h1_imag_q;    % Quantized signal matrix with N_q levels between -1,+1

%% Received signal sample covariance matrice (SCM)
R_h0 = X_h0*X_h0'/n; R_h1 = X_h1*X_h1'/n; 

% Received signal sample covariance matrix for DCR model
R_h0_p = X_h0_q*X_h0_q'/n;                                                          
R_h1_p = X_h1_q*X_h1_q'/n;

%% GID (Gini index detector) statistic
x_h0 = R_h0(:); x_h1 = R_h1(:); 
Num_h0=0; for u=1:m^2; for j=1:m^2 % Faster with j=u:m^2 instead of j=1:m^2 (sum of (m^4 + m^2)/2 terms, instead of m^4)
Num_h0 = Num_h0 + abs(x_h0(u)-x_h0(j));
end; end; Tgid_h0(i) = (2*(m^2-m))*sum(abs(x_h0))/Num_h0;
Num_h1=0; for u=1:m^2; for j=1:m^2 % Faster with j=u:m^2 instead of j=1:m^2 (sum of (m^4 + m^2)/2 terms, instead of m^4)
Num_h1 = Num_h1 + abs(x_h1(u)-x_h1(j));
end; end; Tgid_h1(i) = (2*(m^2-m))*sum(abs(x_h1))/Num_h1;

% GID test statistic for DCR model 
x_h0_p = R_h0_p(:); x_h1_p = R_h1_p(:); 
Num_h0=0;
for u=1:m^2
    for j=u:m^2 
        Num_h0 = Num_h0 + abs(x_h0_p(u)-x_h0_p(j));
    end
end
Tgid_h0_p(i) = (1*(m^2-m))*sum(abs(x_h0_p))/Num_h0;
Num_h1=0;
for u=1:m^2
    for j=u:m^2
        Num_h1 = Num_h1 + abs(x_h1_p(u)-x_h1_p(j));
    end
end
Tgid_h1_p(i) = (1*(m^2-m))*sum(abs(x_h1_p))/Num_h1;

%% PRIDe (Pietra-Ricci index detector) statistic
% https://en.wikipedia.org/wiki/Hoover_index
x_h0 = R_h0(:); x_h1 = R_h1(:); 
m0 = mean(x_h0); m1 = mean(x_h1);
Tpride_h0(i) = sum(abs(x_h0))/sum(abs(x_h0 - m0));
Tpride_h1(i) = sum(abs(x_h1))/sum(abs(x_h1 - m1));

% PRIDe test statistic for DCR model  
Tpride_h0_p(i) = sum(abs(x_h0_p))/sum(abs(x_h0_p - mean(x_h0_p)));
Tpride_h1_p(i) = sum(abs(x_h1_p))/sum(abs(x_h1_p - mean(x_h1_p)));

%% TID (Theil index detector) statistic
aR_h0 = abs(R_h0); aR_h1 = abs(R_h1);
m0 = mean(mean(aR_h0)); m1 = mean(mean(aR_h1));
SUM0=0; SUM1=0;
for ROW=1:m
    for COL=ROW:m
    I=COL==ROW;
    SUM0 = SUM0 + (2-I)*aR_h0(ROW,COL)*log(aR_h0(ROW,COL)/m0);
    SUM1 = SUM1 + (2-I)*aR_h1(ROW,COL)*log(aR_h1(ROW,COL)/m1);
    end
end
Ttid_h0(i) = m0/SUM0;
Ttid_h1(i) = m1/SUM1;

% TID test statistic for DCR model
aR_h0_p = abs(R_h0_p); aR_h1_p = abs(R_h1_p);
m0_p = mean(mean(aR_h0_p)); m1_p = mean(mean(aR_h1_p));
SUM0=0; SUM1=0;
for ROW=1:m
    for COL=ROW:m
    I=COL==ROW;
    SUM0 = SUM0 + (2-I)*aR_h0_p(ROW,COL)*log(aR_h0_p(ROW,COL)/m0_p);
    SUM1 = SUM1 + (2-I)*aR_h1_p(ROW,COL)*log(aR_h1_p(ROW,COL)/m1_p);
    end
end
Ttid_h0_p(i) = m0/SUM0;
Ttid_h1_p(i) = m1/SUM1;

%% AID (Atkinson index detector) statistic
m0 = mean(mean(R_h0)); m1 = mean(mean(R_h1));
if E==0.5
SUM0=0; SUM1=0;
for ROW=1:m
    for COL=ROW:m
    I=COL==ROW;
    SUM0 = SUM0 + (2-I)*sqrt(abs(R_h0(ROW,COL))+real(R_h0(ROW,COL)));
    SUM1 = SUM1 + (2-I)*sqrt(abs(R_h1(ROW,COL))+real(R_h1(ROW,COL)));
    end
end
Taid_h0(i) = (SUM0^2)/m0; Taid_h1(i) = (SUM1^2)/m1;
else
Taid_h0(i) = ((sum(sum(R_h0.^(1-E)))).^(1/(1-E)))/m0; % Corrected on Nov/2024
Taid_h1(i) = ((sum(sum(R_h1.^(1-E)))).^(1/(1-E)))/m1; % Corrected on Nov/2024
end

% AID test statistic for DCR model
m0_p = mean(mean(R_h0_p)); m1_p = mean(mean(R_h1_p));
if E==0.5
SUM0=0; SUM1=0;
for ROW=1:m
    for COL=ROW:m
    I=COL==ROW;
    SUM0 = SUM0 + (2-I)*sqrt(abs(R_h0_p(ROW,COL))+real(R_h0_p(ROW,COL)));
    SUM1 = SUM1 + (2-I)*sqrt(abs(R_h1_p(ROW,COL))+real(R_h1_p(ROW,COL)));
    end
end
Taid_h0_p(i) = (SUM0^2)/m0_p; Taid_h1_p(i) = (SUM1^2)/m1_p;
else
Taid_h0_p(i) = ((sum(sum(R_h0_p.^(1-E)))).^(1/(1-E)))/m0_p; % Corrected on Nov/2024
Taid_h1_p(i) = ((sum(sum(R_h1_p.^(1-E)))).^(1/(1-E)))/m1_p; % Corrected on Nov/2024
end

end

%% Empirical Pd for convencional receiver

Th0=Tgid_h0; Th1=Tgid_h1;
Z=sort(Th0); Gamma = Z((1-Pfa)*runs);
aux_h0 = 0; aux_h1 = 0;
    for ii=1:runs
        if Th1(ii)<Gamma
            aux_h1 = aux_h1+1;
        end
    end
CDF_Tgid_h1(loop) = aux_h1/runs;

Th0=Tpride_h0; Th1=Tpride_h1;
Z=sort(Th0); Gamma = Z((1-Pfa)*runs);
aux_h0 = 0; aux_h1 = 0;
    for ii=1:runs
        if Th1(ii)<Gamma
            aux_h1 = aux_h1+1;
        end
    end
CDF_Tpride_h1(loop) = aux_h1/runs;

Th0=Ttid_h0; Th1=Ttid_h1;
Z=sort(Th0); Gamma = Z((1-Pfa)*runs);
aux_h0 = 0; aux_h1 = 0;
    for ii=1:runs
        if Th1(ii)<Gamma
            aux_h1 = aux_h1+1;
        end
    end
CDF_Ttid_h1(loop) = aux_h1/runs;

Th0=Taid_h0; Th1=Taid_h1;
Z=sort(Th0); Gamma = Z((1-Pfa)*runs);
aux_h0 = 0; aux_h1 = 0;
    for ii=1:runs
        if Th1(ii)<Gamma
            aux_h1 = aux_h1+1;
        end
    end
CDF_Taid_h1(loop) = aux_h1/runs;

%% Empirical Pd for DCR model
Th0=Tgid_h0_p; Th1=Tgid_h1_p;
Z=sort(Th0); lambda = Z((1-Pfa)*runs);
aux_h0 = 0; aux_h1 = 0;
    for ii=1:runs
        if Th1(ii)<lambda
            aux_h1 = aux_h1+1;
        end
    end
CDF_Tgid_h1_p(loop) = aux_h1/runs;

Th0=Tpride_h0_p; Th1=Tpride_h1_p;
Z=sort(Th0); lambda = Z((1-Pfa)*runs);
aux_h0 = 0; aux_h1 = 0;
    for ii=1:runs
        if Th1(ii)<lambda
            aux_h1 = aux_h1+1;
        end
    end
CDF_Tpride_h1_p(loop) = aux_h1/runs;

Th0=Ttid_h0_p; Th1=Ttid_h1_p;
Z=sort(Th0); lambda = Z((1-Pfa)*runs);
aux_h0 = 0; aux_h1 = 0;
    for ii=1:runs
        if Th1(ii)<lambda
            aux_h1 = aux_h1+1;
        end
    end
CDF_Ttid_h1_p(loop) = aux_h1/runs;

Th0=Taid_h0_p; Th1=Taid_h1_p;
Z=sort(Th0); lambda = Z((1-Pfa)*runs);
aux_h0 = 0; aux_h1 = 0;
    for ii=1:runs
        if Th1(ii)<lambda
            aux_h1 = aux_h1+1;
        end
    end
CDF_Taid_h1_p(loop) = aux_h1/runs;

end % end of parameters' loop


%% Summary of main parameters
if Laplace==1
    str="Laplacian";
else
    str="Gaussian";
end
alpha = 0.05; % confidence level
[Pd_hat,CI] = binofit(runs/2,runs,alpha); % (1-alpha)% Confidence interval

disp( [ 'Number of SUs:                  m = ', num2str(m) ]);
disp( [ 'Number of samples per SU:       n = ', num2str(n) ]);
disp( [ 'Configured SNR in dB:           SNR = ', num2str(SNR) ]);
disp( [ 'Measured SNR in dB:             SNR = ', num2str(10*log10(mean(snr)))]);
disp( [ 'Number of sensing rounds:       ', num2str(runs)]);
disp( [ 'Coverage radius, r:             ', num2str(r),' m']);
disp( [ 'PU tx coordinates:              ', '(',num2str(xPU),',',num2str(xPU),')',' m']);
disp( [ 'Path-loss exponent:             ', num2str(eta)]);
disp( [ 'Random K (1=yes, 0=no):         ', num2str(randK) ]);
disp( [ 'Rice factor K, dB: mean, stdev = ', num2str(meanK),', ', num2str(sdK) ]);
disp( [ 'Noise variation fraction:       ', num2str(rho)]);
disp( [ 'PU signal (0=Gaussian,1=QPSK):  ', num2str(PUsignal) ]);
disp( [ 'Number of Monte Carlo runs:     ', num2str(runs) ]);
disp( [ 'Noise type:                     ', num2str(str)]);
disp( [ 'ADC quantization levels:        Nq = ', num2str(N_q) ]);
disp( [ 'ADC overdrive factor:           fod = ', num2str(f_od) ]);
disp( [ 'Lenght of the MA filter:        L = ', num2str(L) ]);
disp( [ 'Configured SDCR, dB:            SDCR = ', num2str(SDCR) ]);
% disp( [ 'Measured SDCR, dB:              SDCR = ', num2str(10*log10(mean(PRx_measured/Pdc_measured)))]);
disp( [ 'Conf. interval @ Pd=0.5:   CI = ', num2str(CI)]);
disp( [ 'Reference CFAR:                 Pfa = ', num2str(Pfa) ]);
disp( [ 'Varying parameter values:       ', num2str(Parameter) ]);

%%
figure(1); % set(gcf,'Position',[550 200 300 400]) % main screen
Pos1 = [0.085 0.1 0.425 0.8];
subplot('Position',Pos1)
p1 = plot(Parameter,1-CDF_Tgid_h1,'k-s','LineWidth',1.5,'MarkerFaceColor','w','MarkerSize',7);  hold on
p2 = plot(Parameter,1-CDF_Tpride_h1,'r-d','LineWidth',1.5,'MarkerFaceColor','w','MarkerSize',5);
p3 = plot(Parameter,1-CDF_Ttid_h1,'k->','LineWidth',1.5,'MarkerFaceColor','w');
p4 = plot(Parameter,1-CDF_Taid_h1,'k-o','LineWidth',1.5,'MarkerFaceColor','g');
legend([p1,p2,p3,p4],{'GID','PRIDe','TID','AID'},'Location','southeast'); yticks(0:0.1:1); grid on;
ylabel('Probabilidade de detecção, {\it{P}}_{d}');
axis([min(Parameter) max(Parameter) 0 1]);
%axis square
if Flag==1; xlabel('Média do fator de Rice, \mu_{\it{K}} , em dB'); end
if Flag==2; xlabel('Número de SUs, {\it{m}}'); end
if Flag==3; xlabel('Expoente de perdas, \eta'); end
if Flag==4; xlabel('Número de amostras, {\it{n}}'); axis([0 max(Parameter) 0 1]); xticks(0:200:max(Parameter)); end
if Flag==5; xlabel('SNR em dB'); end
if Flag==6; xlabel('Fração \rho'); axis([0 1 0 1]); xticks(0:0.2:1); end
if Flag==7; xlabel('Coordenada x=y do tx PU, em m'); end
uistack(p1,'top')
uistack(p3,'top')
hold off

Pos2 = [0.56 0.1 0.425 0.8];
subplot('Position',Pos2)
p1 = plot(Parameter,1-CDF_Tgid_h1_p,'k-s','LineWidth',1.5,'MarkerFaceColor','w','MarkerSize',7);  hold on
p2 = plot(Parameter,1-CDF_Tpride_h1_p,'r-d','LineWidth',1.5,'MarkerFaceColor','w','MarkerSize',5);
p3 = plot(Parameter,1-CDF_Ttid_h1_p,'k->','LineWidth',1.5,'MarkerFaceColor','w');
p4 = plot(Parameter,1-CDF_Taid_h1_p,'k-o','LineWidth',1.5,'MarkerFaceColor','g');
% legend([p1,p2,p3,p4],{'GID','PRIDe','TID','AID'},'Location','eastoutside'); 
yticks(0:0.1:1); grid on;
% ylabel('Probabilidade de detecção, {\it{P}}_{d}');
axis([min(Parameter) max(Parameter) 0 1]);
%axis square
if Flag==1; xlabel('Média do fator de Rice, \mu_{\it{K}} , em dB'); end
if Flag==2; xlabel('Número de SUs, {\it{m}}'); end
if Flag==3; xlabel('Expoente de perdas, \eta'); end
if Flag==4; xlabel('Número de amostras, {\it{n}}'); axis([0 max(Parameter) 0 1]); xticks(0:200:max(Parameter)); end
if Flag==5; xlabel('SNR em dB'); end
if Flag==6; xlabel('Fração \rho'); axis([0 1 0 1]); xticks(0:0.2:1); end
if Flag==7; xlabel('Coordenada x=y do tx PU, em m'); end
uistack(p1,'top')
uistack(p3,'top')
hold off


