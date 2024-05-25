
%% Performances of detectors under Laplacian noise
% Dayan Adionel Guimarães and Luiz Gustavo Barros Guedes, 2024.

clear variables; clc; % close all;
% rng default; rng(9); % uncomment for repeated realizations of random variables.
% Change the rng(1) to rng(2, etc) to generate a new pattern of realizations. 

%% System parameters
Laplace = 1;         % Gaussian noise (0) or Laplacian noise (1).
m = 4;               % Number of SU receivers. (6)
SNR = -10;           % Average signal-to-noise ratio over all SUs, dB. (-10)
runs = 10000;        % Number of events for computing the empirical CDFs.
eta = 2.5;           % Path-loss exponent.
r = 1;               % Coverage radius, m.
d0 = 0.001*r;        % Reference distance for path-loss calculation, m.
P_txPU = 5;          % PU tx power, W. (5)
xPU = 1*r;           % x-coordinate of the PU tx, m. Equal to y-coordinate
n = 300;             % Number of samples per SU. (250) (300 for the others)
T = n/10;            % Number of samples per QPSK PU symbol (n/T must be integer).
rho = 0.5;           % Fraction of noise power variations about the mean. (0.5)
meanK = 1.88;        % Mean of Rice factor (dB) for variable K over the runs and SUs.
sdK = 4.13;          % Standard deviation (dB) of K over the runs and SUs.
randK = 1;           % If randK = 1, K is random; if randK = 0, K = meanK.
PUsignal = 0;        % PU signal: "0" = iid Gaussian; "1" = niid (T>1) or iid (T=1) QPSK.
Npt = 40;            % Number of points on the ROCs.
Pfa = 0.1;           % Reference Pfa for threshold computation.
NU = 0;              % Enable (1) or disable (0) noise uncertainty for ED, AVC, MED.
psi_f = 0.5;         % FLOM exponent
psi_lp = 4;          % Lp-Norm exponent 

% OBS: For urban area: meanK = 1.88, sdK = 4.13. For rural area: meanK = 2.63, sdK = 3.82. 

Parameter = [1 2 3 4]; Flag = 1; % Vector that controls the selection after ordering
% Parameter = [0.01 0.5 1 1.5 2 2.5 3 3.5 4 4.5 5]; Flag = 2; % Possible values of the exponent p in Norm-based detectors

for loop = 1:length(Parameter)
    disp( [ 'Parameter value: ', num2str(Parameter(loop))]); % Display the parameter varied
    if Flag == 1
%         First set of results (uncomment only if you are using 'Second set
%         of results', 'Third set of results' or 'Fourth set of results'
%         Influence of only one signal, starting by the largest
        k1 = [0 0 0 0];
        if loop == 1
          k1 = [0 0 0 4];
          elseif loop == 2
              k1 = [0 0 3 0];
          elseif loop == 3
              k1 = [0 2 0 0];
          elseif loop == 4
              k1 = [1 0 0 0];
        end

        % Second set of results (uncomment only if you are using 'First set
        % of results', 'Third set of results' or 'Fourth set of results'
        % Influence of choosing the largests (the largest, the two largests and so on)
%         k1 = [0 0 0 0];
%         if loop == 1
%             k1 = [0 0 0 4];
%         elseif loop == 2
%             k1 = [0 0 3 4];
%         elseif loop == 3
%             k1 = [0 2 3 4];
%         elseif loop == 4
%             k1 = [1 2 3 4];
%         end

%         % Third set of results (uncomment only if you are using 'First set
%         % of results', 'Second set of results' or 'Fourth set of results' 
          % Influence of the smallests (the smallest, the two smallests and so on)
%         k1 = [0 0 0 0];
%         if loop == 1
%             k1 = [1 0 0 0];
%         elseif loop == 2
%             k1 = [1 2 0 0];
%         elseif loop == 3
%             k1 = [1 2 3 0];
%         elseif loop == 4
%             k1 = [1 2 3 4];
%         end

%         % Fourth set of results (uncomment only if you are using 'First set
%         % of results', 'Second set of results' or 'Third set of results'
%         % Setting the second smallest value and varying the other one in order to
%         % assess the influence of the choice of the largest value
%         k1 = [0 0 0 0];
%         if loop == 1
%             k1 = [1 2 0 0];
%         elseif loop == 2
%             k1 = [0 2 0 0];
%         elseif loop == 3
%             k1 = [0 2 3 0];
%         elseif loop == 4
%             k1 = [0 2 0 4];
%         end
        disp(k1);
    end
    
    if Flag == 2
        p = loop;
    end
    %% Pre-allocation of variables
    PRx_measured = zeros(runs,1); Pnoise_measured = zeros(runs,1);
    Tnorm_h0 = zeros(runs,1); Tnorm_h1 = zeros(runs,1);
    Ted_h0 = zeros(runs,1); Ted_h1 = zeros(runs,1);
    Tavc_h0 = zeros(runs,1); Tavc_h1 = zeros(runs,1);
    Tflom1_h0 = zeros(runs,1); Tflom1_h1 = zeros(runs,1);
    Tlp1_h0 = zeros(runs,1); Tlp1_h1 = zeros(runs,1);
    Tcoseed1_h0 = zeros(runs,1); Tcoseed1_h1 = zeros(runs,1);
    Tcosavc1_h0 = zeros(runs,1); Tcosavc1_h1 = zeros(runs,1);
    Tcosflom_h0 = zeros(runs,1); Tcosflom_h1 = zeros(runs,1);
    Tcoslp_h0 = zeros(runs,1); Tcoslp_h1 = zeros(runs,1);
    
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

        %% PU signal (nx1):
        if PUsignal==0 % Cplx iid Gaussian PU signal (1xn)
           S = normrnd(0,1/sqrt(2),1,n) + 1j*normrnd(0,1/sqrt(2),1,n); S = (S'*diag(sqrt(P_txPU)))';
        else if PUsignal==1 % QPSK PU signal (1xn) with T samples per symbol
           S = []; for symb = 1:n/T
           S = [S (randi([0,1],1,1)*2-1)*ones(1,T)+1j*(randi([0,1],1,1)*2-1)*ones(1,T)];
            end; S = (S'*diag(sqrt(P_txPU/2)))'; 
        end; end

        %% Measured SNR in each run and loop
        snr(i,loop) = mean(sum(abs(h*S).^2,2)./sum(abs(W0).^2,2)); % Correct
        snr2(i,loop) = mean(sum(abs(h*S).^2,2))/mean(sum(abs(W0).^2,2)); % Incorrect
        snr2(i,loop) = mean(sum(abs(h*S).^2,2))/mean(sum(abs(W0).^2,2))*SNRcorrectionFactor; % Incorrect corrected

        %% Received signal matrices under H0 and H1 (mxn)
        X_h0 = W0; X_h1 = h*S + W1;

%         %% Norm-based statistic:
%         SUM0 = 0; SUM1 = 0; for c = 1:m 
%         if NU == 1
%         Sigma2(c) = Sigma2_avg; % enabled to measure the effect of noise uncertainty
%         end
%         SUM0 = SUM0 + sum(abs(X_h0(c,:)).^p)/((sqrt(Sigma2(c))).^p);
%         SUM1 = SUM1 + sum(abs(X_h1(c,:)).^p)/((sqrt(Sigma2(c))).^p);
%         end;  Tnorm_h0(i) = SUM0; Tnorm_h1(i) = SUM1;

        %% ED (energy detection) statistic:
        SUM0 = 0; SUM1 = 0; for c = 1:m 
        if NU == 1
        Sigma2(c) = Sigma2_avg; % enabled to measure the effect of noise uncertainty
        end
        SUM0 = SUM0 + sum(abs(X_h0(c,:)).^2)/((sqrt(Sigma2(c))).^2);
        SUM1 = SUM1 + sum(abs(X_h1(c,:)).^2)/((sqrt(Sigma2(c))).^2);
        end;  Ted_h0(i) = SUM0; Ted_h1(i) = SUM1;

        %% AVC (Absolute Value Cumulating) statistic (considering only AWGN variance):
        % test statistic for conventional model
        SUM0 = 0; SUM1 = 0;
        for c = 1:m 
            SUM0 = SUM0 + sum(abs(X_h0(c,:)))/(sqrt(Sigma2(c)));
            SUM1 = SUM1 + sum(abs(X_h1(c,:)))/(sqrt(Sigma2(c)));
        end
        Tavc_h0(i) = SUM0; Tavc_h1(i) = SUM1;

        %% FLOM 1 (Fractional Lower Order Moments) statistic:
        SUM0 = 0; SUM1 = 0;
        for c = 1:m 
            SUM0 = SUM0 + sum(abs(X_h0(c,:)).^psi_f)/((sqrt(Sigma2(c))).^psi_f);
            SUM1 = SUM1 + sum(abs(X_h1(c,:)).^psi_f)/((sqrt(Sigma2(c))).^psi_f);
        end
        Tflom1_h0(i) = SUM0; Tflom1_h1(i) = SUM1;

        %% Lp-Norm statistic 1:
        SUM0 = 0; SUM1 = 0;
        for c = 1:m 
            SUM0 = SUM0 + sum(abs(X_h0(c,:)).^psi_lp)/(sqrt(Sigma2(c)).^psi_lp);
            SUM1 = SUM1 + sum(abs(X_h1(c,:)).^psi_lp)/(sqrt(Sigma2(c)).^psi_lp);
        end
        Tlp1_h0(i) = SUM0; Tlp1_h1(i) = SUM1;

        %% COS ED (Combining with Order Statistics Enhanced Energy Detector):
        SUM0 = 0; SUM1 = 0;
        for c = 1:m
            v01(c) = sum(abs(X_h0(c,:)).^2)/(sqrt(Sigma2(c)).^2);
            v11(c) = sum(abs(X_h1(c,:)).^2)/(sqrt(Sigma2(c)).^2);
        end

        v_h01 = sort(v01,'ascend'); v_h11 = sort(v11,'ascend');

        % SUM0 = 0; SUM1 = 0;
        for c = 1:m
            if k1(c) == c
                SUM0 = SUM0+v_h01(c);
                SUM1 = SUM1+v_h11(c);
            end
        end

        Tcoseed1_h0(i) = SUM0; Tcoseed1_h1(i) = SUM1;

        %% COS AVC (Combining with Order Statistics Enhanced Energy Detector):
        SUM0 = 0; SUM1 = 0;
        for c = 1:m
            v03(c) = sum(abs(X_h0(c,:)))/(sqrt(Sigma2(c)));
            v13(c) = sum(abs(X_h1(c,:)))/(sqrt(Sigma2(c)));
        end

        v_h03 = sort(v03,'ascend'); v_h13 = sort(v13,'ascend');

        % SUM0 = 0; SUM1 = 0;
        for c = 1:m
            if k1(c) == c
                SUM0 = SUM0+v_h03(c);
                SUM1 = SUM1+v_h13(c);
            end
        end

        Tcosavc1_h0(i) = SUM0; Tcosavc1_h1(i) = SUM1;

        %% COS FLOM (Combining with Order Statistics FLOM):
        SUM0 = 0; SUM1 = 0;
        for c = 1:m
            v04(c) = sum(abs(X_h0(c,:)).^psi_f)/((sqrt(Sigma2(c))).^psi_f);
            v14(c) = sum(abs(X_h1(c,:)).^psi_f)/((sqrt(Sigma2(c))).^psi_f);
        end

        v_h04 = sort(v04,'ascend'); v_h14 = sort(v14,'ascend');

        % SUM0 = 0; SUM1 = 0;
        for c = 1:m
            if k1(c) == c
                SUM0 = SUM0+v_h04(c);
                SUM1 = SUM1+v_h14(c);
            end
        end

        Tcosflom_h0(i) = SUM0; Tcosflom_h1(i) = SUM1;
        
        %% COS Lp-Norm (Combining with Order Statistics Lp-Norm):
        SUM0 = 0; SUM1 = 0;
        for c = 1:m
            v05(c) = sum(abs(X_h0(c,:)).^psi_lp)/((sqrt(Sigma2(c))).^psi_lp);
            v15(c) = sum(abs(X_h1(c,:)).^psi_lp)/((sqrt(Sigma2(c))).^psi_lp);
        end

        v_h05 = sort(v05,'ascend'); v_h15 = sort(v15,'ascend');

        % SUM0 = 0; SUM1 = 0;
        for c = 1:m
            if k1(c) == c
                SUM0 = SUM0+v_h05(c);
                SUM1 = SUM1+v_h15(c);
            end
        end

        Tcoslp_h0(i) = SUM0; Tcoslp_h1(i) = SUM1;
    end

    %% Empirical CDFs

%     % Norm-based detection
%     T_h0 = Tnorm_h0; T_h1 = Tnorm_h1; 
%     Z = sort(T_h0); Gamma = Z((1-Pfa)*runs);
%         aux_h0 = 0; aux_h1 = 0;
%         for ii=1:runs
%             if T_h1(ii) < Gamma
%                 aux_h1 = aux_h1 + 1;
%             end
%         end
%         CDF_Tnorm_H1(loop) = aux_h1/runs;
        
    % Energy detection (ED)
    T_h0 = Ted_h0; T_h1 = Ted_h1; 
    Z = sort(T_h0); Gamma = Z((1-Pfa)*runs);
        aux_h0 = 0; aux_h1 = 0;
        for ii=1:runs
            if T_h1(ii) < Gamma
                aux_h1 = aux_h1 + 1;
            end
        end
        CDF_Ted_H1(loop) = aux_h1/runs;
        
    % Absolute value cumulating (AVC)
    T_h0 = Tavc_h0; T_h1 = Tavc_h1; 
    Z = sort(T_h0); Gamma = Z((1-Pfa)*runs);
        aux_h0 = 0; aux_h1 = 0;
        for ii=1:runs
            if T_h1(ii) < Gamma
                aux_h1 = aux_h1 + 1;
            end
        end
        CDF_Tavc_H1(loop) = aux_h1/runs;
    
    % Fractional Lower Ordered Moments (FLOM)
    T_h0 = Tflom1_h0; T_h1 = Tflom1_h1; 
    Z = sort(T_h0); Gamma = Z((1-Pfa)*runs);
    aux_h0 = 0; aux_h1 = 0;
        for ii=1:runs
            if T_h1(ii) < Gamma
                aux_h1 = aux_h1 + 1;
            end
        end
    CDF_Tflom1_H1(loop) = aux_h1/runs;    
    
    % Lp-Norm 
    T_h0 = Tlp1_h0; T_h1 = Tlp1_h1; 
    Z = sort(T_h0); Gamma = Z((1-Pfa)*runs);
    aux_h0 = 0; aux_h1 = 0;
        for ii=1:runs
            if T_h1(ii) < Gamma
                aux_h1 = aux_h1 + 1;
            end
        end
    CDF_Tlp1_H1(loop) = aux_h1/runs;
    
    % COS Fractional Lower Ordered Moments (COS FLOM)
    T_h0 = Tcosflom_h0; T_h1 = Tcosflom_h1; 
    Z = sort(T_h0); Gamma = Z((1-Pfa)*runs);
    aux_h0 = 0; aux_h1 = 0;
        for ii=1:runs
            if T_h1(ii) < Gamma
                aux_h1 = aux_h1 + 1;
            end
        end
    CDF_Tcosflom_H1(loop) = aux_h1/runs;
    
    % COS Lp-Norm (COS Lp-Norm)
    T_h0 = Tcoslp_h0; T_h1 = Tcoslp_h1; 
    Z = sort(T_h0); Gamma = Z((1-Pfa)*runs);
    aux_h0 = 0; aux_h1 = 0;
        for ii=1:runs
            if T_h1(ii) < Gamma
                aux_h1 = aux_h1 + 1;
            end
        end
    CDF_Tcoslp_H1(loop) = aux_h1/runs;
    
    % Combining with order statistics energy detector (COS ED)
    T_h0 = Tcoseed1_h0; T_h1 = Tcoseed1_h1; 
    Z = sort(T_h0); Gamma = Z((1-Pfa)*runs);
    aux_h0 = 0; aux_h1 = 0;
        for ii=1:runs
            if T_h1(ii) < Gamma
                aux_h1 = aux_h1 + 1;
            end
        end
    CDF_Tcoseed1_H1(loop) = aux_h1/runs;
    
    % Combining with order statistics absolute value cumulating (COS AVC)
    T_h0 = Tcosavc1_h0; T_h1 = Tcosavc1_h1; 
    Z = sort(T_h0); Gamma = Z((1-Pfa)*runs);
    aux_h0 = 0; aux_h1 = 0;
        for ii=1:runs
            if T_h1(ii) < Gamma
                aux_h1 = aux_h1 + 1;
            end
        end
    CDF_Tcosavc1_H1(loop) = aux_h1/runs;      
end

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
disp( [ 'Conf. interval @ Pd=0.5:   CI = ', num2str(CI)]);
disp( [ 'Reference CFAR:                 Pfa = ', num2str(Pfa) ]);
disp( [ 'Varying parameter values:       ', num2str(Parameter) ]);

%%
figure(1); % set(gcf,'Position',[550 200 300 400]) % main screen
if Flag == 1
    p1 = plot(Parameter,1-CDF_Ted_H1,'k-s','LineWidth',1.5,'MarkerFaceColor','w','MarkerSize',7);  hold on
    p2 = plot(Parameter,1-CDF_Tavc_H1,'r-o','LineWidth',1.5,'MarkerFaceColor','w','MarkerSize',5);
    p3 = plot(Parameter,1-CDF_Tcoseed1_H1,'k--s','LineWidth',1.5,'MarkerFaceColor','w');
    p4 = plot(Parameter,1-CDF_Tcosavc1_H1,'r--o','LineWidth',1.5,'MarkerFaceColor','w');
    p5 = plot(Parameter,1-CDF_Tflom1_H1,'b-^','LineWidth',1.5,'MarkerFaceColor','w');
    p6 = plot(Parameter,1-CDF_Tcosflom_H1,'b--^','LineWidth',1.5,'MarkerFaceColor','w');
    p7 = plot(Parameter,1-CDF_Tlp1_H1,'g-h','LineWidth',1.5,'MarkerFaceColor','w');
    p8 = plot(Parameter,1-CDF_Tcoslp_H1,'g--h','LineWidth',1.5,'MarkerFaceColor','w');
   
    legend([p1,p3,p2,p4,p5,p6,p7,p8],{'ED','COS ED','AVC','COS AVC','FLOM','COS FLOM','L4-Norm', 'COS L4-Norm'},'Location','eastoutside'); 
    xlabel('{\it{n}}-th largest');
    xticks(min(Parameter):1:max(Parameter)); 
end

if Flag==2
    p1 = plot(Parameter,1-CDF_Tnorm_H1,'m-d','LineWidth',1.5,'MarkerFaceColor','w','MarkerSize',7);  hold on
    legend(p1,{'Norm-based detector'},'Location','eastoutside');
    xlabel('p'); 
%     xticks([0.01 0.1 0.5 1 1.5 2 2.5 3 3.5 4]); 
end

yticks(0:0.1:1); grid on;
ylabel('Probability of detection, {\it{P}}_{d}');
axis([min(Parameter) max(Parameter) 0 1]);
hold off