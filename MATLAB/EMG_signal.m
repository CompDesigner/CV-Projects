% Freq & time parameters
Fs = 1000;           % Sampling frequency (Hz)
t = 0:1/Fs:10-1/Fs;  % Time vector from 0 to 10 seconds
N = length(t);       % Number of samples

% Parameters (realistic sim)
nMU = 250; % number of motor units (biceps, triceps)
base_fr = 8;
peak_fr = 30; 
activation = 0.6; % activation of motor units (0.2 low-thresh units & 1.0 all)
thresh = sort(rand(1, nMU).^0.3); % threshold
recruited = find(activation >= thresh); %  MU recruited off of threshold
recruit_n = numel(recruited);

% Dynamic Contraction
% activation_time = 0.2 + 0.8*sin(2*pi*0.1*t); % oscillating activation
% recruitment_time = arrayfun(@(a) find(a >= thresh), activation_time, 'UniformOutput', false);

muap_dur = 10; %ms
muap_len = round(muap_dur/1000*Fs); % biphasic waveform
tau = (muap_len)/6;
x = linspace(-3,3,muap_len); % x = symmetric timing (time axis)
muap_base = -x .* exp(-x.^2/2);

% Prepare MU (vary amplitude & slight shape)
MUAPs = zeros(nMU, muap_len);

% Added parameters for direction/fatigue
% fatigue_level_global = 0.3; 
% direction = "pull"; 

% Define agonist vs antagonist recruitment
% n_ag = round(0.7 * recruit_n);  % 70% of recruited MUs as agonist
% n_ant = recruit_n - n_ag;       % rest are antagonist
% agonist_MUs = 1:n_ag;
% antagonist_MUs = (n_ag+1):recruit_n;

for m = 1:nMU
    scale_ftr = 0.5 + 2*rand();     % amp scaling factor (add activation)
    width_var = 0.9 + 0.2*randn();  % slight width variation (shape)
    x2 = x * width_var;
    muap = -x2 .* exp(-x2.^2/2);
    MUAPs(m,:) = muap / max(abs(muap)) * scale_ftr;

    % Can create subsets for push/pull (agonist/antagonist)
    % if direction == "pull"
    % MUAPs(m,:) = -MUAPs(m,:);  % invert waveform for pull
    % end

    % Fatigue
    % fatigue_level = fatigue_level_global * rand();
    % MUAPs(m,:) = MUAPs(m,:) * (1 - fatigue_level);
end

% Gamma-ISI generator
% gamma-ISI parameters
T = N / Fs;        % total sim time (s)
k_shape = 4;       % gamma shape parameter (CV ~ 0.5) (lower = stress)
min_isi = 0.002;   % optional abs ref in s (2 ms)

spikes_gamma = zeros(nMU, N);  %  gamma spike train

for m = 1:nMU
    if m <= recruit_n
        % Direction based
        % if strcmp(direction, "push") && ~ismember(m, agonist_MUs)
        %     spikes_gamma(m,:) = 0; continue;
        % elseif strcmp(direction, "pull") && ~ismember(m, antagonist_MUs)
        %     spikes_gamma(m,:) = 0; continue;
        % end

        ord = m / nMU;
        fr = base_fr + (peak_fr - base_fr) * (activation * (1 - 0.5*ord));

        mean_ISI = 1 / fr;        % s
        theta = mean_ISI / k_shape;  

        % generate spike times
        t_spk = 0;
        spike_times = [];
        while t_spk < T
            
            % Dynamic Contraction
            % t_idx = min(round(t_spk*Fs)+1, N); % idx in activation_time
            % act = activation_time(t_idx);

            isi = gamrnd(k_shape, theta);  % draw gamma ISI
            isi = max(isi, min_isi);       % enforce abs ref
            t_spk = t_spk + isi;
            if t_spk < T
                spike_times(end+1) = t_spk; %#ok<SAGROW>
            end
        end

        % Convert spike times to binary vector
        idxs = round(spike_times * Fs) + 1;
        idxs(idxs < 1) = []; idxs(idxs > N) = [];
        spikes_gamma(m, idxs) = 1;

    else
        spikes_gamma(m, :) = 0;  % not recruited
    end
end

% Spike trains (Bernoulli per-sample approx Poisson)
% spikes = zeros(nMU, N);
% 
% for m = 1:nMU
%     % Determines firing rate for MU (recruitment order)
%     % lower-thresh MUs fire at lr and recruit earlier
%     ord = m / nMU; % normalizes motor unit idx between 0 & 1
%     if m <= recruit_n
%         % rate increases with activation and MU order
%         fr = base_fr + (peak_fr-base_fr) * (activation * (1 - 0.5*ord));
%         % include some jitter in ISI by using Bernoulli
%         p = fr / Fs;
%         spikes(m, :) = rand(1, N) < p; 
%         % add short refractory period
%         ref_ms = 20; 
%         ref_samples = round(ref_ms/1000*Fs);
%         last = -inf;
%         idxs = find(spikes(m,:));
%         for id = idxs
%             if id - last < ref_samples
%                 spikes(m, id) = 0; 
%             else
%                 last = id;
%             end
%         end
%     else
%         spikes(m, :) = 0; % not recruited
%     end
% end

% Add into Bernoulli
 % last = -inf;
 %        ref_ms = 20; 
 %        ref_samples = round(ref_ms/1000*Fs);
 % 
 %        for n = 1:N
 %            act = activation_time(n);  % activation at this time sample
 % 
 %            % MU recruited at this activation?
 %            if act >= thresh(m)
 %                fr = base_fr + (peak_fr - base_fr) * (act * (1 - 0.5*ord));
 %                p = fr / Fs;
 %                spike = rand() < p;
 % 
 %                % enforce refractory period
 %                if spike && (n - last >= ref_samples)
 %                    spikes(m, n) = 1;
 %                    last = n;
 %                else
 %                    spikes(m, n) = 0;
 %                end
 %            else
 %                spikes(m, n) = 0;
 %            end
 %        end

% Convolve spikes with MUAPs and sum
% clean_emg = zeros(1, N + muap_len);
% for m = 1:nMU
%     clean_emg = clean_emg + conv(spikes(m,:), MUAPs(m,:));
% end
% clean_emg = clean_emg(1:N);  % trim

% gamma-ISI
clean_emg_gamma = zeros(1, N + muap_len-1);
for m = 1:nMU
    clean_emg_gamma = clean_emg_gamma + conv(spikes_gamma(m,:), MUAPs(m,:));
end
clean_emg_gamma = clean_emg_gamma(1:N);

% Add baseline noise (Gaussian) and optional low-freq motion artifact
baseline_noise = 0.05;
noisy_emg = clean_emg_gamma + baseline_noise*randn(1,N);

% Add mains hum (60 Hz) and optional 2nd harmonic
% hum_amp = 0.02;
% noisy_emg = noisy_emg + hum_amp*sin(2*pi*60*t) + 0.005*sin(2*pi*120*t);

% Generate random white noise to represent MUAP interference pattern
% Motor Unit Action Potiental (MUAP)
% noise_muap = randn(1, N);

% Band-pass filter on raw signal
hp = 20; % remove low frequency motion 
lp = 450; % remove high frequency noise
[b_bp, a_bp] = butter(4, [hp lp]/(Fs/2), 'bandpass'); % 4th order Butterworth filter
raw_bp = filtfilt(b_bp, a_bp, noisy_emg); % filtering

% Add 60 Hz for mains hum
% [bn, an] = iirnotch(60/(Fs/2), 60/(Fs/2)/35); % Q ~ 35
% raw_bp = filtfilt(bn, an, raw_bp);

% Simulate muscle contraction (rectification and smoothing)
%  Rectify the signal (absolute value) to simulate a muscle contracting,
%  as EMG has a non-negative envelope.
rec_emg = abs(raw_bp);

% Low-pass filter to simulate the filtering effect of tissue
%  and the smoothing of muscle contraction activity.
cutoff_freq = 5; % Hz (2-10 Hz)
[b, a] = butter(4, cutoff_freq / (Fs/2), 'low'); % 4th order Butterworth filter

% Apply the filter to the rectified signal
sim_emg = filtfilt(b, a, rec_emg);

% Scale the signal to a more realistic amplitude range
emg_bp_uV = raw_bp * 1000;    % if MUAP scale assumed mV -> µV adjustment
env_uV = sim_emg * 1000;

% Plot the raw bp
figure;
plot(t, emg_bp_uV);
title('Raw Band-pass');
xlabel('Time (s)');
ylabel('Amplitude (\muV)');
grid on;

% PLot the rectified EMG signal
figure;
plot(t, rec_emg*1000);
title('Rectified Normal EMG Signal');
xlabel('Time (s)');
ylabel('Amplitude (\muV)');
grid on;

% Plot the simulated EMG signal
figure;
plot(t, env_uV);
title('Simulated Normal EMG Signal');
xlabel('Time (s)');
ylabel('Amplitude (\muV)');
grid on;

% Plot gamma spike (raster plot)
% figure;
% hold on;
% for m = 1:nMU
%     spike_times = find(spikes_gamma(m,:))/Fs; % in seconds
%     y = m*ones(size(spike_times));
%     plot(spike_times, y, 'k.', 'MarkerSize', 5);
% end
% xlabel('Time (s)');
% ylabel('Motor Unit #');
% title('Spike Raster (Gamma ISI)');
% grid on;
% hold off;

% FFT
% L = N; % length of signal
% Y = fft(sim_emg);
% P2 = abs(Y/L);          % two-sided spectrum
% P1 = P2(1:L/2+1);       % one-sided spectrum
% P1(2:end-1) = 2*P1(2:end-1); 
% f = Fs*(0:(L/2))/L;     % frequency vector

% Plot EMG Freq
% figure;
% plot(f, P1*100); % scale to %
% title('EMG Frequency Spectrum');
% xlabel('Frequency (Hz)');
% ylabel('Amplitude (%)');
% xlim([0 500]);   % EMG relevant freq range
% grid on;

% EMG Spectrogram
 % figure;
 % window = 256;
 % noverlap = 128;
 % nfft = 512;
 % spectrogram(sim_emg, window, noverlap, nfft, Fs, 'yaxis');
 % title('Time-Frequency EMG (Spectrogram)');
 % ylabel('Frequency (Hz)');
 % xlabel('Time (s)');
 % colorbar;