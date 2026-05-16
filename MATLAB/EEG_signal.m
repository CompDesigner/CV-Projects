% Define parameters
Fs = 500;            % Sampling frequency (Hz)
t = 0:1/Fs:10-1/Fs;        % Time vector from 0 to 10 seconds
N = length(t);       % Number of samples

% Generate component sine waves for different brain rhythms
delta_wave = 10 * sin(2 * pi * 2 * t);    % 0.5-4 Hz ,2 Hz, high amp (deep sleep)
theta_wave = 7 * sin(2 * pi * 6 * t);     % 4-8 Hz ,6 Hz, medium amp (drowsiness)
alpha_wave = 8 * sin(2 * pi * 10 * t);    % 8-13 Hz ,10 Hz, prominent amp (idle/relax)
beta_wave = 15 * sin(2 * pi * 20 * t);    % 13-30 Hz ,20 Hz, low amp (alertness)

% Combine waves to form a composite signal
eeg_signal = alpha_wave + beta_wave + theta_wave + delta_wave;

% Add random Gaussian noise
noise = 5 * randn(1, N); % 5 is the standard deviation for the noise level
simulated_eeg = eeg_signal + noise;

% Plot the delta wave
figure;
plot(t, delta_wave);
title('High Amp Signal');
xlabel('Time (s)');
ylabel('Amplitude (\muV)');
grid on;

% Plot the theta wave
figure;
plot(t, theta_wave);
title('Medium Amp Signal');
xlabel('Time (s)');
ylabel('Amplitude (\muV)');
grid on;

% Plot the alpha wave
figure;
plot(t, alpha_wave);
title('Normal Amp EEG Signal');
xlabel('Time (s)');
ylabel('Amplitude (\muV)');
grid on;

% Plot the beta wave
figure;
plot(t, beta_wave);
title('Low Amp Signal');
xlabel('Time (s)');
ylabel('Amplitude (\muV)');
grid on;

% Plot the composite signal
figure;
plot(t, eeg_signal);
title('Normal EEG Signal');
xlabel('Time (s)');
ylabel('Amplitude (\muV)');
grid on;

% Plot the simulated EEG signal
figure;
plot(t, simulated_eeg);
title('Simulated Normal EEG Signal');
xlabel('Time (s)');
ylabel('Amplitude (\muV)');
grid on;

% Plot the power spectral density (PSD) to confirm frequency content
% figure;
% pwelch(simulated_eeg, [], [], [], Fs);
% title('Power Spectral Density of Simulated EEG');