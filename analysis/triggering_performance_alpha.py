import numpy as np
import pandas as pd
from scipy.signal import hilbert, butter, sosfilt
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Set up plotting backend
matplotlib.use('Qt5Agg')

# Set path and other variables
path = 'path to processed data' 
subs = ['strings with subject IDs']  # List of subject IDs
task = 'alpha'
frequency_range = (9, 12)
sfreq = 1000
# Set the search range for stimulation triggers to a cycle at the lowest frequency targeted
stim_range = int(1000 / frequency_range[0])
target_angles = {'ascending': 270, 'peak': 0, 'descending': 90, 'trough': 180}

# Load targeting info
info = pd.read_csv('', header=0, sep=',')
targets = ['ascending', 'descending', 'trough', 'peak']

# Initialize an empty DataFrame to hold the results
fpga_stats_df = pd.DataFrame(columns=['target', 'phase', 'phase_error', 'envelope', 'frequency', 'eyes', 'sensitivity', 'subject'])

# Create the phase detection function
def detect_phase_events(data, filtered_signal, envelope_col='envelope', target_labels=('peak', 'trough', 'ascending', 'descending'),
                        threshold=1000):

    signal = filtered_signal

    # Compute the signal derivatives for peak/trough detection
    d_signal = np.diff(signal, prepend=signal[0])
    deriv_sign = np.sign(d_signal)
    peak_locs = np.where((deriv_sign[:-1] > 0) & (deriv_sign[1:] < 0))[0]
    trough_locs = np.where((deriv_sign[:-1] < 0) & (deriv_sign[1:] > 0))[0]

    # Detect ascending and descending zero-crossings
    sig_sign = np.sign(signal)
    ascending_locs = np.where((sig_sign[:-1] < 0) & (sig_sign[1:] > 0))[0]
    descending_locs = np.where((sig_sign[:-1] > 0) & (sig_sign[1:] < 0))[0]

    event_map = {
        'peak': peak_locs,
        'trough': trough_locs,
        'ascending': ascending_locs,
        'descending': descending_locs,
    }

    def apply_filters(indices):
        """Apply envelope and min_distance filters."""
        if len(indices) == 0:
            return indices
        env_vals = data[envelope_col].iloc[indices].values
        indices = indices[env_vals >= threshold]

        return np.array(indices)

    for label in target_labels:
        hits = np.zeros(len(data), dtype=int)
        if label in event_map:
            locs = event_map[label]
            valid = apply_filters(locs)
            # Keep hits where the target label matches
            matching_targets = data['target'].iloc[valid] == label
            valid = valid[matching_targets.values]
            hits[valid] = 1
        data[f'{label}_hit'] = hits

    return data

# Loop through each subject to process their RT-CL data
for sub in subs:

    # Concatenate file names
    file = f"{path}{sub}_performance_{task}.csv"
    # Read fiel to data frame
    data = pd.read_csv(file, sep=',')

    # Set the amplitude threshold for the subject
    threshold = info[(info['subject'] == int(sub)) & (info['task'] == task)]['threshold'].values[0]

    # Design bandpass filters for frequency and phase signals
    # Use a narrow higher order filter for the frequency estimation and a wider low order filter for the phase estimation
    Bfilter_frequ = butter(2, frequency_range, 'bandpass', fs=sfreq, output='sos')
    Bfilter_phase = butter(1, [1, 20], 'bandpass', fs=sfreq, output='sos')

    # Apply filters to the data
    filtered_signal_frequ = sosfilt(Bfilter_frequ, data['signal'])
    filtered_signal_phase = sosfilt(Bfilter_phase, data['signal'])

    # Get analytic signals using the Hilbert transform
    analytic_signal_frequ = hilbert(filtered_signal_frequ)
    analytic_signal_phase = hilbert(filtered_signal_phase)

    # Calculate phase and instantaneous frequency
    phase_frequ = np.angle(analytic_signal_frequ)
    phase_phase = np.angle(analytic_signal_phase)
    instantaneous_phase_frequ = np.unwrap(phase_frequ)
    instantaneous_frequency = np.diff(instantaneous_phase_frequ) / (2.0 * np.pi) * sfreq
    data['frequency'] = np.append(np.abs(instantaneous_frequency), [0])

    # Store phase (wrapped to 0 to 360 degrees) and the amplitude envelope
    data['phase'] = np.mod(np.rad2deg(phase_phase), 360)
    data['envelope'] = np.abs(analytic_signal_frequ)

    # Calculate phase error for stimulation (stimulated phase - targeted phase angle)
    data['phase_error'] = data.apply(lambda row: ((row['phase'] - target_angles[row['target']] + 180) % 360) - 180, axis=1)

    # Detect phase events
    data = detect_phase_events(
        data, filtered_signal_frequ, threshold=threshold)

    # Create phaseStats DataFrame for stimulation timings
    phase_stats_mask = (data['stim'] == 1)
    phase_stats = data.loc[phase_stats_mask, ['target', 'phase', 'phase_error', 'envelope', 'frequency', 'eyes']].copy()
    phase_stats['subject'] = int(sub)

    # Calculate sensitivity for each target label
    sensitivity = {}
    for label in target_angles.keys():
        target_hit_col = f"{label}_hit"
        target_indices = data.index[data[target_hit_col] == 1]
        correct_stims = sum(
            (data.iloc[start:end]['stim'] == 1) & (data.iloc[start:end]['target'] == label)
            for idx in target_indices
            for start, end in [(max(0, idx - stim_range), min(len(data), idx + (stim_range + 1)))]
        )
        total_occurrences = len(target_indices)
        sensitivity[label] = correct_stims / total_occurrences if total_occurrences > 0 else np.nan
        print(f'sub {sub}({label}), correct_stims: {correct_stims}, total_occurrences: {total_occurrences}, sensitivity: {sensitivity[label]}')

    # Add sensitivity to phase stats
    phase_stats['sensitivity'] = phase_stats['target'].map(sensitivity)

    # Append to the overall DataFrame
    fpga_stats_df = pd.concat([fpga_stats_df, phase_stats], axis=0)

######

# Apply the same analysis to the eventIDE data
# Set some variables again
path = 'path to processed data' 

# Initialize two new data frames
eventide_stats_df = pd.DataFrame(columns=['target', 'phase', 'phase_error', 'envelope', 'frequency', 'eyes', 'sensitivity', 'subject'])

for sub in subs:

    # Concatenate the path and the file
    file = f"{path}{sub}_performance_{task}.csv"
    # Read fiel to data frame
    data = pd.read_csv(file, sep=',')
  
    # Set the amplitude threshold for an individual subject
    threshold = info[(info['subject'] == int(sub)) & (info['task'] == task)]['threshold'].to_numpy()[0]

    # Apply filters to the data
    filtered_signal_frequ = sosfilt(Bfilter_frequ, data['signal'])
    filtered_signal_phase = sosfilt(Bfilter_phase, data['signal'])

    # Get analytic signals using the Hilbert transform
    analytic_signal_frequ = hilbert(filtered_signal_frequ)
    analytic_signal_phase = hilbert(filtered_signal_phase)

    # Calculate phase and instantaneous frequency
    phase_frequ = np.angle(analytic_signal_frequ)
    phase_phase = np.angle(analytic_signal_phase)
    instantaneous_phase_frequ = np.unwrap(phase_frequ)
    instantaneous_frequency = np.diff(instantaneous_phase_frequ) / (2.0 * np.pi) * sfreq
    data['frequency'] = np.append(np.abs(instantaneous_frequency), [0])

    # Store phase (wrapped to 0 to 360 degrees) and the amplitude envelope
    data['phase'] = np.mod(np.rad2deg(phase_phase), 360)
    data['envelope'] = np.abs(analytic_signal_frequ)

    # Calculate phase error for stimulation (stimulated phase - targeted phase angle)
    data['phase_error'] = data.apply(lambda row: ((row['phase'] - target_angles[row['target']] + 180) % 360) - 180, axis=1)

    # Detect phase events
    data = detect_phase_events(
        data, filtered_signal_frequ, threshold=threshold)

    # Create phaseStats DataFrame for stimulation timings
    phase_stats_mask = (data['stim'] == 1)
    phase_stats = data.loc[phase_stats_mask, ['target', 'phase', 'phase_error', 'envelope', 'frequency', 'eyes']].copy()
    phase_stats['subject'] = int(sub)

    # Calculate sensitivity for each target label
    sensitivity = {}
    for label in target_angles.keys():
        target_hit_col = f"{label}_hit"
        target_indices = data.index[data[target_hit_col] == 1]
        correct_stims = sum(
            (data.iloc[start:end]['stim'] == 1) & (data.iloc[start:end]['target'] == label)
            for idx in target_indices
            for start, end in [(max(0, idx - stim_range), min(len(data), idx + (stim_range + 1)))]
        )
        total_occurrences = len(target_indices)
        sensitivity[label] = correct_stims / total_occurrences if total_occurrences > 0 else np.nan
        print(f'sub {sub}({label}), correct_stims: {correct_stims}, total_occurrences: {total_occurrences}, sensitivity: {sensitivity[label]}')

    # Add sensitivity to phase stats
    phase_stats['sensitivity'] = phase_stats['target'].map(sensitivity)

    # Append to the overall DataFrame
    eventide_stats_df = pd.concat([eventide_stats_df, phase_stats], axis=0)

fpga_stats_df['run'] = 'fpga'
eventide_stats_df['run'] = 'eventIDE'

combinedDF = pd.concat([fpga_stats_df, eventide_stats_df], axis=0)
combinedDF.to_csv('output_path/performance_' + task + '.csv', index=False)

# Add plots to check results

## Circular histogram of phase errors by run
fig, axes = plt.subplots(2, 2, subplot_kw={'projection': 'polar'}, figsize=(12, 10))
axes = axes.flatten()

for plot, target in enumerate(target_angles.keys()):
    subset = fpgaStatsDF[fpga_stats_df['target'] == target]

    # Wrap phase errors into [0, 360) degrees for proper full-circle display
    phase_errors_deg = (subset['phase_error'] + 360) % 360
    errors_rad = np.deg2rad(phase_errors_deg)

    # Plot histogram
    axes[plot].hist(errors_rad, bins=36, color=f"C{plot}", alpha=0.75)

    # Configure the polar plot so 0 degrees is at the top
    # and deviations are to the left and right and bottom
    axes[plot].set_theta_zero_location('N') 
    axes[plot].set_theta_direction(-1) 

    # Set custom ticks from -180° to 180°
    tick_angles = np.arange(-180, 181, 45)
    tick_labels = [f"{int(t)}°" for t in tick_angles]
    tick_positions = np.deg2rad((tick_angles + 360) % 360)

    axes[plot].set_xticks(tick_positions)
    axes[plot].set_xticklabels(tick_labels)

    axes[plot].set_title(f"{target.capitalize()} phase error", va='bottom')

plt.suptitle("fpga", fontsize=16)

fig, axes = plt.subplots(2, 2, subplot_kw={'projection': 'polar'}, figsize=(12, 10))
axes = axes.flatten()

for plot, target in enumerate(target_angles.keys()):
    subset = eventide_stats_df[eventide_stats_df['target'] == target]

    # Wrap phase errors into [0, 360) degrees for proper full-circle display
    phase_errors_deg = (subset['phase_error'] + 360) % 360
    errors_rad = np.deg2rad(phase_errors_deg)

    # Plot histogram
    axes[plot].hist(errors_rad, bins=36, color=f"C{plot}", alpha=0.75)

    # Configure the polar plot so 0 degrees is at the top
    # and deviations are to the left and right and bottom
    axes[plot].set_theta_zero_location('N') 
    axes[plot].set_theta_direction(-1) 

    # Set custom ticks from -180° to 180°
    tick_angles = np.arange(-180, 181, 45)
    tick_labels = [f"{int(t)}°" for t in tick_angles]
    tick_positions = np.deg2rad((tick_angles + 360) % 360)

    axes[plot].set_xticks(tick_positions)
    axes[plot].set_xticklabels(tick_labels)

    axes[plot].set_title(f"{target.capitalize()} phase error", va='bottom')

plt.suptitle("eventide", fontsize=16)

# Barplot for triggering probability
fig, ax = plt.subplots()
sns.barplot(combinedDF, x='target', y='sensitivity', hue='run')
ax.set_ylim([0, 1])
