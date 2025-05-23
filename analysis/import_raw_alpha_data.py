import pandas as pd
import numpy as np

# Set up paths and other variables
path = 'path_to_raw_data'
columns = ['samples', 'events', 'stim', 'stimChan', 'EOG1', 'EOG2', 'FCz', 'PO7', 'Cz', 'PO8', 'Pz', 'Oz']
subs = ['strings of subject IDs']
task = 'alpha'
frequency_range = (9, 12)
# Load targeting information
info = pd.read_csv('pseudo_random_target.csv', sep=',', header=0)
targets = ['ascending', 'descending', 'trough', 'peak']

# RT-CL loop
# Iterate over each subject
for sub in subs:
  
    # Create file name
    file = path + task + '_' + sub + '.lvm'
    # Read file as a data frame
    data = pd.read_csv(file, sep='\t', names=columns, header=None)
    
    # Get target information for the current subject from csv file containing experiment info
    targetDict = info[(info['subject'] == int(sub)) & (info['task'] == task)].iloc[:, 2:6].to_numpy()
    
    # Add stimulus channel data
    data['signal'] = data['stimChan']

    # Replace stimulation triggers with phase target coding and event_onset with eyes-open vs. eyes-closed
    events = data['events']
    block = 1

    # Iterate over the events starting from the second line
    for line in np.arange(1, len(events)):
    
        # If there is a transition from 0 to 1 (tone onset marking block change), increase the block number
        if (events[line - 1] == 0) and (events[line] == 1):
            block += 1
            events.loc[line] = block
    
    # Get the unique event codes starting from index 3 to skip 0 and 1
    unique_events = np.unique(events)
    codes = unique_events[3:] 

    # Accounting for cases where recording starts at the calibration and not the instruction
    if len(codes) < 10:

        block = 2
        for line in np.arange(1, len(events)):

            if (events[line - 1] == 0) and (events[line] == 1) and (block > 1):
                block = block + 1
                events[line] = block

        codes = np.unique(events)[2:len(np.unique(events))]
        data['events'] = events

        # Set the target column
        data['target'] = 'none'
        data['eyes'] = 'none'

        data['target'].iloc[0:data[data['events'] == codes[0]].index[0]] = 'calibration'
        data['eyes'].iloc[0:data[data['events'] == codes[0]].index[0]] = 'closed'
        data['target'].iloc[data[data['events'] == codes[0]].index[0]:data[data['events'] == codes[1]].index[0]] = 'calibration'
        data['eyes'].iloc[data[data['events'] == codes[0]].index[0]:data[data['events'] == codes[1]].index[0]] = 'open'

        data['target'].iloc[data[data['events'] == codes[1]].index[0]:data[data['events'] == codes[2]].index[0]] = targetDict[0,0]
        data['eyes'].iloc[data[data['events'] == codes[1]].index[0]:data[data['events'] == codes[2]].index[0]] = 'closed'
        data['target'].iloc[data[data['events'] == codes[2]].index[0]:data[data['events'] == codes[3]].index[0]] = targetDict[0,0]
        data['eyes'].iloc[data[data['events'] == codes[2]].index[0]:data[data['events'] == codes[3]].index[0]] = 'open'

        data['target'].iloc[data[data['events'] == codes[3]].index[0]:data[data['events'] == codes[4]].index[0]] = targetDict[0,1]
        data['eyes'].iloc[data[data['events'] == codes[3]].index[0]:data[data['events'] == codes[4]].index[0]] = 'closed'
        data['target'].iloc[data[data['events'] == codes[4]].index[0]:data[data['events'] == codes[5]].index[0]] = targetDict[0,1]
        data['eyes'].iloc[data[data['events'] == codes[4]].index[0]:data[data['events'] == codes[5]].index[0]] = 'open'

        data['target'].iloc[data[data['events'] == codes[5]].index[0]:data[data['events'] == codes[6]].index[0]] = targetDict[0,2]
        data['eyes'].iloc[data[data['events'] == codes[5]].index[0]:data[data['events'] == codes[6]].index[0]] = 'closed'
        data['target'].iloc[data[data['events'] == codes[6]].index[0]:data[data['events'] == codes[7]].index[0]] = targetDict[0,2]
        data['eyes'].iloc[data[data['events'] == codes[6]].index[0]:data[data['events'] == codes[7]].index[0]] = 'open'

        data['target'].iloc[data[data['events'] == codes[7]].index[0]:data[data['events'] == codes[8]].index[0]] = targetDict[0,3]
        data['eyes'].iloc[data[data['events'] == codes[7]].index[0]:data[data['events'] == codes[8]].index[0]] = 'closed'
        data['target'].iloc[data[data['events'] == codes[8]].index[0]:len(data)] = targetDict[0,3]
        data['eyes'].iloc[data[data['events'] == codes[8]].index[0]:len(data)] = 'open'

    # Cases where recording started at the instruction
    elif len(codes) == 10:

        data['events'] = events

        # Set the target column
        data['target'] = 'none'
        data['eyes'] = 'none'

        data['target'].iloc[0:data[data['events'] == codes[0]].index[0]] = 'instruction'
        data['eyes'].iloc[0:data[data['events'] == codes[0]].index[0]] = 'instruction'

        data['target'].iloc[
        data[data['events'] == codes[0]].index[0]:data[data['events'] == codes[1]].index[0]] = 'calibration'
        data['eyes'].iloc[
        data[data['events'] == codes[0]].index[0]:data[data['events'] == codes[1]].index[0]] = 'closed'
        data['target'].iloc[
        data[data['events'] == codes[1]].index[0]:data[data['events'] == codes[2]].index[0]] = 'calibration'
        data['eyes'].iloc[data[data['events'] == codes[1]].index[0]:data[data['events'] == codes[2]].index[0]] = 'open'

        data['target'].iloc[data[data['events'] == codes[2]].index[0]:data[data['events'] == codes[3]].index[0]] = \
        targetDict[0, 0]
        data['eyes'].iloc[
        data[data['events'] == codes[2]].index[0]:data[data['events'] == codes[3]].index[0]] = 'closed'
        data['target'].iloc[data[data['events'] == codes[3]].index[0]:data[data['events'] == codes[4]].index[0]] = \
        targetDict[0, 0]
        data['eyes'].iloc[data[data['events'] == codes[3]].index[0]:data[data['events'] == codes[4]].index[0]] = 'open'

        data['target'].iloc[data[data['events'] == codes[4]].index[0]:data[data['events'] == codes[5]].index[0]] = \
        targetDict[0, 1]
        data['eyes'].iloc[
        data[data['events'] == codes[4]].index[0]:data[data['events'] == codes[5]].index[0]] = 'closed'
        data['target'].iloc[data[data['events'] == codes[5]].index[0]:data[data['events'] == codes[6]].index[0]] = \
        targetDict[0, 1]
        data['eyes'].iloc[data[data['events'] == codes[5]].index[0]:data[data['events'] == codes[6]].index[0]] = 'open'

        data['target'].iloc[data[data['events'] == codes[6]].index[0]:data[data['events'] == codes[7]].index[0]] = \
        targetDict[0, 2]
        data['eyes'].iloc[
        data[data['events'] == codes[6]].index[0]:data[data['events'] == codes[7]].index[0]] = 'closed'
        data['target'].iloc[data[data['events'] == codes[7]].index[0]:data[data['events'] == codes[8]].index[0]] = \
        targetDict[0, 2]
        data['eyes'].iloc[data[data['events'] == codes[7]].index[0]:data[data['events'] == codes[8]].index[0]] = 'open'

        data['target'].iloc[data[data['events'] == codes[8]].index[0]:data[data['events'] == codes[9]].index[0]] = \
        targetDict[0, 3]
        data['eyes'].iloc[
        data[data['events'] == codes[8]].index[0]:data[data['events'] == codes[9]].index[0]] = 'closed'
        data['target'].iloc[data[data['events'] == codes[9]].index[0]:len(data)] = targetDict[0, 3]
        data['eyes'].iloc[data[data['events'] == codes[9]].index[0]:len(data)] = 'open'

    # Cut out the calibration and instructions portion
    # and skip the first 5 seconds of every new block (accounting for switching delay)
    cropped_data = pd.DataFrame(np.zeros(shape=(0, data.shape[1])))
    cropped_data.columns = data.columns
    cropped_data = cropped_data.astype(data.dtypes.to_dict())

    for target in targets:
      current_data = data[data['target'] == target]
      current_data.reset_index(drop=True, inplace=True)
      current_data.drop(index=current_data.index[:5000], axis=0, inplace=True)
      cropped_data = pd.concat([cropped_data, current_data])
  
    data = cropped_data
  
    # The code for stim stays in the digital channel for 2 ms, remove the second instance of 1
    data['stim'] = data['stim'].where((data['stim'] != 1) | (data['stim'].shift() != 1), 0)
    data.reset_index(drop=True, inplace=True)

    # Save processed data frame
    data.to_csv('output_path/' + sub + '_performance_' + task + '_fpga.csv', index=False)

#########

# Process eventIDE data
path = 'path to raw data'
# Because of the output format of eventIDE two paths are needed
# and phase targets were run in separate files
path2 = '/Prediction/SignalFileWriter/SignalData/SignalData_markers.csv'

eventideStatsDF = pd.DataFrame(np.zeros(shape=(0, 8)))
eventideStatsDF.columns = ['target', 'phase', 'phase_error', 'envelope', 'frequency', 'eyes', 'sensitivity', 'subject']
eventideStatsDF = eventideStatsDF.astype({'target': str, 'phase': float, 'phase_error': float, 'envelope': float,
                                  'frequency': float, 'eyes': str, 'sensitivity': float, 'subject': int})

data = pd.DataFrame(np.zeros(shape=(0, 6)), columns=columns)
data = data.astype({'samples': float, 'signal': float, 'stimMarker': str,
                    'stimTime': float, 'stim': int, 'diesdas': int})
data = data.drop(columns=['stimMarker', 'diesdas'])

for sub in subs:

    for target in targets:
        # Concatenate the path and the file
        file = path + 'sub' + sub + '_' + task + '_' + target + path2
        # Read the file into a pandas data frame
        current_data = pd.read_csv(file, sep=',', names=columns, skiprows=1, header=None)
        current_data = current_data.fillna(0)
        current_data = current_data.drop(columns=['stimMarker', 'diesdas'])
        current_data['target'] = target
        data = pd.concat([data, current_data], axis=0)

    # Set the order of phase targets
    targetDict = info[(info['subject'] == int(sub)) & (info['task'] == task)].iloc[:,2:6].to_numpy()

    # Cut out the calibration and instructions portion
    # and skip the first 5 seconds of every new block (accounting for switching delay)
    cropped_data = pd.DataFrame(np.zeros(shape=(0, data.shape[1])))
    cropped_data.columns = data.columns
    cropped_data = cropped_data.astype(data.dtypes.to_dict())

    for target in targets:
        current_data = data[data['target'] == target]
        current_data.reset_index(drop=True, inplace=True)
        current_data.drop(index=current_data.index[:5000], axis=0, inplace=True)
        cropped_data = pd.concat([cropped_data, current_data])

    data = cropped_data
    # The code for stim stays in the digital channel for 2 ms, remove the second instance of 1
    data['stim'] = data['stim'].where((data['stim'] != 1) | (data['stim'].shift() != 1), 0)
    data.reset_index(drop=True, inplace=True)
  
    # Save processed data frame
    data.to_csv('output_path/' + sub + '_performance_' + task + '_eventide.csv', index=False)
