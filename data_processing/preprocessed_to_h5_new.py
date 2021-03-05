# lucas way

## for FBC model
import mne, os, h5py, random
import numpy as np
from scipy.io.wavfile import read as read_wav
# from tqdm import tqdm
import tensorflow as tf

from TrialsOfNeuralVocalRecon.tools.nice_tools import rms_normalize

seed = 14
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

n_splits = 1  # 1 30
preprocessing = 'eeg'  # , 'eeg', 'fbc', raw_eeg
seconds = int(60 / n_splits)

if seconds == 60:
    time_folder = '60s'
elif seconds == 2:
    time_folder = '2s'
else:
    NotImplementedError

CDIR = os.path.dirname(os.path.realpath(__file__))
# CDIR = 'C:/Users\hoss3301\work\TrialsOfNeuralVocalRecon\data_processing'
DATADIR = os.path.abspath(os.path.join(*[CDIR, '..', 'data', 'Cocktail_Party']))
TIMEDIR = os.path.join(*[DATADIR, 'Normalized', time_folder])
h5_DIR = os.path.join(*[TIMEDIR, preprocessing])

EEGDIR = os.path.join(*[DATADIR, 'EEG'])
FULLAUDIODIR = os.path.join(*[DATADIR, 'Stimuli', 'Full_Audio'])

if preprocessing == 'raw_eeg':
    PEEGDIR = os.path.join(*[DATADIR, 'preprocessed_EEG', 'RAW_EEG'])
elif preprocessing == 'eeg':
    PEEGDIR = os.path.join(*[DATADIR, 'preprocessed_EEG', 'EEG'])
elif preprocessing == 'fbc':
    PEEGDIR = os.path.join(*[DATADIR, 'preprocessed_EEG', 'FBC'])
else:
    raise NotImplementedError

for path in [TIMEDIR, h5_DIR, FULLAUDIODIR, PEEGDIR]:
    if not os.path.isdir(path):
        os.mkdir(path)



# detect which runs are missing for each subject


subjects = [s for s in os.listdir(EEGDIR) if not 'txt' in s]
subject_trials = {}
for s in subjects:
    subject_folder = os.path.join(EEGDIR, s)
    runs = [r.split('.')[0].split('_')[1].replace('Run', '') for r in os.listdir(subject_folder)]
    runs = sorted([int(r) for r in runs])
    subject_trials.update({s: runs})

# go through each subject and each trial
min_t_activity = 1e8
min_t_audio = 1e8
# subject_indices = np.arange(1,
#                             33).tolist()  # [1, 2, 5, 7, 8, 13, 15, 17]#[ 19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31]#journey subjects#[1, 2, 5, 7, 8, 13, 15, 17, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31]
subject_indices = [1, 2, 5, 7, 8, 13, 15, 17, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31]
#subject_indices = [1, 2, 5, 7, 8, 13]

'''delete_subject=[1,2,3]
for i in range(len(subject_indices)):
    print(i)
    if 'Subject{}'.format(subject_indices[i]) not in subjects:
        delete_subject.append(i)
        for i in range(len(delete_subject)):
            print(delete_subject[i])
            
            del(subject_indices[delete_subject[i]])
'''
win_eeg = 2 * 128
hop_eeg = int(0.1 * 128)
test_subjects = [1, 2, 30, 31]  # np.arange(28, 34).tolist()
test_trials = np.arange(25, 31).tolist()
val_subjects = [3, 29]
val_trials = np.arange(23, 25).tolist()


data = {}
for k_1 in ['eegs_', 'noisy_', 'clean_', 'unattended_', 'subjects_']:
    for k_2 in ['train', 'val', 'test']:
        data[k_1 + k_2] = []

max_val = None
for subject_i in subject_indices:
    print('Subject {}'.format(subject_i))
    set_filepath = PEEGDIR + r'/subject{}.set'.format(subject_i)

    epochs = mne.io.read_epochs_eeglab(set_filepath)
    raw = epochs._data
    events = epochs.events
    events_times = events[:, 0]
    data_subject = raw[:]  # (29,128,7680)

    t_1 = 0
    for n, t in zip(subject_trials['Subject{}'.format(subject_i)],
                    range(1, np.shape(events_times)[0] + 1)):

        trial = np.transpose(data_subject[t_1:t, :, :], (0, 2, 1))
        trial_reshape = np.concatenate(np.split(trial, n_splits, axis=1), axis=0)

        t_1 = t
        audio_path_20000 = os.path.join(FULLAUDIODIR, r'20000/20000_{}.wav'.format(n))
        audio_path_Journey = os.path.join(FULLAUDIODIR, r'Journey/Journey_{}.wav'.format(n))
        _, twenty = read_wav(audio_path_20000)
        _, journey = read_wav(audio_path_Journey)
        twenty = rms_normalize(twenty[0:2646000])
        journey = rms_normalize(journey[0:2646000])

        clean_sound = twenty if subject_i <= 17 else journey
        unattended_sound = twenty if subject_i > 17 else journey

        clean_sound_reshape = np.concatenate(np.split(clean_sound[None], n_splits, axis=1), axis=0)
        unattended_sound_reshape = np.concatenate(np.split(unattended_sound[None], n_splits, axis=1), axis=0)

        # for i in range(n_win_snd):
        # clean_sound_reshapes[i,:]=clean_sound[:,(i)*hop_snd + np.asarray([x for x in range(0,win_snd)]),:]

        noisysnd = (twenty + journey)[None]
        # noisysnd_reshape = np.concatenate(np.split(noisysnd, 30, axis=1), axis=0)
        noisysnd_reshape = np.concatenate(np.split(noisysnd, n_splits, axis=1), axis=0)
        subject_list = (np.array([subject_i] * n_splits))

        # trial_i = n[t - 1]
        trial_i = n
        if subject_i in test_subjects or trial_i in test_trials:
            if subject_i in test_subjects:
                print('this subject is in test subjects: {}'.format(subject_i))
            elif trial_i in test_trials:
                print('this trial is in test trials: {}'.format(trial_i))
            else:
                raise NotImplementedError

            data['eegs_test'].append(trial_reshape)
            data['clean_test'].append(clean_sound_reshape[..., None])
            data['noisy_test'].append(noisysnd_reshape[..., None])
            data['unattended_test'].append(unattended_sound_reshape[..., None])
            data['subjects_test'].append(subject_list)

        elif subject_i in val_subjects or trial_i in val_trials:
            if subject_i in val_subjects:
                print('this subject is in val subjects: {}'.format(subject_i))
            elif trial_i in val_trials:
                print('this trial is in val trials: {}'.format(trial_i))
            else:
                raise NotImplementedError

            data['eegs_val'].append(trial_reshape)
            data['clean_val'].append(clean_sound_reshape[..., None])
            data['noisy_val'].append(noisysnd_reshape[..., None])
            data['unattended_val'].append(unattended_sound_reshape[..., None])
            data['subjects_val'].append(subject_list)

        else:
            data['eegs_train'].append(trial_reshape)
            data['clean_train'].append(clean_sound_reshape[..., None])
            data['noisy_train'].append(noisysnd_reshape[..., None])
            data['unattended_train'].append(unattended_sound_reshape[..., None])
            data['subjects_train'].append(subject_list)

        min_t_activity = min(min_t_activity, trial_reshape.shape[1])
        min_t_audio = min(min_t_audio, clean_sound_reshape.shape[1])

min_t = None
data_copy = data

# get the train max for normalization
train_maxes = {}
for k in data.keys():
    set = [s for s in ['train', 'val', 'test'] if s in k][0]
    train_string = k.replace(set, 'train')
    train_maxes[k] = np.max(np.abs(data_copy[train_string]))

permutations = {}
for k in data.keys():
    set = [s for s in ['train', 'val', 'test'] if s in k][0]
    min_t = min_t_activity if 'eeg' in k else min_t_audio

    if not 'subject' in k:
        # normalize
        data_copy[k] = data_copy[k] / train_maxes[k]
        # concatenate
        data_copy[k] = np.concatenate([m[:, :min_t] for m in data_copy[k]], axis=0)
    else:
        data_copy[k] = np.concatenate(data_copy[k], axis=0)

    print('{}.shape:  {}'.format(k, data_copy[k].shape))

    # shuffle
    if not set in permutations.keys():
        n_samples = data_copy[k].shape[0]
        permutations[set] = np.random.permutation(n_samples)
    data_copy[k] = data_copy[k][permutations[set]]

for k in data_copy.keys():
    print('{}.shape:  '.format(k), data_copy[k].shape)
    f = h5py.File(h5_DIR + '/{}.h5'.format(k), 'w')
    f.create_dataset('{}='.format(k), data=data_copy[k])
    f.close()
