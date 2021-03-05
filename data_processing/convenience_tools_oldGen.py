import time, os, itertools

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.io import wavfile

from BESD.data_processing.data_generators_old import Prediction_Generator, Random_Generator, \
    CPC_Generator

CDIR = os.path.dirname(os.path.realpath(__file__))
DATADIR = os.path.abspath(os.path.join(*[CDIR, '..', 'data']))
EEG_h5_DIR = os.path.abspath(os.path.join(*[DATADIR, 'Cocktail_Party', 'Normalized']))


def timeStructured():
    named_tuple = time.localtime()  # get struct_time
    time_string = time.strftime("%Y-%m-%d-%H-%M-%S", named_tuple)
    return time_string


def plot_predictions(sound2sound, spike2sound, generator_test_snd2snd, generator_test_spk2snd, ex):
    # test spike to sound

    batch_spk_test, batch_snd_test = generator_test_spk2snd.__getitem__()

    predicted_sound = spike2sound.predict_on_batch(batch_spk_test)
    one_spike, one_sound, one_predicted_sound = batch_spk_test[0], batch_snd_test[0], predicted_sound[0]

    fig, axs = plt.subplots(3)
    fig.suptitle('Vertically stacked subplots')
    axs[0].plot(one_spike)
    axs[0].set_title('spike')
    axs[1].plot(one_sound)
    axs[1].set_title('sound')
    axs[2].plot(one_predicted_sound)
    axs[2].set_title('predicted sound')

    fig_path = 'model/spk2snd.pdf'
    fig.savefig(fig_path, bbox_inches='tight')
    ex.add_artifact(fig_path)

    # test sound to sound
    batch_snd_input, batch_snd_output = generator_test_snd2snd.__getitem__()

    predicted_sound = sound2sound.predict_on_batch(batch_snd_input)
    one_sound_input, one_sound_output, one_predicted_sound = batch_snd_input[0], batch_snd_output[0], predicted_sound[0]

    fig, axs = plt.subplots(3)
    fig.suptitle('Vertically stacked subplots')
    axs[0].plot(one_sound_input)
    axs[0].set_title('input sound')
    axs[1].plot(one_sound_output)
    axs[1].set_title('output sound')

    axs[2].plot(one_predicted_sound)
    axs[2].set_title('predicted sound')

    fig_path = 'model/snd2snd.pdf'
    fig.savefig(fig_path, bbox_inches='tight')
    ex.add_artifact(fig_path)


def plot_losses(n2n_lh, k2n_lh, n2n_lh_cpc, k2n_lh_cpc, ex):
    # plot training losses
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(n2n_lh, label='n2n_lh')
    ax.plot(k2n_lh, label='k2n_lh')
    ax.plot(n2n_lh_cpc, label='n2n_lh_cpc')
    ax.plot(k2n_lh_cpc, label='k2n_lh_cpc')
    ax.set_title('model loss')
    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    ax.legend()

    random_string = ''.join([str(r) for r in np.random.choice(10, 4)])
    plot_filename = 'model/{}_train_losses.pdf'.format(random_string)
    fig.savefig(plot_filename, bbox_inches='tight')
    ex.add_artifact(plot_filename)


def getDataPaths(data_type):
    data_paths = {}
    for set in ['train', 'val', 'test']:
        time_folder = '60s' if set == 'test' else '2s'

        # FIXME:

        if 'eeg' in data_type:
            if 'denoising' in data_type:
                if 'FBC' in data_type:
                    data_paths['in1_{}'.format(set)] = os.path.join(
                        *[EEG_h5_DIR, time_folder, 'fbc', 'noisy_{}.h5'.format(set)])
                    data_paths['in2_{}'.format(set)] = os.path.join(
                        *[EEG_h5_DIR, time_folder, 'fbc', 'eegs_{}.h5'.format(set)])
                    data_paths['out_{}'.format(set)] = os.path.join(
                        *[EEG_h5_DIR, time_folder, 'fbc', 'clean_{}.h5'.format(set)])
                    data_paths['out_{}_unattended'.format(set)] = os.path.join(
                        *[EEG_h5_DIR, time_folder, 'fbc', 'unattended_{}.h5'.format(set)])

                elif 'RAW' in data_type:
                    data_paths['in1_{}'.format(set)] = os.path.join(
                        *[EEG_h5_DIR, time_folder, 'raw_eeg', 'noisy_{}.h5'.format(set)])
                    data_paths['in2_{}'.format(set)] = os.path.join(
                        *[EEG_h5_DIR, time_folder, 'raw_eeg', 'eegs_{}.h5'.format(set)])
                    data_paths['out_{}'.format(set)] = os.path.join(
                        *[EEG_h5_DIR, time_folder, 'raw_eeg', 'clean_{}.h5'.format(set)])
                    data_paths['out_{}_unattended'.format(set)] = os.path.join(
                        *[EEG_h5_DIR, time_folder, 'raw_eeg', 'unattended_{}.h5'.format(set)])

                else:
                    data_paths['in1_{}'.format(set)] = os.path.join(
                        *[EEG_h5_DIR, time_folder, 'eeg', 'noisy_{}.h5'.format(set)])
                    data_paths['in2_{}'.format(set)] = os.path.join(
                        *[EEG_h5_DIR, time_folder, 'eeg', 'eegs_{}.h5'.format(set)])
                    data_paths['out_{}'.format(set)] = os.path.join(
                        *[EEG_h5_DIR, time_folder, 'eeg', 'clean_{}.h5'.format(set)])
                    data_paths['out_{}_unattended'.format(set)] = os.path.join(
                        *[EEG_h5_DIR, time_folder, 'eeg', 'unattended_{}.h5'.format(set)])

            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    return data_paths


def getData(
        sound_shape=(3, 1),
        spike_shape=(3, 1),
        data_type='real_prediction',
        batch_size=128,
        downsample_sound_by=3,
        terms=3, predict_terms=3):
    data_paths = {}
    if not 'random' in data_type:
        data_paths = getDataPaths(data_type)

    generators = {}
    if not any([i in data_type for i in ['cpc', 'random']]):

        generator_train, generator_val, generator_test = [
            Prediction_Generator(
                filepath_input_first=data_paths['in1_{}'.format(set_name)],
                filepath_input_second=data_paths['in2_{}'.format(set_name)],
                filepath_output=data_paths['out_{}'.format(set_name)],
                sound_shape=sound_shape,
                spike_shape=spike_shape,
                batch_size=b,
                data_type=data_type,
                downsample_sound_by=downsample_sound_by)
            for b, set_name in zip([batch_size, batch_size, 1], ['train', 'val', 'test'])]

        try:
            generator_test_unattended = Prediction_Generator(
                filepath_input_first=data_paths['in1_test'],
                filepath_input_second=data_paths['in2_test'],
                filepath_output=data_paths['out_test_unattended'],
                sound_shape=sound_shape,
                spike_shape=spike_shape,
                batch_size=1,
                data_type=data_type,
                downsample_sound_by=downsample_sound_by)

            generators.update(test_unattended=generator_test_unattended)
        except:
            print('Run preprocessed_to_h5.py again to generate the unattended_x.h5')

    elif 'random' in data_type:

        generator_train, generator_val, generator_test = [Random_Generator(sound_shape=sound_shape,
                                                                           spike_shape=spike_shape,
                                                                           batch_size=batch_size,
                                                                           data_type=data_type,
                                                                           downsample_sound_by=downsample_sound_by)
                                                          for _ in range(3)]
        generators.update(test_unattended=generator_test)
    else:
        raise NotImplementedError

    generators.update(
        train=generator_train,
        val=generator_val,
        test=generator_test)
    return generators


def plot_test(prediction, batch_input, batch_snd_out_test, exp_type, image_title, fig_path, batch_sample):
    if 'WithSpikes' in exp_type:
        one_sound = batch_input[0][batch_sample]
    elif 'noSpike' in exp_type:
        one_sound = batch_input[batch_sample]
    else:
        raise NotImplementedError

    one_sound_out, one_predicted_sound = batch_snd_out_test[batch_sample], prediction[batch_sample]

    fig, axs = plt.subplots(3)
    fig.suptitle(image_title)
    axs[0].plot(one_sound)
    axs[0].set_title('input sound')
    axs[1].plot(one_sound_out)
    axs[1].set_title('output sound')
    axs[2].plot(one_predicted_sound)
    axs[2].set_title('predicted sound')

    fig.savefig(fig_path, bbox_inches='tight')
    plt.close('all')


def save_wav(prediction, batch_input, batch_snd_out_test, exp_type, batch_sample, fs, images_dir):
    if 'WithSpikes' in exp_type:
        one_sound_in = batch_input[0][batch_sample]
    elif 'noSpike' in exp_type:
        one_sound_in = batch_input[batch_sample]
    else:
        raise NotImplementedError

    clean_sound_path = os.path.join(*[images_dir, 'clean_{}_{}.wav'.format(exp_type, batch_sample)])
    noisy_sound_path = os.path.join(*[images_dir, 'noisy_{}_{}.wav'.format(exp_type, batch_sample)])
    predicted_sound_path = os.path.join(*[images_dir, 'prediction_{}_{}.wav'.format(exp_type, batch_sample)])

    one_sound_out, one_predicted_sound = batch_snd_out_test[batch_sample], prediction[batch_sample]
    m = np.max(np.abs(one_sound_in))
    one_sound_in32 = (one_sound_in / m).astype(np.float32)
    wavfile.write(noisy_sound_path, int(fs), one_sound_in32)

    m = np.max(np.abs(one_sound_out))
    one_sound_out32 = (one_sound_out / m).astype(np.float32)
    wavfile.write(clean_sound_path, int(fs), one_sound_out32)

    m = np.max(np.abs(one_predicted_sound))
    one_predicted_sound32 = (one_predicted_sound / m).astype(np.float32)
    wavfile.write(predicted_sound_path, int(fs), one_predicted_sound32)


if __name__ == '__main__':
    data_type = 'denoising_eeg_'
    segment_length = ''
    batch_size = 10
    downsample_sound_by = 2
    generators = getData(sound_shape=(3 * downsample_sound_by, 1),
                         spike_shape=(3, 128),
                         data_type=data_type,
                         segment_length=segment_length,
                         batch_size=batch_size,
                         downsample_sound_by=downsample_sound_by)

    print(generators.keys())
