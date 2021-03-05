# done, changed to correct

import h5py
import numpy as np
from numpy.random import seed
import tensorflow as tf

'''
def Prediction_Generator(
        filepath, nb_lines, max_len, shift, downsample_by):
    file = h5py.File(filepath, 'r')
    key = [key for key in file][0]

    i = 0
    while i < nb_lines:
        batch = file[key][i, shift:max_len + shift]
        #the problem is the line above gives an output of shape (218880,1), to fix it we either add a dimension (1,218880,1) or on the line
        #below we change batch=batch[::downsample_by]. Don't know which one is the format required by the generator. will test
        #batch = batch[:, ::downsample_by]
        batch = batch[::downsample_by]
        i += 1
        yield batch'''

def Prediction_Generator(
        filepath, nb_lines, max_len, shift, downsample_by,batch_size):
    file = h5py.File(filepath, 'r')
    key = [key for key in file][0]

    batch_index = 0
    batch_stop=0
    while batch_stop < nb_lines:
        batch_start = batch_index * batch_size
        batch_stop = batch_start + batch_size
        
        batch = file[key][batch_start:batch_stop, shift:max_len + shift]
        #the problem is the line above gives an output of shape (218880,1), to fix it we either add a dimension (1,218880,1) or on the line
        #below we change batch=batch[::downsample_by]. Don't know which one is the format required by the generator. will test
        #batch = batch[:, ::downsample_by]
        batch = batch[:,::downsample_by]
        batch_index += 1
        yield batch
        

            
def Prediction_Dataset(
        filepath_input_first,
        filepath_input_second,
        filepath_output,
        sound_shape=(3, 1),
        spike_shape=(3, 1),
        batch_size=32,
        data_type='',
        downsample_sound_by=3):
    #this lower line didn't work
    #with h5py.File(filepath_input_first, 'r') as f:
        #nb_lines = len(f[f[0]])
    
    with h5py.File(filepath_input_first, 'r') as f:
        for key in f.keys():
            nb_lines = len(f[key])
            
    print('Number of samples in the file: {}'.format(nb_lines))

    downsampled_sound_shape = (int(sound_shape[0] / downsample_sound_by), sound_shape[1])

    input_sound_generator = tf.data.Dataset.from_generator(
        Prediction_Generator,
        args=[filepath_input_first, nb_lines, sound_shape[0], 0, downsample_sound_by,batch_size],
        output_types=(tf.float32), output_shapes=(None,downsampled_sound_shape[0],downsampled_sound_shape[1]))
    output_sound_generator = tf.data.Dataset.from_generator(
        Prediction_Generator,
        args=[filepath_output, nb_lines, sound_shape[0], 1, downsample_sound_by,batch_size],
        output_types=(tf.float32), output_shapes=(None,downsampled_sound_shape[0],downsampled_sound_shape[1]))

    if data_type == 'noSpikes':
        input_generator = input_sound_generator
    else:
        spike_generator = tf.data.Dataset.from_generator(
            Prediction_Generator,
            args=[filepath_input_second, nb_lines, spike_shape[0], 0, 1,batch_size],
            output_types=(tf.float32), output_shapes=(None,spike_shape[0],spike_shape[1]))
        input_generator = tf.data.Dataset.zip((input_sound_generator, spike_generator))

    generator = tf.data.Dataset.zip((input_generator, output_sound_generator))

    generator_batch = generator.shuffle(20)#.batch(batch_size)
    return generator_batch


def Random_Generator(
        shape=None,
        n_samples=32):
    i = 0
    while i < n_samples:
        i += 1
        yield np.array(np.random.rand(*shape), dtype='float32')


def Random_Dataset(sound_shape,
                   spike_shape,
                   batch_size,
                   data_type,
                   downsample_sound_by):
    downsampled_sound_shape = (int(sound_shape[0] / downsample_sound_by), sound_shape[1])
    n_samples = 100
    sound_generator = tf.data.Dataset.from_generator(
        Random_Generator, args=[downsampled_sound_shape, n_samples],
        output_types=(tf.float32), output_shapes=downsampled_sound_shape)

    if data_type == 'noSpikes':
        input_generator = sound_generator
    else:
        spike_generator = tf.data.Dataset.from_generator(
            Random_Generator, args=[spike_shape, n_samples],
            output_types=(tf.float32), output_shapes=spike_shape)
        input_generator = tf.data.Dataset.zip((sound_generator, spike_generator))

    generator = tf.data.Dataset.zip((input_generator, sound_generator))

    generator_batch = generator.shuffle(20).batch(batch_size)
    return generator_batch


if __name__ == '__main__':
    epochs = 2  # 15  # 75  # 3
    batch_size = 2  # for 5 seconds #16 for 2 seconds

    downsample_sound_by = 3  # choices: 3 and 10
    sound_len = 4 * 3  # 87552  # 87040 for downsample by 10 #87552 for downsample sound by=3  # 87552  # insteead of88200  #2626560#2610860
    fs = 44100 / downsample_sound_by
    spike_len = 2  # 256  # 7680 # 7679

    fusion_type = '_add'  ## choices: 1) _concatenate 2) _FiLM_v1 3) _FiLM_v2 4) _FiLM_v3
    exp_type = 'WithSpikes'  # choices: 1) noSpike 2) WithSpikes
    input_type = 'random_eeg_'  # choices: 1) denoising_eeg_ 2) denoising_eeg_FBC_ 3) real_prediction_ 4) random_eeg_
    data_type = input_type + exp_type + fusion_type
    n_channels = 128 if 'eeg' in data_type else 1

    sound_shape = (sound_len, 1)
    spike_shape = (spike_len, n_channels)

    data_type = 0 if data_type == 'noSpikes' else 1
    downsampled_sound_shape = (int(sound_shape[0] / downsample_sound_by), sound_shape[1])
    n_samples = 100
    sound_generator = tf.data.Dataset.from_generator(Random_Generator, args=[downsampled_sound_shape, n_samples],
                                                     output_types=(tf.float32), output_shapes=downsampled_sound_shape)
    spike_generator = tf.data.Dataset.from_generator(Random_Generator, args=[spike_shape, n_samples],
                                                     output_types=(tf.float32), output_shapes=spike_shape)

    input_generator = tf.data.Dataset.zip((sound_generator, spike_generator))
    generator = tf.data.Dataset.zip((input_generator, sound_generator))

    ds_series_batch = generator.shuffle(20).batch(batch_size)

    input, output = next(iter(ds_series_batch))
    print(input)
    print()
    print(output)
