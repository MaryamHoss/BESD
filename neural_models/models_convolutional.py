# done, checked

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from BESD.neural_models.fusions import FiLM_Fusion, Fusion
from BESD.tools.utils.losses import si_sdr_loss, estoi_loss


def build_conv_with_fusion(learning_rate=0.001,
                           sound_shape=(31900, 1),
                           spike_shape=(7975, 1),
                           downsample_sound_by=3,
                           data_type='WithSpikes'):
    
    #first version:
    activation_encode = 'relu'
    activation_spikes = 'relu'
    activation_decode = 'relu'
    activation_all = 'tanh'
    n_convolutions = 3

    filters = np.linspace(5, 100, n_convolutions).astype(int)
    c = filters[::-1]
    c_end = c[-1]
    c = c[0:c.shape[0] - 1]
    decay_rate = learning_rate / 150

    #downsampled_sound_shape = (int(sound_shape[0] / downsample_sound_by), sound_shape[1])
    input_sound = Input(shape=(None, sound_shape[1]))
    input_spike = Input(shape=(None, spike_shape[1]))

    sound = input_sound
    spikes = input_spike

    for n_filters in c:
        sound = Conv1D(n_filters, 25, strides=1, activation=activation_encode, padding='causal')(sound)
        sound = Dropout(0.3)(LeakyReLU()(LayerNormalization(axis=[2])(sound)))
        spikes = Conv1D(n_filters, 25, strides=1, activation=activation_spikes, padding='causal')(spikes)
        spikes = Dropout(0.3)(LeakyReLU()(LayerNormalization(axis=[2])(spikes)))

        sound, spikes = FiLM_Fusion(sound.shape[2], data_type)([sound, spikes])

    sound = Conv1D(c_end, 25, strides=1, activation=activation_all, padding='causal')(sound)
    sound = Dropout(0.3)(LeakyReLU()(LayerNormalization(axis=[2])(sound)))

    spikes = Conv1D(c_end, 25, strides=1, activation=activation_all, padding='causal')(spikes)
    spikes = Dropout(0.3)(LeakyReLU()(LayerNormalization(axis=[2])(spikes)))
    decoded = Fusion(data_type)([sound, spikes])

    for n_filters in filters[1::]:
        decoded = Conv1D(n_filters, 25, strides=1, activation=activation_decode, padding='causal')(decoded)
        decoded = Dropout(0.3)(LeakyReLU()(LayerNormalization(axis=[2])(decoded)))# first version

    decoded = Conv1D(1, 25, strides=1, activation=activation_all, padding='causal')(decoded)
    decoded = Activation('tanh')(decoded)

    # define autoencoder
    inputs = [input_sound, input_spike] if 'WithSpikes' in data_type else input_sound
    autoencoder = Model(inputs=inputs, outputs=decoded)
    adam = Adam(lr=learning_rate)

    autoencoder.compile(optimizer=adam, loss=si_sdr_loss, metrics=['mse'])
    return autoencoder

