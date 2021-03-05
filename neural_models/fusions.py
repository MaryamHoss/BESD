from tensorflow.keras.layers import *


def FiLM_Fusion(size, data_type='sum'):
    def fuse(inputs):
        sound, spikes = inputs
        if 'FiLM_v1' in data_type or 'FiLM_v2' in data_type:
            # FiLM starts -------
            beta_snd = Conv1D(size, 3, padding='same')(spikes)
            gamma_snd = Conv1D(size, 3, padding='same')(spikes)

            beta_spk = Conv1D(size, 3, padding='same')(sound)
            gamma_spk = Conv1D(size, 3, padding='same')(sound)

            # changes: 20-8-20 instead of + I made a layer with ADD
            # sound = Multiply()([sound, gamma_snd]) + beta_snd
            sound = Add()([Multiply()([sound, gamma_snd]), beta_snd])
            # spikes = Multiply()([spikes, gamma_spk]) + beta_spk
            spikes = Add()([Multiply()([spikes, gamma_spk]), beta_spk])

            # FiLM ends ---------

            layer = [sound, spikes]

        elif 'FiLM_v3' in data_type or 'FiLM_v4' in data_type:
            # FiLM starts -------
            beta_snd = Dense(size)(spikes)
            gamma_snd = Dense(size)(spikes)

            beta_spk = Dense(size)(sound)
            gamma_spk = Dense(size)(sound)
            # changes: 20-8-20 instead of + I made a layer with ADD

            # sound = Multiply()([sound, gamma_snd]) + beta_snd
            sound = Add()([Multiply()([sound, gamma_snd]), beta_snd])
            # spikes = Multiply()([spikes, gamma_spk]) + beta_spk
            spikes = Add()([Multiply()([spikes, gamma_spk]), beta_spk])

            # FiLM ends ---------

            layer = [sound, spikes]

        else:
            layer = inputs
        return layer

    return fuse


def Fusion(data_type='Add'):
    def fuse(inputs):
        sound, spikes = inputs
        if '_add' in data_type:
            layer = Add()(inputs)

        elif '_concatenate' in data_type:
            layer = Concatenate(axis=-1)(inputs)

        elif '_choice' in data_type:
            layer = ChoiceGated()(inputs)

        elif 'FiLM_v1' in data_type or 'FiLM_v3' in data_type:
            layer = Concatenate(axis=-1)(inputs)

        elif 'FiLM_v2' in data_type or 'FiLM_v4' in data_type:
            layer = sound

        elif '_gating' in data_type:
            gated = Multiply()([sound, spikes])
            gated = Activation('softmax')(gated)
            gated = Multiply()([gated, sound])

            concat = Concatenate(axis=-1)([sound, gated])
            layer = concat

        elif 'noSpike' in data_type:
            layer = sound

        else:
            raise NotImplementedError

        return layer

    return fuse


def ChoiceGated():
    # output =  probs*sound + (1-probs)*spike
    def gating(inputs):
        sound, spikes = inputs
        mul = Multiply()([sound, spikes])
        probs = Activation('softmax')(mul)
        p_snd = Multiply()([probs, sound])
        # changed from sound to spikes
        # p_spk = Multiply()([1 - probs, sound])
        p_spk = Multiply()([1 - probs, spikes])

        output = Add()([p_snd, p_spk])
        return output

    return gating
