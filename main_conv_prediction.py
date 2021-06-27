import os, sys, shutil, json

from BESD.data_processing.data_generators import getData
from GenericTools.KerasTools.plot_tools import plot_history

sys.path.append('../')
from BESD.neural_models import build_model

import numpy as np
import tensorflow as tf
from GenericTools.StayOrganizedTools.VeryCustomSacred import CustomExperiment, ChooseGPU
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

tf.compat.v1.disable_eager_execution()
from BESD.tools.utils.losses import si_sdr_loss, estoi_loss, stsa_mse
from tqdm import tqdm

CDIR = os.path.dirname(os.path.realpath(__file__))
ex = CustomExperiment('mcp', base_dir=CDIR, seed=14)


@ex.config
def cfg():
    GPU = 0
    learning_rate = 1e-05

    epochs = 1
    batch_size = 16  # 8 for 5 seconds #16 for 2 seconds

    downsample_sound_by = 3  # choices: 3 and 10
    sound_len = 256*3  # 87552  # 87040 for downsample by 10 #87552 for downsample sound by=3  # 87552  # insteead of88200  #2626560#2610860
    fs = 44100 / downsample_sound_by
    spike_len = 256  # 7680 # 7679

    fusion_type = '_FiLM_v2'  ## choices: 1) _concatenate 2) _FiLM_v1 3) _FiLM_v2 4) _FiLM_v3
    # 5) _FiLM_v4 6) _choice 7) _add 8) _transformer_classic 9) _transformer_parallel 10) _transformer_stairs 11)'' for no spikes
    # 11) _transformer_crossed_stairs
    exp_type = 'WithSpikes'  # choices: 1) noSpike 2) WithSpikes
    # fusion_type = fusion_type if not exp_type == 'noSpike' else ''
    input_type = 'denoising_eeg_'  # choices: 1) denoising_eeg_ 2) denoising_eeg_FBC_ 3) real_prediction_ 4) random_eeg_
    # 5) real_reconstruction_ 6) denoising_ 7) cpc_prediction_ 8) real_prediction_eeg_
    data_type = input_type + exp_type + fusion_type
    load_model = False  # wether we start from a previously trained model
    n_channels = 128 if 'eeg' in data_type else 1
    testing=False


@ex.automain
def main(exp_type, data_type,
         learning_rate, epochs, sound_len, spike_len, batch_size, load_model,
         n_channels, downsample_sound_by, GPU, fs,testing):
    exp_dir = os.path.join(*[CDIR, ex.observers[0].basedir])
    images_dir = os.path.join(*[exp_dir, 'images'])
    models_dir = os.path.join(*[exp_dir, 'trained_models'])
    path_best_model = os.path.join(*[models_dir, 'model_weights_{}_predict.h5'.format(exp_type)])
    other_dir = os.path.join(*[exp_dir, 'other_outputs'])
    metrics_filename = os.path.join(*[other_dir, 'metrics.json'])

    ChooseGPU(GPU)

    model = build_model(learning_rate=learning_rate,
                        sound_shape=(None, 1),
                        spike_shape=(None, n_channels),
                        downsample_sound_by=downsample_sound_by,
                        data_type=data_type)


    if not load_model is False:
        load_model = r'{}/{}'.format(CDIR, load_model)
        print('Loading weights from {}'.format(load_model))
        model = tf.keras.models.load_model(load_model, custom_objects={'si_sdr_loss': si_sdr_loss})

    ##############################################################
    #                    train
    ##############################################################

    print("fitting model")
    generators = getData(sound_shape=(sound_len, 1),
                         spike_shape=(spike_len, n_channels),
                         data_type=data_type,
                         batch_size=batch_size,
                         downsample_sound_by=downsample_sound_by)
    checkpoint = ModelCheckpoint(path_best_model, monitor='val_loss', verbose=1, save_best_only=True)
    earlystopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=int(epochs / 5))
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1, cooldown=1)
    tensorboard = TensorBoard(log_dir=other_dir)

    callbacks = [checkpoint, earlystopping, reduce_lr, tensorboard]
    history = model.fit_generator(generators['train'],
                                  epochs=epochs,
                                  validation_data=generators['val'],
                                  callbacks=callbacks)

    plot_filename = os.path.join(*[images_dir, 'train_history_{}.png'.format(exp_type)])
    plot_history(history, plot_filename, epochs)

    del history, plot_filename
    del callbacks, checkpoint, earlystopping, reduce_lr, tensorboard
    del generators['train'], generators['val']

    print('fitting done, saving model')
    model.save(path_best_model)

    if testing:
    ##############################################################
    #                    tests training
    ##############################################################

        print('testing the model')
        # batch_input, batch_snd_out_test = generators['test'].__getitem__()
    
        """
        prediction = model.predict(batch_input)
        image_title = 'prediction plot vs inputs'
        for batch_sample in range(min(32, prediction.shape[0])):
            fig_path = os.path.join(*[images_dir, 'prediction_{}_{}.pdf'.format(exp_type, batch_sample)])
            plot_test(prediction, batch_input, batch_snd_out_test, exp_type, image_title, fig_path, batch_sample)
            fig_path = os.path.join(*[images_dir, 'prediction_{}_{}.png'.format(exp_type, batch_sample)])
            plot_test(prediction, batch_input, batch_snd_out_test, exp_type, image_title, fig_path, batch_sample)
    
        for batch_sample in range(min(32, prediction.shape[0])):
            save_wav(prediction, batch_input, batch_snd_out_test, exp_type, batch_sample, fs, images_dir)
        """
    
        nbf = (sound_len / (downsample_sound_by * 128)) - 1
        print('compile the model')
        model.compile(optimizer='adam', loss=si_sdr_loss,
                      metrics=['mse', si_sdr_loss,
                               estoi_loss(batch_size=batch_size, nbf=nbf, fs=fs),
                               stsa_mse
                               # calc_sdr
                               ])
    
        metrics = {}
    
        # tests on the attended speaker per subject
        for subject in tqdm(range(1, 32)):
            try:
                generators['test'].select_subject(subject)
                prediction = model.predict(generators['test'])
                save_path = os.path.join(*[images_dir, 'prediction_{}.npy'.format(subject)])
                np.save(save_path, prediction)
                del prediction, save_path
                evaluation = model.evaluate(generators['test'])
                for name, metric in zip(model.metrics_names, evaluation):
                    metrics[name + '_subject_{}'.format(subject)] = np.asscalar(metric)
                del evaluation
                json.dump(metrics, open(metrics_filename, "w"))
    
            except Exception as e:
                print('subject {} gave exception'.format(subject))
                print(e)
    
        # tests on the attended speaker
        evaluation = model.evaluate(generators['test'])
        for name, metric in zip(model.metrics_names, evaluation):
            metrics[name] = np.asscalar(metric)
        del evaluation
        json.dump(metrics, open(metrics_filename, "w"))
    
    
        # same tests on the unattended speaker
        evaluation = model.evaluate(generators['test_unattended'])
        for name, metric in zip(model.metrics_names, evaluation):
            metrics[name + '_unattended'] = np.asscalar(metric)
        del evaluation
        json.dump(metrics, open(metrics_filename, "w"))
    
        print(metrics)
    
        shutil.make_archive(ex.observers[0].basedir, 'zip', exp_dir)
    
        return metrics

