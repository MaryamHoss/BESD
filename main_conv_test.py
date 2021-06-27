import os, sys, shutil, json
from datetime import timedelta
import time

sys.path.append('../')

## For Luca: please put every thing you want to add after this line 
from GenericTools.KerasTools.esoteric_optimizers.AdaBelief import AdaBelief

import pandas as pd
from GenericTools.StayOrganizedTools.VeryCustomSacred import CustomExperiment, ChooseGPU
from GenericTools.StayOrganizedTools.utils import timeStructured

from TrialsOfNeuralVocalRecon.neural_models import build_model
from TrialsOfNeuralVocalRecon.tools.plotting import save_wav, one_plot_test
from TrialsOfNeuralVocalRecon.data_processing.convenience_tools_oldGen import getData
from tensorflow.keras.optimizers import Adam
from TrialsOfNeuralVocalRecon.tools.calculate_intelligibility import find_intel
from TrialsOfNeuralVocalRecon.tools.utils.losses import *
import pickle
from GenericTools.KerasTools.convenience_operations import snake
import numpy as np

tf.compat.v1.enable_eager_execution()

from GenericTools.StayOrganizedTools.utils import setReproducible

FILENAME = os.path.realpath(__file__)
CDIR = os.path.dirname(FILENAME)

random_string = ''.join([str(r) for r in np.random.choice(10, 4)])
ex = CustomExperiment(random_string + '-mcp', base_dir=CDIR, seed=14)


@ex.config
def cfg():
    GPU = 0
    learning_rate = 1e-05
    seed = 14
    epochs = 2
    batch_size = 2  # 8 for 5 seconds #16 for 2 seconds

    downsample_sound_by = 3  # choices: 3 and 10
    sound_len = 87552  # 87552  # 87040 for downsample by 10 #87552 for downsample sound by=3  # 87552  # insteead of88200  #2626560#2610860
    fs = 44100 / downsample_sound_by
    spike_len = 256  # 7680 # 7679

    fusion_type = '_FiLM_v1_orthogonal_multirresolution_convblock:crnd_contrastive_noiseinput_dilation:3_mmfilter:1:5_nconvs:2'  ## choices: 1) _concatenate 2) _FiLM_v1_orthogonal_noiseinput 3) _FiLM_v2 4) _FiLM_v3 5) transformer_classic
    fusion_type = 'performer_concatenate_mmfilter:4:4_nconvs:2'  ## choices: 1) _concatenate 2) _FiLM_v1_orthogonal_noiseinput 3) _FiLM_v2 4) _FiLM_v3 5) transformer_classic
    # 5) _FiLM_v4 6) _choice 7) _add 8) _transformer_classic 9) _transformer_parallel 10) _transformer_stairs 11)'' for no spikes
    # 11) _transformer_crossed_stairs '_FiLM_v1_orthogonal_multirresolution_convblock:crnd_contrastive_noiseinput'
    exp_type = 'WithSpikes'  # choices: 1) noSpike 2) WithSpikes
    # fusion_type = fusion_type if not exp_type == 'noSpike' else ''
    input_type = 'random_eeg_'  # choices: 1) denoising_eeg_ 2) denoising_eeg_FBC_ 3) real_prediction_ 4) random_eeg_
    # 5) real_reconstruction_ 6) denoising_ 7) cpc_prediction_ 8) real_prediction_eeg_ 9) denoising_eeg_RAW_ 10) kuleuven_denoising_eeg_
    data_type = input_type + exp_type + fusion_type
    test_type = 'speaker_independent'
    exp_folder = '2021-01-06--10-26-02--mcp_'
    load_model = os.path.abspath(os.path.join(*[CDIR, 'experiments', exp_folder, 'trained_models',
                                                'model_weights_WithSpikes_predict.h5']))  # wether we start from a previously trained model
    
    n_channels = 128 if 'eeg' in data_type else 1
    testing = True
    optimizer = 'cwAdaBelief'  # adam #adablief
    activation = 'relu'  # sanke
    batch_size_test = 70  # 70 for speaker specific #118 speaker independent
    sound_len_test = 2626560
    spike_len_test = 7680
    batch_start = 4
    batch_stop = 33
    batch_step = 4


dummy_loss = lambda x, y: 0.


@ex.automain
def main(exp_type, data_type,
         learning_rate, epochs, sound_len, spike_len, batch_size, load_model,
         n_channels, downsample_sound_by, GPU, fs, testing, optimizer, activation, test_type, batch_size_test,
         sound_len_test, spike_len_test, batch_start, batch_stop, batch_step, seed):
    exp_dir = os.path.join(*[CDIR, ex.observers[0].basedir])
    images_dir = os.path.join(*[exp_dir, 'images'])
    text_dir = os.path.join(*[exp_dir, 'text'])
    models_dir = os.path.join(*[exp_dir, 'trained_models'])
    path_best_model = os.path.join(*[models_dir, 'model_weights_{}_predict.h5'.format(exp_type)])
    path_best_optimizer = os.path.join(*[models_dir, 'optimizer_{}_predict.pkl'.format(exp_type)])
    other_dir = os.path.join(*[exp_dir, 'other_outputs'])
    history_path = other_dir + '/log.csv'

    starts_at, starts_at_s = timeStructured(False, True)

    ChooseGPU(GPU)
    setReproducible(seed)

    model = build_model(learning_rate=learning_rate,
                        sound_shape=(None, 1),
                        spike_shape=(None, n_channels),
                        downsample_sound_by=downsample_sound_by,
                        data_type=data_type)

    # comment for now to run all the models with old structure
    # total_epochs = epochs * len(generators['train'])
    # print(total_epochs)
    # learning_rate = tf.keras.experimental.CosineDecay(learning_rate, decay_steps=int(4 * total_epochs / 5), alpha=.1)
    # learning_rate = AddWarmUpToSchedule(learning_rate, warmup_steps=total_epochs / 6)
    # optimizer = AdaBelief(learning_rate=learning_rate, clipnorm=1.0, weight_decay=.1)
    if optimizer == 'cwAdaBelief':
        opt='cwAdaBelief'
        optimizer = AdaBelief(learning_rate=learning_rate, weight_decay=.1, clipnorm=1.)
    elif optimizer == 'AdaBelief':
        opt='AdaBelief'
        optimizer = AdaBelief(learning_rate=learning_rate)
    else:
        opt='adam'
        optimizer = Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss=si_sdr_loss, metrics=[si_sdr_loss, 'mse'])  # dummy_loss

    if not load_model is False:
        print('Loading weights from {}'.format(load_model))
        if "noiseinput" in data_type:
            model.load_weights(load_model)
            with open(path_best_optimizer, 'rb') as f:
                weight_values = pickle.load(f)
                model.optimizer.set_weights(weight_values)
        else:

            if opt == 'AdaBelief' or opt == 'cwAdaBelief':
                print('good')
                model = tf.keras.models.load_model(load_model,
                                                   custom_objects={'si_sdr_loss': si_sdr_loss, 'AdaBelief': AdaBelief})
            elif activation == 'snake':
                print('no')
                model = tf.keras.models.load_model(load_model,
                                                   custom_objects={'si_sdr_loss': si_sdr_loss, 'snake': snake})
            else:
                print('no')
                model = tf.keras.models.load_model(load_model, custom_objects={'si_sdr_loss': si_sdr_loss})

   
    model.summary()

    
    #target_model.set_weights(model.get_weights()) 

    ##############################################################
    #                    tests training
    ##############################################################

    shutil.copyfile(FILENAME, text_dir + '/' + os.path.split(FILENAME)[-1])
    shutil.copyfile(os.path.join(CDIR, 'neural_models', 'models_convolutional.py'),
                    text_dir + '/' + 'models_convolutional.py')
    shutil.copyfile(os.path.join(CDIR, 'neural_models', 'layers_transformer.py'),
                    text_dir + '/' + 'layers_transformer.py')
    shutil.copyfile(os.path.join(CDIR, 'neural_models', 'models_transformer_classic.py'),
                    text_dir + '/' + 'models_transformer_classic.py')
    shutil.copyfile(os.path.join(CDIR, 'neural_models', '__init__.py'),
                    text_dir + '/' + '__init__.py')

    results = {}
    ends_at, ends_at_s = timeStructured(False, True)
    results['starts_at'] = starts_at
    results['ends_at'] = ends_at

    duration_experiment = timedelta(seconds=ends_at_s - starts_at_s)
    
    results['duration_experiment'] = str(duration_experiment)

    if testing:
        print('testing the model')

        generators = getData(sound_shape=(sound_len_test, 1),
                             spike_shape=(spike_len_test, n_channels),
                             sound_shape_test=(sound_len_test, 1),
                             spike_shape_test=(spike_len_test, n_channels),
                             data_type=data_type,
                             batch_size=1,
                             downsample_sound_by=downsample_sound_by,
                             test_type=test_type)
        del generators['train'], generators['val']
        prediction_metrics = ['stoi', 'pesq',  'estoi', 'si-sdr']

        noisy_metrics = [m + '_noisy' for m in prediction_metrics]
        df1 = pd.DataFrame(columns=prediction_metrics + noisy_metrics)

        prediction = []

        inference_time = []
        for batch_sample, b in enumerate(generators['test']):
            noisy_snd = b[0][0]
            # eeg = b[0][1]
            clean = b[0][2]

            intel_list = []
            intel_list_noisy = []
            print('batch sample is: ' + str(batch_sample))
            print('sound length is: {}'.format(noisy_snd.shape[1]))
            print('predicting')
            inf_start_s = time.time()
            pred = model.predict(b[0])
            inf_t = time.time() - inf_start_s
            inference_time.append(inf_t)

            fig_path = os.path.join(*[images_dir, 'prediction_{}.png'.format(batch_sample)])
            one_plot_test(pred, clean, noisy_snd, exp_type, '', fig_path)

            prediction.append(pred)
            prediction_concat = np.concatenate(prediction, axis=0)
            print('saving sound')
            save_wav(pred, noisy_snd, clean, exp_type, batch_sample, fs, images_dir)

            print('finding metrics')
            for m in prediction_metrics:
                print('     ', m)
                pred_m = find_intel(clean, pred, metric=m)
                intel_list.append(pred_m)

                noisy_m = find_intel(clean, noisy_snd, metric=m)
                intel_list_noisy.append(noisy_m)

            e_series = pd.Series(intel_list + intel_list_noisy, index=df1.columns)
            df1 = df1.append(e_series, ignore_index=True)

        prediction_filename = os.path.join(*[images_dir, 'prediction_{}.npy'.format(exp_type)])
        np.save(prediction_filename, prediction_concat)

        del prediction, intel_list, intel_list_noisy, pred, prediction_concat, e_series
        df1.to_csv(os.path.join(*[other_dir, 'evaluation.csv']), index=False)

        import matplotlib.pyplot as plt

        for column in df1.columns:
            fig, ax = plt.subplots(1, figsize=(9, 4))

            ax.set_title(column)
            ax.violinplot(df1[column])
            fig.savefig(os.path.join(*[images_dir, '{}.png'.format(column)]), bbox_inches='tight')

        results['inference_time'] = np.mean(inference_time)
    results_filename = os.path.join(*[other_dir, 'results.json'])
    json.dump(results, open(results_filename, "w"))

    shutil.make_archive(ex.observers[0].basedir, 'zip', exp_dir)

    return results
'''
CDIR='C:/Users/hoss3301/work/TrialsOfNeuralVocalRecon'
import os, sys, shutil, json
from datetime import timedelta
import time

sys.path.append('../')

## For Luca: please put every thing you want to add after this line 
from tensorflow_addons.callbacks import TimeStopping
from GenericTools.KerasTools.noise_curriculum import NoiseSchedule
from GenericTools.KerasTools.esoteric_optimizers.AdaBelief import AdaBelief

import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from GenericTools.KerasTools.plot_tools import plot_history
from GenericTools.SacredTools.VeryCustomSacred import CustomExperiment, ChooseGPU
from GenericTools.StayOrganizedTools.utils import timeStructured
from GenericTools.KerasTools.learning_rate_schedules import AddWarmUpToSchedule

from TrialsOfNeuralVocalRecon.neural_models import build_model
from TrialsOfNeuralVocalRecon.tools.plotting import save_wav, one_plot_test
from TrialsOfNeuralVocalRecon.data_processing.convenience_tools_oldGen import getData
from tensorflow.keras.optimizers import Adam
from TrialsOfNeuralVocalRecon.tools.calculate_intelligibility import find_intel
from TrialsOfNeuralVocalRecon.tools.utils.losses import *
import tensorflow.keras.backend as K
import pickle
from GenericTools.KerasTools.convenience_operations import snake
import numpy as np

tf.compat.v1.enable_eager_execution()
from GenericTools.StayOrganizedTools.utils import setReproducible
setReproducible(seed)
seed = 14
setReproducible(seed)
exp_type = 'WithSpikes'  # choices: 1) noSpike 2) WithSpikes
input_type='denoising_eeg_FBC_'
n_channels = 128 if 'eeg' in data_type else 1
learning_rate = 1e-05
fusion_type ='denoising_eeg_FBC_WithSpikes_FiLM_v1'
data_type = input_type + exp_type + fusion_type
test_type = 'speaker_independent'
spike_len = 256  # 7680 # 7679
downsample_sound_by = 3  # choices: 3 and 10
sound_len = 87552  # 87552  # 87040 for downsample by 10 #87552 for downsample sound by=3  # 87552  # insteead of88200  #2626560#2610860
fs = 44100 / downsample_sound_by
exp_folder='2021-01-09--01-53-34--mcp_'
load_model = os.path.abspath(os.path.join(*[CDIR, 'experiments', exp_folder, 'trained_models',
                                            'model_weights_WithSpikes_predict.h5']))  # wether we start from a previously trained model
model = tf.keras.models.load_model(load_model, custom_objects={'si_sdr_loss': si_sdr_loss})
exp_dir=CDIR+'/New folder'
images_dir = os.path.join(*[exp_dir, 'images'])
other_dir = os.path.join(*[exp_dir, 'other_outputs'])
history_path = other_dir + '/log.csv'
sound_len_test = sound_len
spike_len_test = spike_len
path_data=CDIR+'data/Cocktail_Party/Normalized/2s/fbc'
seed = 14
setReproducible(seed)
exp_type = 'WithSpikes'  # choices: 1) noSpike 2) WithSpikes
input_type='denoising_eeg_FBC_'
n_channels = 128 if 'eeg' in data_type else 1
learning_rate = 1e-05
fusion_type ='denoising_eeg_FBC_WithSpikes_FiLM_v1'
data_type = input_type + exp_type + fusion_type
test_type = 'speaker_independent'
spike_len = 256  # 7680 # 7679
downsample_sound_by = 3  # choices: 3 and 10
sound_len = 87552  # 87552  # 87040 for downsample by 10 #87552 for downsample sound by=3  # 87552  # insteead of88200  #2626560#2610860
fs = 44100 / downsample_sound_by
exp_folder='2021-01-09--01-53-34--mcp_'
load_model = os.path.abspath(os.path.join(*[CDIR, 'experiments', exp_folder, 'trained_models',
                                            'model_weights_WithSpikes_predict.h5']))  # wether we start from a previously trained model
model = tf.keras.models.load_model(load_model, custom_objects={'si_sdr_loss': si_sdr_loss})
exp_dir=CDIR+'/New folder'
images_dir = os.path.join(*[exp_dir, 'images'])
other_dir = os.path.join(*[exp_dir, 'other_outputs'])
history_path = other_dir + '/log.csv'
sound_len_test = sound_len
spike_len_test = spike_len
path_data=CDIR+'data/Cocktail_Party/Normalized/2s/fbc'
seed = 14
setReproducible(seed)
exp_type = 'WithSpikes'  # choices: 1) noSpike 2) WithSpikes
input_type='denoising_eeg_FBC_'
n_channels = 128 if 'eeg' in data_type else 1
learning_rate = 1e-05
fusion_type ='denoising_eeg_FBC_WithSpikes_FiLM_v1'
data_type = input_type + exp_type + fusion_type
test_type = 'speaker_independent'
data_type='denoising_eeg_fbc'
spike_len = 256  # 7680 # 7679
downsample_sound_by = 3  # choices: 3 and 10
sound_len = 87552  # 87552  # 87040 for downsample by 10 #87552 for downsample sound by=3  # 87552  # insteead of88200  #2626560#2610860
fs = 44100 / downsample_sound_by
exp_folder='2021-01-09--01-53-34--mcp_'
load_model = os.path.abspath(os.path.join(*[CDIR, 'experiments', exp_folder, 'trained_models',
                                            'model_weights_WithSpikes_predict.h5']))  # wether we start from a previously trained model
model = tf.keras.models.load_model(load_model, custom_objects={'si_sdr_loss': si_sdr_loss})
exp_dir=CDIR+'/New folder'
images_dir = os.path.join(*[exp_dir, 'images'])
other_dir = os.path.join(*[exp_dir, 'other_outputs'])
history_path = other_dir + '/log.csv'
sound_len_test = sound_len
spike_len_test = spike_len
path_data=CDIR+'data/Cocktail_Party/Normalized/2s/fbc'
seed = 14
setReproducible(seed)
exp_type = 'WithSpikes'  # choices: 1) noSpike 2) WithSpikes
input_type='denoising_eeg_FBC_'
n_channels = 128 
learning_rate = 1e-05
fusion_type ='denoising_eeg_FBC_WithSpikes_FiLM_v1'
data_type = input_type + exp_type + fusion_type
test_type = 'speaker_independent'
spike_len = 256  # 7680 # 7679
downsample_sound_by = 3  # choices: 3 and 10
sound_len = 87552  # 87552  # 87040 for downsample by 10 #87552 for downsample sound by=3  # 87552  # insteead of88200  #2626560#2610860
fs = 44100 / downsample_sound_by
exp_folder='2021-01-09--01-53-34--mcp_'
load_model = os.path.abspath(os.path.join(*[CDIR, 'experiments', exp_folder, 'trained_models',
                                            'model_weights_WithSpikes_predict.h5']))  # wether we start from a previously trained model
model = tf.keras.models.load_model(load_model, custom_objects={'si_sdr_loss': si_sdr_loss})
exp_dir=CDIR+'/New folder'
images_dir = os.path.join(*[exp_dir, 'images'])
other_dir = os.path.join(*[exp_dir, 'other_outputs'])
history_path = other_dir + '/log.csv'
sound_len_test = sound_len
spike_len_test = spike_len
path_data=CDIR+'data/Cocktail_Party/Normalized/2s/fbc'
import h5py as hp
File=hp.File(path_data+'/clean_test.h5','r')
path_data=CDIR+'/data/Cocktail_Party/Normalized/2s/fbc'
File=hp.File(path_data+'/clean_test.h5','r')
clean=File['clean_test'][:]
File.close()
clean=clean[:,:sound_len,:]
clean=clean[:,::3,:]
File=hp.File(path_data+'/eegs_test.h5','r')
eeg=File['eegs_test'][:]
File.close()
File=hp.File(path_data+'/noisy_test.h5','r')
noisy_test=File['noisy_test'][:]
File.close()
noisy_test=noisy_test[:,:sound_len,:]
noisy_test=noisy_test[:,::3,:]
prediction=model.predict([noisy_test,eeg])
m='si-sdr'
pred_m = find_intel(clean, prediction, metric=m)
intel_list = []
intel_list_noisy = []
prediction_metrics = ['stoi', 'pesq',  'estoi', 'si-sdr']
noisy_metrics = [m + '_noisy' for m in prediction_metrics]   
for i in range(3516):
    print(i)
    pred_m = find_intel(clean[i,:,:], prediction[i,:,:], metric='stoi')
    intel_list.append(pred_m)
    noisy_m = find_intel(clean[i,:,:], noisy_test[i,:,:], metric='stoi')
    intel_list_noisy.append(noisy_m)
    
for i in range(3516):
    print(i)
    pred_m = find_intel(clean[i:i+1,:,:], prediction[i,:,:], metric='stoi')
    intel_list.append(pred_m)
    noisy_m = find_intel(clean[i:i+1,:,:], noisy_test[i,:,:], metric='stoi')
    intel_list_noisy.append(noisy_m)
    
for i in range(3516):
    print(i)
    pred_m = find_intel(clean[i:i+1,:,:], prediction[i:i+1,:,:], metric='stoi')
    intel_list.append(pred_m)
    noisy_m = find_intel(clean[i:i+1,:,:], noisy_test[i:i+1,:,:], metric='stoi')
    intel_list_noisy.append(noisy_m)
    
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, figsize=(9, 4))ax.violinplot(df1[column])
fig, ax = plt.subplots(1, figsize=(9, 4))
ax.violinplot(intel_list)
fig, ax = plt.subplots(1, figsize=(9, 4))
ax.violinplot(intel_list)
fig, ax = plt.subplots(1, figsize=(9, 4))
ax.violinplot(intel_list_noisy)
'''
'''#to test the model with long sounds, doesnt work, takes too much memory

sound_test=batch_input[0]
sound_len_test=sound_test.shape[1]
spike_test=batch_input[1]
spike_len_test=spike_test.shape[1]

test_model=build_model(learning_rate=learning_rate,
                sound_shape=(sound_len_test, 1),
                spike_shape=(spike_len_test, n_channels),
                downsample_sound_by=downsample_sound_by,
                data_type=data_type)
W_init = model.get_weights()
test_model.set_weights(W_init)
prediction = test_model.predict(batch_input)
image_title = 'prediction plot vs inputs'
for batch_sample in range(min(32, batch_size)):
fig_path = os.path.join(*[images_dir, 'prediction_{}_{}.pdf'.format(exp_type, batch_sample)])
plot_test(prediction, batch_input, batch_snd_out_test, exp_type, image_title, fig_path, batch_sample)
fig_path = os.path.join(*[images_dir, 'prediction_{}_{}.png'.format(exp_type, batch_sample)])
plot_test(prediction, batch_input, batch_snd_out_test, exp_type, image_title, fig_path, batch_sample)

for batch_sample in range(min(32, batch_size)):
save_wav(prediction, batch_input, batch_snd_out_test, exp_type, batch_sample, fs, images_dir)

model.compile(optimizer='adam', loss=si_sdr_loss,
          metrics=['mse', si_sdr_loss, estoi_loss(batch_size=batch_size, fs=fs), stsa_mse])
evaluation = model.evaluate(generator_test)

metrics = {}
for name, metric in zip(model.metrics_names, evaluation):
metrics[name] = np.asscalar(metric)

shutil.make_archive(ex.observers[0].basedir, 'zip', exp_dir)

# email_results(
#    folders_list=[config_dir],
#    name_experiment=' guinea, attention ',
#    receiver_emails=['manucelotti@gmail.com', 'm.hosseinite@gmail.com'])

return metrics'''

# tests on the attended speaker per subject


'''        for subject in tqdm(range(1, 32)):
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
     print(e)'''

# tests on the attended speaker
# attended_metrics = model.metrics_names
# unattended_metrics = [m + '_unattended' for m in model.metrics_names]
# df = pd.DataFrame(columns=attended_metrics + unattended_metrics)

# for sample, unattended_sample in tqdm(zip(generators['test'], generators['test_unattended'])):
#  evaluation = model.evaluate(*sample)
#  unattended_evaluation = model.evaluate(*unattended_sample)
# del sample, unattended_sample
# e_series = pd.Series(evaluation + unattended_evaluation, index=df.columns)
# df = df.append(e_series, ignore_index=True)
# print(evaluation, unattended_evaluation)
# df.to_csv(os.path.join(*[other_dir, 'evaluation.csv']), index=False)

# print(df)


'''batch_input_unattended, clean_unattended = generators['test_unattended'].__getitem__()
del batch_input_unattended
unattended_metrics = [m + '_unattended' for m in prediction_metrics]    
df2 = pd.DataFrame(columns=unattended_metrics)     



for batch_sample in range(batch_size_test):
    intel_list_unattended = []
    print(' batch sample is: ' + str(batch_sample))


    intel_stoi_pred_unattended = find_intel(clean_unattended[batch_sample:batch_sample + 1, :, :], pred, metric='stoi')
    intel_list_unattended.append(intel_stoi_pred_unattended)

    intel_estoi_pred_unattended = find_intel(clean_unattended[batch_sample:batch_sample + 1, :, :], pred, metric='estoi')
    intel_list_unattended.append(intel_estoi_pred_unattended[0])

    intel_sdr_pred_unattended = find_intel(clean_unattended[batch_sample:batch_sample + 1, :, :], pred, metric='si-sdr')
    intel_list_unattended.append(intel_sdr_pred_unattended)

    intel_pesq_pred_unattended = find_intel(clean_unattended[batch_sample:batch_sample + 1, :, :], pred)
    intel_list_unattended.append(intel_pesq_pred_unattended)


    e_series = pd.Series(intel_list_unattended, index=df2.columns)
    df2 = df2.append(e_series, ignore_index=True)

df=pd.concat([df1, df2], axis=1)
df.to_csv(os.path.join(*[other_dir, 'evaluation.csv']), index=False)'''

