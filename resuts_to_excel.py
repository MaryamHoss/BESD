import os, json
import pandas as pd,sys

sys.path.append('../')

from GenericTools.KerasTools.plot_tools import plot_history
from GenericTools.SacredTools.unzip import unzip_good_exps
from GenericTools.StayOrganizedTools.plot_tricks import large_num_to_reasonable_string

pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', 1)
pd.set_option('precision', 3)
pd.options.display.width = 500
# pd.options.display.max_colwidth = 16

CDIR = os.path.dirname(os.path.realpath(__file__))
EXPERIMENTS = os.path.join(*[CDIR, 'experiments'])
# GEXPERIMENTS = r'D:/work/stochastic_spiking/good_experiments/2021-01-05--ptb-small-noise-good'
GEXPERIMENTS = os.path.join(*[CDIR, 'good_experiments'])

unzip_good_exps(
    GEXPERIMENTS, EXPERIMENTS,
    exp_identifiers=[''], except_identifiers=[],
    unzip_what=['run.json', 'history'])
ds = [d for d in os.listdir(EXPERIMENTS) if not 'other_outputs' in d and 'mnl' in d]
ds = os.listdir(EXPERIMENTS)

def preprocess_key(k):
    k = k.replace('n_dt_per_step', 'n_dt')
    return k


def postprocess_results(k, v):
    if k == 'n_params':
        v = large_num_to_reasonable_string(v, 1)
    return v


print()
history_keys = ['val_mse', 'val_si_sdr_loss',]

config_keys = ['batch_size', 'data_type', 'epochs', 'exp_type', 'fusion_type', 'input_type', "test_type",
               'optimizer','testing',]
hyperparams_keys = ['duration_experiment']
extras = ['d_name', 'where','actual_epochs_ran']

histories = []
method_names = []
df = pd.DataFrame(columns=history_keys + config_keys + hyperparams_keys + extras)
for d in ds:
    d_path = os.path.join(EXPERIMENTS, d)
    history_path = os.path.join(*[d_path, 'other_outputs', 'history.json'])
    hyperparams_path = os.path.join(*[d_path, 'other_outputs', 'results.json'])
    config_path = os.path.join(*[d_path, '1', 'config.json'])
    run_path = os.path.join(*[d_path, '1', 'run.json'])

    with open(config_path) as f:
        config = json.load(f)
        print(config.keys())

    with open(hyperparams_path) as f:
        hyperparams = json.load(f)

    with open(history_path) as f:
        history = json.load(f)
        print(history.keys())

    with open(run_path) as f:
        run = json.load(f)

    results = {}

    if len(extras) > 0:
        results.update({'d_name': d})
        results.update({'where': run['host']['hostname'][:7]})
        results.update({'actual_epochs_ran': len(v) for k, v in history.items()})
    results.update({k: v for k, v in config.items() if k in config_keys})
    what = lambda k, v: max(v) if 'cat' in k else min(v)
    results.update({k.replace('output_net_', '').replace('categorical_', ''): what(k, v) for k, v in history.items() if
                    k in history_keys})    
    results.update({k: postprocess_results(k, v) for k, v in hyperparams.items() if k in hyperparams_keys})

    small_df = pd.DataFrame([results])

    df = df.append(small_df)
    # method_names.append(config['net_name'] + '_' + config['task_name'])
    # khistory = lambda x: None
    # khistory.history = history
    # histories.append(khistory)

# val_categorical_accuracy val_bpc
df = df.sort_values(by=['val_si_sdr_loss'], ascending=True)

print()
print(df.to_string(index=False))
save_path=d_path = os.path.join(CDIR, 'All_results.xlsx')
writer = pd.ExcelWriter(save_path, engine='xlsxwriter')
df.to_excel(writer,na_rep='NaN',sheet_name='Sheet1',index=False)
worksheet = writer.sheets['Sheet1']  # pull worksheet object
for idx, col in enumerate(df.columns):  # loop through all columns
    series = df[col]
    max_len = max((series.astype(str).map(len).max(),  # len of largest item
        len(str(series.name))  # len of column name/header
        )) + 1  # adding a little extra space
    worksheet.set_column(idx, idx, max_len)  # set column width
    
writer.save()
# plot_filename = os.path.join(*['experiments', 'transformer_histories.png'])
# plot_history(histories=histories, plot_filename=plot_filename, epochs=results['final_epochs'],
#              method_names=method_names)
