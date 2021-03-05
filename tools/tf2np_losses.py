import numpy as np
import os
import sys
sys.path.append('../')
sys.path.append('../..')
from TrialsOfNeuralVocalRecon.tools.utils.OBM import OBM
from TrialsOfNeuralVocalRecon.tools.utils.losses import *
import tensorflow as tf

#for this to work, cd in the folder containing this code
DATADIR = os.path.curdir + r'/../data'
Noise_type='NS_'
Level='55'
SNR='5_'
experiment='NoSpike'

#DATADIR='./data'
def test():

    #batch_size = 32
    #time_steps = 30000 #2100
    #features = 1
    #loss_matrix=np.zeros((3,768)) #the matrix at the end, having the three losses in it
    loss_matrix_w=np.zeros((1,2082))
    loss_matrix_n=np.zeros((1,2082))
    #np_true = np.random.rand(batch_size, time_steps, features).astype(np.float32)
    #np_pred = np.random.rand(batch_size, time_steps, features).astype(np.float32)
    content = [d for d in os.listdir(DATADIR) if 'npy' in d]
    print(content)
    path_true=DATADIR +'/4layers/'+experiment+ '/clean_'+Level+'.npy'
    np_true_1 = np.load(path_true)[:].astype(np.float32)
    np_true=np.zeros(shape=(np_true_1.shape[0]*np_true_1.shape[1],np_true_1.shape[2],1))
    np_true=np.reshape(np_true_1,(np_true_1.shape[0]*np_true_1.shape[1],np_true_1.shape[2],1))
    np_true=np_true[:,1:,:]
    #np_pred = np.load(DATADIR + '/prediction_NoSpike.npy')[36+36*z:37+36*z, :, np.newaxis].astype(np.float32)
    path_pred=DATADIR + '/4layers/'+experiment+ '/prediction_'+Noise_type+SNR+Level+'.npy'
    np_pred = np.load(path_pred)[:, :,:].astype(np.float32)
    print('real data shapes: ', np_true.shape, np_pred.shape)
    np_pred_w=prediction_old_filmv1
    np_pred_n=prediction_old_noSpike
    for batch in range(np_true.shape[0]):
        print(batch)
        np_true_batch=np_true[batch:batch+1,:,:] #for the code to run for each example we have separately
                #so at the end we can take a mean over all the estoi metrics
        np_pred_batch_w=np_pred_w[batch:batch+1,:,:]
        np_pred_batch_n=np_pred_n[batch:batch+1,:,:]
        batch_size = np_true_batch.shape[0]
        nbf=(29184/128)-1#nbf=(31900/128)-1
        loss_1 = si_sdr_loss  # works tf1, tf2
        loss_2 = estoi_loss(batch_size=batch_size, nbf=nbf,fs=44100/3) # works tf2 but strange dependency with batch size and timesteps
        loss_3 = stoi_loss(batch_size=batch_size, nbf=nbf) # works tf2
        loss_4 = stsa_mse # works tf1, tf2
        loss_5 = pmsqe_log_mse_loss(batch_size=batch_size) # doesn't work: need .mat tf1
        numerator=0
        for loss in [loss_2]: #[loss_1, loss_2, loss_3]:
            lw_ = loss(np_true_batch.astype(np.float32), np_pred_batch_w.astype(np.float32))
            ln_ = loss(np_true_batch.astype(np.float32), np_pred_batch_n.astype(np.float32))
    
    
            if tf.__version__[:2] == '1.':
                grad = tf.gradients(l_, [np_pred_batch, np_true_batch])
                sess = tf.Session()
                with sess.as_default():
                    try:
                        l_w = lw_.eval()
                        l_n = ln_.eval()
                    except:
                        ln = ln_[0].eval()
                        lw = lw_[0].eval()
                    #grad = [g.eval() for g in grad]
                loss_matrix_w[numerator,batch]=lw
                loss_matrix_n[numerator,batch]=ln
            elif tf.__version__[:2] == '2.':
                grad = 'hi'
                lw = lw_.numpy()
                loss_matrix_w[numerator,batch]=lw
                ln = ln_.numpy()
                loss_matrix_n[numerator,batch]=ln
            else:
                raise NotImplementedError
            numerator+=1
            
    
    Estoi_arr_n=1-loss_matrix_n

Estoi_arr_w=1-loss_matrix_w

Sum_estoi_n=np.sum(Estoi_arr_n)

Sum_estoi_w=np.sum(Estoi_arr_w)

Estoi_score_n= Sum_estoi_n/2082

Estoi_score_w= Sum_estoi_w/2082
    
            # check value and gradient
    #print('loss: ', loss_matrix)
    
    #print('grad: ', grad)
    #print('')
    loss_save_path=DATADIR +'/4layers/'+experiment+ '/'+'loss_'+Noise_type+SNR+Level+'.npy'
    np.save(loss_save_path,loss_matrix)
    Estoi_arr=1-loss_matrix[2,:]
    Sum_estoi=np.sum(Estoi_arr)
    Estoi_score= Sum_estoi/((loss_matrix.shape[1]*2)/3)
    Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
    np.save(Estoi_save_path,Estoi_score)

    
if __name__ == '__main__':
    test()
    
    
    
    
    
#np_true_batch=np_true[batch,1:] #for the code to run for each example we have separately
                #so at the end we can take a mean over all the estoi metrics
#np_pred_batch=np_pred[batch,:,0]
#batch=0

 



#####make the pandas dataframe to plot:
#with spike
    
"""import pandas as pd
import os
import numpy as np

DATADIR = os.path.curdir + r'/../data'
Noise_type='wh_'
Level='65'
SNR='5_'
experiment='WithSpike'

loss_save_path=DATADIR +'/4layers/'+experiment+ '/'+'loss_'+Noise_type+SNR+Level+'.npy'
loss_matrix=np.load(loss_save_path)
Estoi_arr=1-loss_matrix[1,:]
Sum_estoi=np.sum(Estoi_arr)
estoi_wh_5_65= Sum_estoi/((loss_matrix.shape[1]*2)/3)
Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
np.save(Estoi_save_path,estoi_wh_5_65)


#Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
#estoi_wh_5_65=np.load(Estoi_save_path)

Noise_type='wh_'
Level='65'
SNR='15_'


loss_save_path=DATADIR +'/4layers/'+experiment+ '/'+'loss_'+Noise_type+SNR+Level+'.npy'
loss_matrix=np.load(loss_save_path)
Estoi_arr=1-loss_matrix[1,:]
Sum_estoi=np.sum(Estoi_arr)
estoi_wh_15_65= Sum_estoi/((loss_matrix.shape[1]*2)/3)
Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
np.save(Estoi_save_path,estoi_wh_15_65)


#Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
#estoi_wh_15_65=np.load(Estoi_save_path)

Noise_type='wh_'
SNR='15_'
Level='55'

loss_save_path=DATADIR +'/4layers/'+experiment+ '/'+'loss_'+Noise_type+SNR+Level+'.npy'
loss_matrix=np.load(loss_save_path)
Estoi_arr=1-loss_matrix[1,:]
Sum_estoi=np.sum(Estoi_arr)
estoi_wh_15_55= Sum_estoi/((loss_matrix.shape[1]*2)/3)
Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
np.save(Estoi_save_path,estoi_wh_15_55)



#Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
#estoi_wh_15_55_NoSpike=np.load(Estoi_save_path)


Noise_type='wh_'
SNR='5_'
Level='55'

loss_save_path=DATADIR +'/4layers/'+experiment+ '/'+'loss_'+Noise_type+SNR+Level+'.npy'
loss_matrix=np.load(loss_save_path)
Estoi_arr=1-loss_matrix[1,:]
Sum_estoi=np.sum(Estoi_arr)
estoi_wh_5_55= Sum_estoi/((loss_matrix.shape[1]*2)/3)
Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
np.save(Estoi_save_path,estoi_wh_5_55)



#Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
#estoi_wh_5_55_NoSpike=np.load(Estoi_save_path)


Noise_type='Na_'
SNR='5_'
Level='55'

loss_save_path=DATADIR +'/4layers/'+experiment+ '/'+'loss_'+Noise_type+SNR+Level+'.npy'
loss_matrix=np.load(loss_save_path)
Estoi_arr=1-loss_matrix[1,:]
Sum_estoi=np.sum(Estoi_arr)
estoi_Na_5_55= Sum_estoi/((loss_matrix.shape[1]*2)/3)
Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
np.save(Estoi_save_path,estoi_Na_5_55)



#Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
#estoi_Na_5_55_NoSpike=np.load(Estoi_save_path)

Noise_type='Na_'
SNR='15_'
Level='55'

loss_save_path=DATADIR +'/4layers/'+experiment+ '/'+'loss_'+Noise_type+SNR+Level+'.npy'
loss_matrix=np.load(loss_save_path)
Estoi_arr=1-loss_matrix[1,:]
Sum_estoi=np.sum(Estoi_arr)
estoi_Na_15_55= Sum_estoi/((loss_matrix.shape[1]*2)/3)
Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
np.save(Estoi_save_path,estoi_Na_15_55)
#
#Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
#estoi_Na_15_55_NoSpike=np.load(Estoi_save_path)

Noise_type='Na_'
SNR='15_'
Level='65'

loss_save_path=DATADIR +'/4layers/'+experiment+ '/'+'loss_'+Noise_type+SNR+Level+'.npy'
loss_matrix=np.load(loss_save_path)
Estoi_arr=1-loss_matrix[1,:]
Sum_estoi=np.sum(Estoi_arr)
estoi_Na_15_65= Sum_estoi/((loss_matrix.shape[1]*2)/3)
Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
np.save(Estoi_save_path,estoi_Na_15_65)


#Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
#estoi_Na_15_65_NoSpike=np.load(Estoi_save_path)


Noise_type='Na_'
SNR='5_'
Level='65'

loss_save_path=DATADIR +'/4layers/'+experiment+ '/'+'loss_'+Noise_type+SNR+Level+'.npy'
loss_matrix=np.load(loss_save_path)
Estoi_arr=1-loss_matrix[1,:]
Sum_estoi=np.sum(Estoi_arr)
estoi_Na_5_65= Sum_estoi/((loss_matrix.shape[1]*2)/3)
Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
np.save(Estoi_save_path,estoi_Na_5_65)


#Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
#estoi_Na_5_65_NoSpike=np.load(Estoi_save_path)



Noise_type='NS_'
SNR='5_'
Level='55'

loss_save_path=DATADIR +'/4layers/'+experiment+ '/'+'loss_'+Noise_type+SNR+Level+'.npy'
loss_matrix=np.load(loss_save_path)
Estoi_arr=1-loss_matrix[1,:]
Sum_estoi=np.sum(Estoi_arr)
estoi_NS_5_55= Sum_estoi/((loss_matrix.shape[1]*2)/3)
Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
np.save(Estoi_save_path,estoi_NS_5_55)

#Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
#estoi_NS_5_55_NoSpike=np.load(Estoi_save_path)

Noise_type='NS_'
SNR='15_'
Level='55'

loss_save_path=DATADIR +'/4layers/'+experiment+ '/'+'loss_'+Noise_type+SNR+Level+'.npy'
loss_matrix=np.load(loss_save_path)
Estoi_arr=1-loss_matrix[1,:]
Sum_estoi=np.sum(Estoi_arr)
estoi_NS_15_55= Sum_estoi/((loss_matrix.shape[1]*2)/3)
Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
np.save(Estoi_save_path,estoi_NS_15_55)

#Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
#estoi_NS_15_55_NoSpike=np.load(Estoi_save_path)

Noise_type='NS_'
SNR='15_'
Level='65'

loss_save_path=DATADIR +'/4layers/'+experiment+ '/'+'loss_'+Noise_type+SNR+Level+'.npy'
loss_matrix=np.load(loss_save_path)
Estoi_arr=1-loss_matrix[1,:]
Sum_estoi=np.sum(Estoi_arr)
estoi_NS_15_65= Sum_estoi/((loss_matrix.shape[1]*2)/3)
Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
np.save(Estoi_save_path,estoi_NS_15_65)


#Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
#estoi_NS_15_65_NoSpike=np.load(Estoi_save_path)


Noise_type='NS_'
SNR='5_'
Level='65'

loss_save_path=DATADIR +'/4layers/'+experiment+ '/'+'loss_'+Noise_type+SNR+Level+'.npy'
loss_matrix=np.load(loss_save_path)
Estoi_arr=1-loss_matrix[1,:]
Sum_estoi=np.sum(Estoi_arr)
estoi_NS_5_65= Sum_estoi/((loss_matrix.shape[1]*2)/3)
Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
np.save(Estoi_save_path,estoi_NS_5_65)

#Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
#estoi_NS_5_65_NoSpike=np.load(Estoi_save_path)



##no spike
import pandas as pd
import os
import numpy as np

DATADIR = os.path.curdir + r'/../data'
Noise_type='wh_'
Level='65'
SNR='5_'
experiment='NoSpike'

Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
estoi_wh_5_65_NoSpike=np.load(Estoi_save_path)

Noise_type='wh_'
Level='65'
SNR='15_'

Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
estoi_wh_15_65_NoSpike=np.load(Estoi_save_path)

Noise_type='wh_'
SNR='15_'
Level='55'

Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
estoi_wh_15_55_NoSpike=np.load(Estoi_save_path)


Noise_type='wh_'
SNR='5_'
Level='55'

Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
estoi_wh_5_55_NoSpike=np.load(Estoi_save_path)


Noise_type='Na_'
SNR='5_'
Level='55'

Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
estoi_Na_5_55_NoSpike=np.load(Estoi_save_path)

Noise_type='Na_'
SNR='15_'
Level='55'

Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
estoi_Na_15_55_NoSpike=np.load(Estoi_save_path)

Noise_type='Na_'
SNR='15_'
Level='65'

Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
estoi_Na_15_65_NoSpike=np.load(Estoi_save_path)


Noise_type='Na_'
SNR='5_'
Level='65'

Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
estoi_Na_5_65_NoSpike=np.load(Estoi_save_path)



Noise_type='NS_'
SNR='5_'
Level='55'

Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
estoi_NS_5_55_NoSpike=np.load(Estoi_save_path)

Noise_type='NS_'
SNR='15_'
Level='55'

Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
estoi_NS_15_55_NoSpike=np.load(Estoi_save_path)

Noise_type='NS_'
SNR='15_'
Level='65'

Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
estoi_NS_15_65_NoSpike=np.load(Estoi_save_path)


Noise_type='NS_'
SNR='5_'
Level='65'

Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
estoi_NS_5_65_NoSpike=np.load(Estoi_save_path)





#########
models_names = ['convolutional']
n_models = len(models_names)

# with spikes as input
metrics_w = ['ns_5_55_w', 'w_5_55_w', 'na_5_55_w',
           'ns_15_55_w', 'w_15_55_w', 'na_15_55_w',
           'ns_5_65_w', 'w_5_65_w', 'na_5_65_w',
           'ns_15_65_w', 'w_15_65_w', 'na_15_65_w',
           ]

# without spikes as input
metrics_wo = ['ns_5_55_wo', 'w_5_55_wo', 'na_5_55_wo',
           'ns_15_55_wo', 'w_15_55_wo', 'na_15_55_wo',
           'ns_5_65_wo', 'w_5_65_wo', 'na_5_65_wo',
           'ns_15_65_wo', 'w_15_65_wo', 'na_15_65_wo',
           ]
metrics = metrics_wo + metrics_w
n_metrics = len(metrics)

data={metrics_w[0]:estoi_NS_5_55,metrics_w[1]:estoi_wh_5_55,metrics_w[2]:estoi_Na_5_55,
      metrics_w[3]:estoi_NS_15_55,metrics_w[4]:estoi_wh_15_55,metrics_w[5]:estoi_Na_15_55,
      metrics_w[6]:estoi_NS_5_65,metrics_w[7]:estoi_wh_5_65,metrics_w[8]:estoi_Na_5_65,
      metrics_w[9]:estoi_NS_15_65,metrics_w[10]:estoi_wh_15_65,metrics_w[11]:estoi_Na_15_65,
      metrics_wo[0]:estoi_NS_5_55_NoSpike, metrics_wo[1]:estoi_wh_5_55_NoSpike,  
      metrics_wo[2]:estoi_Na_5_55_NoSpike,
      metrics_wo[3]:estoi_NS_15_55_NoSpike,metrics_wo[4]:estoi_wh_15_55_NoSpike, 
      metrics_wo[5]:estoi_Na_15_55_NoSpike,
      metrics_wo[6]:estoi_NS_5_65_NoSpike, metrics_wo[7]:estoi_wh_5_65_NoSpike,  
      metrics_wo[8]:estoi_Na_5_65_NoSpike,
      metrics_wo[9]:estoi_NS_15_65_NoSpike,metrics_wo[10]:estoi_wh_15_65_NoSpike,
      metrics_wo[11]:estoi_Na_15_65_NoSpike}
df=pd.DataFrame(data,columns=metrics,index=models_names)
import matplotlib.pyplot as plt

plot_bar(metrics_w,metrics_wo,df)

def plot_bar(metrics_w,metrics_wo,df):
    metrics_names = [m[:-2] for m in metrics_w]


    fig, axs = plt.subplots(4, 3,
                            figsize=(8, 8), sharex='all', sharey='all',
                            gridspec_kw={'hspace': 0, 'wspace': 0})
    
    
    
    
    
    for m in metrics_names:
        noise_type = m.split('_')[0]
        snr = m.split('_')[1]
        level = m.split('_')[2]
    
        if noise_type == 'ns':
            column = 0
        elif noise_type == 'w':
            column = 1
        elif noise_type == 'na':
            column = 2
    
        if '_5_55' in m:
            row = 0
        elif '_15_55' in m:
            row = 1
        elif '_5_65' in m:
            row = 2
        elif '_15_65' in m:
            row = 3
    
        df[[m + '_w', m + '_wo']].plot(ax=axs[row, column], kind='bar', rot=16, legend=False)
    
    
    fig.suptitle("Title for whole figure", fontsize=16)
    
    cols = ['ns', 'w', 'na']
    rows = ['5 SNR 55 level', '15 SNR 55 level', '5 SNR 65 level', '15 SNR 65 level']
    
    for ax, col in zip(axs[0], cols):
        ax.set_title(col)
    
    for ax, row in zip(axs[:,0], rows):
        ax.set_ylabel(row, rotation=90, size='large')
    


"""















#clean_res=np.zeros(shape=(833,49633,1))
#for i in range(833):
    #clean_res[i,:,0]=resample_signal(clean[i,:,0],1.47,kind='nearest')
    
#snd_res=np.zeros(shape=(833,49633,1))
#for i in range(833):
    #snd_res[i,:,0]=resample_signal(snd[i,:,0],1.47,kind='nearest')
#DATADIR='./data'
def test():

  
    #loss_matrix=np.zeros((3,768)) #the matrix at the end, having the three losses in it
    #loss_matrix_w=np.zeros((1,2082))
    #loss_matrix_n=np.zeros((1,2082))
    loss_matrix_n=np.zeros((1,833))
    #np_true = np.random.rand(batch_size, time_steps, features).astype(np.float32)
    #np_pred = np.random.rand(batch_size, time_steps, features).astype(np.float32)
    np_pred_n=snd_res
    np_true=clean_res
    #np_pred_w=prediction_old_filmv1
    np_pred_n=prediction_res#_old_noSpike
    for batch in range(np_true.shape[0]):
        print(batch)
        np_true_batch=np_true[batch:batch+1,:,:] #for the code to run for each example we have separately
                #so at the end we can take a mean over all the estoi metrics
        #np_pred_batch_w=np_pred_w[batch:batch+1,:,:]
        np_pred_batch_n=np_pred_n[batch:batch+1,:,:]
        
        np_true_batch,np_pred_batch_n=removeSilentFrames(np_true_batch,np_pred_batch_n,40,256,128)
        #np_true_batch,np_pred_batch_n=removeSilentFrames(np_true,np_pred_n,40,256,128)

        batch_size = np_true_batch.shape[0]
        nbf=(np_true_batch.shape[1]/128)-1#nbf=(31900/128)-1
        loss_1 = si_sdr_loss  # works tf1, tf2
        loss_2 = estoi_loss(batch_size=batch_size, nbf=nbf,fs=10000) # works tf2 but strange dependency with batch size and timesteps
        loss_3 = stoi_loss(batch_size=batch_size, nbf=nbf) # works tf2
        loss_4 = stsa_mse # works tf1, tf2
        loss_5 = pmsqe_log_mse_loss(batch_size=batch_size) # doesn't work: need .mat tf1
        numerator=0
        for loss in [loss_3]: #[loss_1, loss_2, loss_3]:
            #lw_ = loss(np_true_batch.astype(np.float32), np_pred_batch_w.astype(np.float32))
            ln_ = loss(np_true_batch.astype(np.float32), np_pred_batch_n.astype(np.float32))
    
    
            if tf.__version__[:2] == '1.':
                grad = tf.gradients(ln_, [np_pred_batch_n, np_true_batch])
                sess = tf.Session()
                with sess.as_default():
                    try:
                        #lw = lw_.eval()
                        ln = ln_.eval()
                    except:
                        ln = ln_[0].eval()
                        #l_w = lw_[0].eval()
                    #grad = [g.eval() for g in grad]
                #loss_matrix_w[numerator,batch]=lw
                loss_matrix_n[numerator,batch]=ln
            elif tf.__version__[:2] == '2.':
                grad = 'hi'
                #lw = lw_.numpy()
                #loss_matrix_w[numerator,batch]=lw
                ln = ln_.numpy()
                loss_matrix_n[numerator,batch]=ln
            else:
                raise NotImplementedError
            numerator+=1
            
 
    
 
## for with spikes
    np_pred_w=prediction_old_filmv1
    for batch in range(np_true.shape[0]):
        print(batch)
        np_true_batch=np_true[batch:batch+1,:,:] #for the code to run for each example we have separately
                #so at the end we can take a mean over all the estoi metrics
        np_pred_batch_w=np_pred_w[batch:batch+1,:,:]
        batch_size = np_true_batch.shape[0]
        nbf=(np_true_batch.shape[1]/128)-1#nbf=(31900/128)-1
        loss_1 = si_sdr_loss  # works tf1, tf2
        loss_2 = estoi_loss(batch_size=batch_size, nbf=nbf,fs=44100/3) # works tf2 but strange dependency with batch size and timesteps
        loss_3 = stoi_loss(batch_size=batch_size, nbf=nbf) # works tf2
        loss_4 = stsa_mse # works tf1, tf2
        loss_5 = pmsqe_log_mse_loss(batch_size=batch_size) # doesn't work: need .mat tf1
        numerator=0
        for loss in [loss_2]: #[loss_1, loss_2, loss_3]:
            lw_ = loss(np_true_batch.astype(np.float32), np_pred_batch_w.astype(np.float32))
    
    
            if tf.__version__[:2] == '1.':
                grad = tf.gradients(l_, [np_pred_batch, np_true_batch])
                sess = tf.Session()
                with sess.as_default():
                    try:
                        l_w = lw_.eval()
                    except:
                        lw = lw_[0].eval()
                    #grad = [g.eval() for g in grad]
                loss_matrix_w[numerator,batch]=lw
            elif tf.__version__[:2] == '2.':
                grad = 'hi'
                lw = lw_.numpy()
                loss_matrix_w[numerator,batch]=lw
            else:
                raise NotImplementedError
            numerator+=1
            
            
            
            
            
Estoi_arr_n=1-loss_matrix_n

Estoi_arr_w=1-loss_matrix_w

Sum_estoi_n=np.sum(Estoi_arr_n)

Sum_estoi_w=np.sum(Estoi_arr_w)

stoi_score_n= Sum_estoi_n/833

Estoi_score_w= Sum_estoi_w/833
    
            # check value and gradient
    #print('loss: ', loss_matrix)
    
    #print('grad: ', grad)
    #print('')
    loss_save_path=DATADIR +'/4layers/'+experiment+ '/'+'loss_'+Noise_type+SNR+Level+'.npy'
    np.save(loss_save_path,loss_matrix)
    Estoi_arr=1-loss_matrix[2,:]
    Sum_estoi=np.sum(Estoi_arr)
    Estoi_score= Sum_estoi/((loss_matrix.shape[1]*2)/3)
    Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
    np.save(Estoi_save_path,Estoi_score)

    
if __name__ == '__main__':
    test()
    
    
    
    
    
#np_true_batch=np_true[batch,1:] #for the code to run for each example we have separately
                #so at the end we can take a mean over all the estoi metrics
#np_pred_batch=np_pred[batch,:,0]
#batch=0

 



#####make the pandas dataframe to plot:
#with spike
    
"""import pandas as pd
import os
import numpy as np

DATADIR = os.path.curdir + r'/../data'
Noise_type='wh_'
Level='65'
SNR='5_'
experiment='WithSpike'

loss_save_path=DATADIR +'/4layers/'+experiment+ '/'+'loss_'+Noise_type+SNR+Level+'.npy'
loss_matrix=np.load(loss_save_path)
Estoi_arr=1-loss_matrix[1,:]
Sum_estoi=np.sum(Estoi_arr)
estoi_wh_5_65= Sum_estoi/((loss_matrix.shape[1]*2)/3)
Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
np.save(Estoi_save_path,estoi_wh_5_65)


#Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
#estoi_wh_5_65=np.load(Estoi_save_path)

Noise_type='wh_'
Level='65'
SNR='15_'


loss_save_path=DATADIR +'/4layers/'+experiment+ '/'+'loss_'+Noise_type+SNR+Level+'.npy'
loss_matrix=np.load(loss_save_path)
Estoi_arr=1-loss_matrix[1,:]
Sum_estoi=np.sum(Estoi_arr)
estoi_wh_15_65= Sum_estoi/((loss_matrix.shape[1]*2)/3)
Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
np.save(Estoi_save_path,estoi_wh_15_65)


#Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
#estoi_wh_15_65=np.load(Estoi_save_path)

Noise_type='wh_'
SNR='15_'
Level='55'

loss_save_path=DATADIR +'/4layers/'+experiment+ '/'+'loss_'+Noise_type+SNR+Level+'.npy'
loss_matrix=np.load(loss_save_path)
Estoi_arr=1-loss_matrix[1,:]
Sum_estoi=np.sum(Estoi_arr)
estoi_wh_15_55= Sum_estoi/((loss_matrix.shape[1]*2)/3)
Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
np.save(Estoi_save_path,estoi_wh_15_55)



#Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
#estoi_wh_15_55_NoSpike=np.load(Estoi_save_path)


Noise_type='wh_'
SNR='5_'
Level='55'

loss_save_path=DATADIR +'/4layers/'+experiment+ '/'+'loss_'+Noise_type+SNR+Level+'.npy'
loss_matrix=np.load(loss_save_path)
Estoi_arr=1-loss_matrix[1,:]
Sum_estoi=np.sum(Estoi_arr)
estoi_wh_5_55= Sum_estoi/((loss_matrix.shape[1]*2)/3)
Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
np.save(Estoi_save_path,estoi_wh_5_55)



#Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
#estoi_wh_5_55_NoSpike=np.load(Estoi_save_path)


Noise_type='Na_'
SNR='5_'
Level='55'

loss_save_path=DATADIR +'/4layers/'+experiment+ '/'+'loss_'+Noise_type+SNR+Level+'.npy'
loss_matrix=np.load(loss_save_path)
Estoi_arr=1-loss_matrix[1,:]
Sum_estoi=np.sum(Estoi_arr)
estoi_Na_5_55= Sum_estoi/((loss_matrix.shape[1]*2)/3)
Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
np.save(Estoi_save_path,estoi_Na_5_55)



#Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
#estoi_Na_5_55_NoSpike=np.load(Estoi_save_path)

Noise_type='Na_'
SNR='15_'
Level='55'

loss_save_path=DATADIR +'/4layers/'+experiment+ '/'+'loss_'+Noise_type+SNR+Level+'.npy'
loss_matrix=np.load(loss_save_path)
Estoi_arr=1-loss_matrix[1,:]
Sum_estoi=np.sum(Estoi_arr)
estoi_Na_15_55= Sum_estoi/((loss_matrix.shape[1]*2)/3)
Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
np.save(Estoi_save_path,estoi_Na_15_55)
#
#Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
#estoi_Na_15_55_NoSpike=np.load(Estoi_save_path)

Noise_type='Na_'
SNR='15_'
Level='65'

loss_save_path=DATADIR +'/4layers/'+experiment+ '/'+'loss_'+Noise_type+SNR+Level+'.npy'
loss_matrix=np.load(loss_save_path)
Estoi_arr=1-loss_matrix[1,:]
Sum_estoi=np.sum(Estoi_arr)
estoi_Na_15_65= Sum_estoi/((loss_matrix.shape[1]*2)/3)
Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
np.save(Estoi_save_path,estoi_Na_15_65)


#Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
#estoi_Na_15_65_NoSpike=np.load(Estoi_save_path)


Noise_type='Na_'
SNR='5_'
Level='65'

loss_save_path=DATADIR +'/4layers/'+experiment+ '/'+'loss_'+Noise_type+SNR+Level+'.npy'
loss_matrix=np.load(loss_save_path)
Estoi_arr=1-loss_matrix[1,:]
Sum_estoi=np.sum(Estoi_arr)
estoi_Na_5_65= Sum_estoi/((loss_matrix.shape[1]*2)/3)
Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
np.save(Estoi_save_path,estoi_Na_5_65)


#Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
#estoi_Na_5_65_NoSpike=np.load(Estoi_save_path)



Noise_type='NS_'
SNR='5_'
Level='55'

loss_save_path=DATADIR +'/4layers/'+experiment+ '/'+'loss_'+Noise_type+SNR+Level+'.npy'
loss_matrix=np.load(loss_save_path)
Estoi_arr=1-loss_matrix[1,:]
Sum_estoi=np.sum(Estoi_arr)
estoi_NS_5_55= Sum_estoi/((loss_matrix.shape[1]*2)/3)
Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
np.save(Estoi_save_path,estoi_NS_5_55)

#Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
#estoi_NS_5_55_NoSpike=np.load(Estoi_save_path)

Noise_type='NS_'
SNR='15_'
Level='55'

loss_save_path=DATADIR +'/4layers/'+experiment+ '/'+'loss_'+Noise_type+SNR+Level+'.npy'
loss_matrix=np.load(loss_save_path)
Estoi_arr=1-loss_matrix[1,:]
Sum_estoi=np.sum(Estoi_arr)
estoi_NS_15_55= Sum_estoi/((loss_matrix.shape[1]*2)/3)
Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
np.save(Estoi_save_path,estoi_NS_15_55)

#Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
#estoi_NS_15_55_NoSpike=np.load(Estoi_save_path)

Noise_type='NS_'
SNR='15_'
Level='65'

loss_save_path=DATADIR +'/4layers/'+experiment+ '/'+'loss_'+Noise_type+SNR+Level+'.npy'
loss_matrix=np.load(loss_save_path)
Estoi_arr=1-loss_matrix[1,:]
Sum_estoi=np.sum(Estoi_arr)
estoi_NS_15_65= Sum_estoi/((loss_matrix.shape[1]*2)/3)
Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
np.save(Estoi_save_path,estoi_NS_15_65)


#Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
#estoi_NS_15_65_NoSpike=np.load(Estoi_save_path)


Noise_type='NS_'
SNR='5_'
Level='65'

loss_save_path=DATADIR +'/4layers/'+experiment+ '/'+'loss_'+Noise_type+SNR+Level+'.npy'
loss_matrix=np.load(loss_save_path)
Estoi_arr=1-loss_matrix[1,:]
Sum_estoi=np.sum(Estoi_arr)
estoi_NS_5_65= Sum_estoi/((loss_matrix.shape[1]*2)/3)
Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
np.save(Estoi_save_path,estoi_NS_5_65)

#Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
#estoi_NS_5_65_NoSpike=np.load(Estoi_save_path)



##no spike
import pandas as pd
import os
import numpy as np

DATADIR = os.path.curdir + r'/../data'
Noise_type='wh_'
Level='65'
SNR='5_'
experiment='NoSpike'

Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
estoi_wh_5_65_NoSpike=np.load(Estoi_save_path)

Noise_type='wh_'
Level='65'
SNR='15_'

Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
estoi_wh_15_65_NoSpike=np.load(Estoi_save_path)

Noise_type='wh_'
SNR='15_'
Level='55'

Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
estoi_wh_15_55_NoSpike=np.load(Estoi_save_path)


Noise_type='wh_'
SNR='5_'
Level='55'

Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
estoi_wh_5_55_NoSpike=np.load(Estoi_save_path)


Noise_type='Na_'
SNR='5_'
Level='55'

Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
estoi_Na_5_55_NoSpike=np.load(Estoi_save_path)

Noise_type='Na_'
SNR='15_'
Level='55'

Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
estoi_Na_15_55_NoSpike=np.load(Estoi_save_path)

Noise_type='Na_'
SNR='15_'
Level='65'

Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
estoi_Na_15_65_NoSpike=np.load(Estoi_save_path)


Noise_type='Na_'
SNR='5_'
Level='65'

Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
estoi_Na_5_65_NoSpike=np.load(Estoi_save_path)



Noise_type='NS_'
SNR='5_'
Level='55'

Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
estoi_NS_5_55_NoSpike=np.load(Estoi_save_path)

Noise_type='NS_'
SNR='15_'
Level='55'

Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
estoi_NS_15_55_NoSpike=np.load(Estoi_save_path)

Noise_type='NS_'
SNR='15_'
Level='65'

Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
estoi_NS_15_65_NoSpike=np.load(Estoi_save_path)


Noise_type='NS_'
SNR='5_'
Level='65'

Estoi_save_path=DATADIR +'/4layers/'+experiment+ '/'+'Estoi_score_'+Noise_type+SNR+Level+'.npy'
estoi_NS_5_65_NoSpike=np.load(Estoi_save_path)





#########
models_names = ['convolutional']
n_models = len(models_names)

# with spikes as input
metrics_w = ['ns_5_55_w', 'w_5_55_w', 'na_5_55_w',
           'ns_15_55_w', 'w_15_55_w', 'na_15_55_w',
           'ns_5_65_w', 'w_5_65_w', 'na_5_65_w',
           'ns_15_65_w', 'w_15_65_w', 'na_15_65_w',
           ]

# without spikes as input
metrics_wo = ['ns_5_55_wo', 'w_5_55_wo', 'na_5_55_wo',
           'ns_15_55_wo', 'w_15_55_wo', 'na_15_55_wo',
           'ns_5_65_wo', 'w_5_65_wo', 'na_5_65_wo',
           'ns_15_65_wo', 'w_15_65_wo', 'na_15_65_wo',
           ]
metrics = metrics_wo + metrics_w
n_metrics = len(metrics)

data={metrics_w[0]:estoi_NS_5_55,metrics_w[1]:estoi_wh_5_55,metrics_w[2]:estoi_Na_5_55,
      metrics_w[3]:estoi_NS_15_55,metrics_w[4]:estoi_wh_15_55,metrics_w[5]:estoi_Na_15_55,
      metrics_w[6]:estoi_NS_5_65,metrics_w[7]:estoi_wh_5_65,metrics_w[8]:estoi_Na_5_65,
      metrics_w[9]:estoi_NS_15_65,metrics_w[10]:estoi_wh_15_65,metrics_w[11]:estoi_Na_15_65,
      metrics_wo[0]:estoi_NS_5_55_NoSpike, metrics_wo[1]:estoi_wh_5_55_NoSpike,  
      metrics_wo[2]:estoi_Na_5_55_NoSpike,
      metrics_wo[3]:estoi_NS_15_55_NoSpike,metrics_wo[4]:estoi_wh_15_55_NoSpike, 
      metrics_wo[5]:estoi_Na_15_55_NoSpike,
      metrics_wo[6]:estoi_NS_5_65_NoSpike, metrics_wo[7]:estoi_wh_5_65_NoSpike,  
      metrics_wo[8]:estoi_Na_5_65_NoSpike,
      metrics_wo[9]:estoi_NS_15_65_NoSpike,metrics_wo[10]:estoi_wh_15_65_NoSpike,
      metrics_wo[11]:estoi_Na_15_65_NoSpike}
df=pd.DataFrame(data,columns=metrics,index=models_names)
import matplotlib.pyplot as plt

plot_bar(metrics_w,metrics_wo,df)

def plot_bar(metrics_w,metrics_wo,df):
    metrics_names = [m[:-2] for m in metrics_w]


    fig, axs = plt.subplots(4, 3,
                            figsize=(8, 8), sharex='all', sharey='all',
                            gridspec_kw={'hspace': 0, 'wspace': 0})
    
    
    
    
    
    for m in metrics_names:
        noise_type = m.split('_')[0]
        snr = m.split('_')[1]
        level = m.split('_')[2]
    
        if noise_type == 'ns':
            column = 0
        elif noise_type == 'w':
            column = 1
        elif noise_type == 'na':
            column = 2
    
        if '_5_55' in m:
            row = 0
        elif '_15_55' in m:
            row = 1
        elif '_5_65' in m:
            row = 2
        elif '_15_65' in m:
            row = 3
    
        df[[m + '_w', m + '_wo']].plot(ax=axs[row, column], kind='bar', rot=16, legend=False)
    
    
    fig.suptitle("Title for whole figure", fontsize=16)
    
    cols = ['ns', 'w', 'na']
    rows = ['5 SNR 55 level', '15 SNR 55 level', '5 SNR 65 level', '15 SNR 65 level']
    
    for ax, col in zip(axs[0], cols):
        ax.set_title(col)
    
    for ax, row in zip(axs[:,0], rows):
        ax.set_ylabel(row, rotation=90, size='large')
    


"""

'''
#for this to work, cd in the folder containing this code
DATADIR = os.path.curdir + r'/../data'
Noise_type='NS_'
Level='55'
SNR='5_'
experiment='NoSpike'
content = [d for d in os.listdir(DATADIR) if 'npy' in d]
    print(content)
    path_true=DATADIR +'/4layers/'+experiment+ '/clean_'+Level+'.npy'
    np_true_1 = np.load(path_true)[:].astype(np.float32)
    np_true=np.zeros(shape=(np_true_1.shape[0]*np_true_1.shape[1],np_true_1.shape[2],1))
    np_true=np.reshape(np_true_1,(np_true_1.shape[0]*np_true_1.shape[1],np_true_1.shape[2],1))
    np_true=np_true[:,1:,:]
    #np_pred = np.load(DATADIR + '/prediction_NoSpike.npy')[36+36*z:37+36*z, :, np.newaxis].astype(np.float32)
    path_pred=DATADIR + '/4layers/'+experiment+ '/prediction_'+Noise_type+SNR+Level+'.npy'
    np_pred = np.load(path_pred)[:, :,:].astype(np.float32)
    print('real data shapes: ', np_true.shape, np_pred.shape)'''
    
    
    
    
    
'''