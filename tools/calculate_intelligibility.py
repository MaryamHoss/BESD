## code to calculate speech quality and intelligibility  using different metrics


import numpy as np
import os
import sys
sys.path.append('../')
sys.path.append('../..')
from TrialsOfNeuralVocalRecon.tools.utils.losses import *
import tensorflow as tf
from tools.nice_tools import *
from pesq import pesq
from pystoi import stoi

Fs_stoi=10000
Fs_pesq=8000 #used to be 16000

def find_intel(true,pred,metric='pesq',fs=14700): 
#to do: add an statement that checks the sizes of true and pred-done
#to do: make a repository for estoi. For now the tf version is fine

    if true.shape!=pred.shape:
        raise Exception('True and prediction must have the same shape'+'found shapes {} and {}'.format(true.shape,pred.shape))
    if true.shape[0]!=1 or pred.shape[0]!=1:
        raise Exception('Inputs must have the first dimension equal to 1')
    if metric=='pesq':
        if fs!=Fs_pesq:
            true=resample_signal(np.squeeze(true),fs/Fs_pesq,kind='nearest')
            pred=resample_signal(np.squeeze(pred),fs/Fs_pesq,kind='nearest')

        #true=np.squeeze(true)
        #pred=np.squeeze(pred)
        true_batch,pred_batch=removeSilentFrames(true[np.newaxis],pred[np.newaxis],40,256,128) #I was not removing silent frames for PESQ before
        true=np.squeeze(true_batch)
        pred=np.squeeze(pred_batch)
        out_metric=pesq(Fs_pesq,true,pred,'nb')
        
    elif metric=='stoi':
        if fs!=Fs_stoi:
            true=resample_signal(np.squeeze(true),fs/Fs_stoi,kind='nearest')
            true=true[np.newaxis]
            pred=resample_signal(np.squeeze(pred),fs/Fs_stoi,kind='nearest')
            pred=pred[np.newaxis]
            
        true_batch,pred_batch=removeSilentFrames(true,pred,40,256,128)
        true_batch=np.squeeze(true_batch)
        pred_batch=np.squeeze(pred_batch)
        out_metric=stoi(true_batch, pred_batch, Fs_stoi, extended=False)
        
        
    elif metric=='estoi':
        if fs!=Fs_stoi:
            true=resample_signal(np.squeeze(true),fs/Fs_stoi,kind='nearest')
            true=true[np.newaxis]
            pred=resample_signal(np.squeeze(pred),fs/Fs_stoi,kind='nearest')
            pred=pred[np.newaxis]
            
        true_batch,pred_batch=removeSilentFrames(true,pred,40,256,128)
            
        
        nbf=(true_batch.shape[1]/128)-1
        batch_size=true_batch.shape[0]
        loss = estoi_loss(batch_size=batch_size, nbf=nbf)
        l_ = loss(true_batch.astype(np.float32), pred_batch.astype(np.float32))
 
        if tf.__version__[:2] == '1.':
            grad = tf.gradients(l_, [pred_batch, true_batch])
            sess = tf.Session()
            with sess.as_default():
                try:
                    
                    l = l_.eval()
                except:
                    l = l_[0].eval()
                   
            out_metric=ln
        elif tf.__version__[:2] == '2.':
            grad = 'hi'
           
            l = l_.numpy()
            out_metric=l
            
        out_metric=1-out_metric
        
    elif metric=='si-sdr':
        
            
        true_batch,pred_batch=removeSilentFrames(true,pred,40,256,128)
        #true_batch=np.squeeze(true_batch)
        #pred_batch=np.squeeze(pred_batch)
        
        loss = si_sdr_loss
        l_ = loss(true_batch.astype(np.float32), pred_batch.astype(np.float32))
 
        if tf.__version__[:2] == '1.':
            grad = tf.gradients(l_, [pred_batch, true_batch])
            sess = tf.Session()
            with sess.as_default():
                try:
                    
                    l = l_.eval()
                except:
                    l = l_[0].eval()
                   
            out_metric=ln
        elif tf.__version__[:2] == '2.':
            grad = 'hi'
           
            l = l_.numpy()
            out_metric=l
            
        out_metric=-out_metric
    elif metric=='si-sdr-mes':
        
            
        #true_batch,pred_batch=removeSilentFrames(true,pred,40,256,128)
        #true_batch=np.squeeze(true_batch)
        #pred_batch=np.squeeze(pred_batch)
        
        out_metric=calc_sdr(pred,true)       
   
              
              
              
              
    elif metric=='mfcc':
                   
        #true_batch,pred_batch=removeSilentFrames(true,pred,40,256,128)
        #true_batch=np.squeeze(true_batch)
        #pred_batch=np.squeeze(pred_batch)
        
        loss = mfcc_loss(fs)
        l_ = loss(true.astype(np.float32), pred.astype(np.float32))
 
        if tf.__version__[:2] == '1.':
            grad = tf.gradients(l_, [pred_batch, true_batch])
            sess = tf.Session()
            with sess.as_default():
                try:
                    
                    l = l_.eval()
                except:
                    l = l_[0].eval()
                   
            out_metric=ln
        elif tf.__version__[:2] == '2.':
            grad = 'hi'
           
            l = l_.numpy()
            out_metric=l
            
        out_metric=out_metric
        
   
    elif metric=='seg_SNR':
        
            
        #true_batch,pred_batch=removeSilentFrames(true,pred,40,256,128)
        #true_batch=np.squeeze(true_batch)
        #pred_batch=np.squeeze(pred_batch)
        
        loss = segSNR_loss(fs)
        l_ = loss(true.astype(np.float32), pred.astype(np.float32))
 
        if tf.__version__[:2] == '1.':
            grad = tf.gradients(l_, [pred_batch, true_batch])
            sess = tf.Session()
            with sess.as_default():
                try:
                    
                    l = l_.eval()
                except:
                    l = l_[0].eval()
                   
            out_metric=ln
        elif tf.__version__[:2] == '2.':
            grad = 'hi'
           
            l = l_.numpy()
            out_metric=l
            
        out_metric=-out_metric
        
        
    elif metric=='stsa-mse':
        
        
        true_batch,pred_batch=removeSilentFrames(true,pred,40,256,128)
            
        
        
        loss = stsa_mse
        l_ = loss(true_batch.astype(np.float32), pred_batch.astype(np.float32))
 
        if tf.__version__[:2] == '1.':
            grad = tf.gradients(l_, [pred_batch, true_batch])
            sess = tf.Session()
            with sess.as_default():
                try:
                    
                    l = l_.eval()
                except:
                    l = l_[0].eval()
                   
            out_metric=ln
        elif tf.__version__[:2] == '2.':
            grad = 'hi'
           
            l = l_.numpy()
            out_metric=l
            
        out_metric=out_metric
    
    
    else:
        print('no')
        raise NotImplementedError   
    
    return out_metric
        
            
            
     
