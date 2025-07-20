
# 1. How low is the loss with 100 collisions?
import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
from filelock import FileLock

from layerhit_schema import parset,OFFSET,LAYERCODES,NUMCHITS,lcols,xcols,pcols

###########
# DEFINE PREPROCESSING TOOLS
############

##this is content-independent preprocessing that can happen in batch.
from ray.data.preprocessors import (
    Concatenator,
    SimpleImputer,
    Chain,
)
from typing import Dict
def bt_oldclasses(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        '''requires Imputed and Concatted data. returns the batch, augmented with the multi-hot encoded calo layer target.
           deprecated!
        '''
        layerEnc = tf.keras.layers.CategoryEncoding(num_tokens=OFFSET*2+1, output_mode="multi_hot", sparse=False)
        batch['calo_encoded'] = layerEnc(batch['calo_samps'])[:,:-1] #chop off the last column, which is NAN
        del batch["calo_samps"]
        return batch

def bt_encodetargets(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    '''makes ALL target arrays decipherable with multihot encoding'''
    batchsize = batch["calo_samps"].shape[0]
    layerEnc = tf.keras.layers.CategoryEncoding(num_tokens=LAYERCODES+1, output_mode="multi_hot", sparse=False)# NAN column created in GET_PREPROCESSOR
    posmomEnc = tf.keras.layers.CategoryEncoding(num_tokens=LAYERCODES+1, output_mode="count", sparse=False)
    asweight = lambda a,i : a.reshape(batchsize,NUMCHITS,3)[:,:,i]
    batch['calo_encoded'] = layerEnc(batch['calo_samps'])[:,:-1] #chop off the last column, which was NAN
    x = posmomEnc(batch['calo_samps'],count_weights=asweight(batch['positions'],0))[:,:-1]
    y = posmomEnc(batch['calo_samps'],count_weights=asweight(batch['positions'],1))[:,:-1]
    z = posmomEnc(batch['calo_samps'],count_weights=asweight(batch['positions'],2))[:,:-1]
    batch['position_enc'] = np.stack([x,y,z],axis=-1).reshape(batchsize,LAYERCODES*3) #innermost axis: -1
    x = posmomEnc(batch['calo_samps'],count_weights=asweight(batch['momenta'],0))[:,:-1]
    y = posmomEnc(batch['calo_samps'],count_weights=asweight(batch['momenta'],1))[:,:-1]
    z = posmomEnc(batch['calo_samps'],count_weights=asweight(batch['momenta'],2))[:,:-1]
    batch['momentum_enc'] = np.stack([x,y,z],axis=-1).reshape(batchsize,LAYERCODES*3) #innermost axis: -1
    del batch['positions']
    del batch['momenta']
    del batch['calo_samps']
    return batch

# switch to new-style output encoding
bt_classes=bt_encodetargets

def get_preprocessor():
    """Makes a chain of content-independent preprocessors.  
       Feature normalization is not included, because we want the model to do that.  
       Category encoding is separate function (`bt_classes()`) since batch_mapper is deprecated. 
    """
    noNoneLayers = SimpleImputer(columns=lcols,strategy="constant",fill_value=LAYERCODES)
    noNoneValues = SimpleImputer(columns=xcols+pcols,strategy="constant",fill_value=0.)  #weights are zero for missing layers
    #these guys drop columns on materialize()
    groupFeatures  = Concatenator(include=['q','pt','d0','z0','eta','sinphi','cosphi','ipt'],output_column_name="features")
    groupPositions = Concatenator(include=xcols,output_column_name="positions")
    groupMomenta   = Concatenator(include=pcols,output_column_name="momenta")
    groupLayers    = Concatenator(include=lcols,output_column_name="calo_samps")

    chained_pp = Chain(noNoneLayers,noNoneValues,groupFeatures,groupPositions,groupLayers,groupMomenta)#,BatchMapper(bt_classes, batch_format="numpy"))
    return chained_pp


def get_normalizertune_from_file(filename):
    '''
    Use this function to update get_static_normalizer from a csv file
    (the input columns listed below should be filled once per track).
    '''
    from pandas import read_csv
    ds=read_csv('texas_layers_as.csv')[['q','pt','d0','z0','eta','sinphi','cosphi','ipt']]
    normalizer=tf.keras.layers.Normalization(name="Normalization",axis=-1)
    normalizer.adapt(ds.values)
    return normalizer.mean, normalizer.variance

def get_static_normalizer():
    '''Normalizes the tracks (to trigger test ttbar event distributions)'''
    norm_means = [[ 0., 1.8665882e+00, -1.5916437e-02, 1.8371961e-01, 0,              0.,  0.,  7.6952296e-01]]
    norm_var   = [[ 1., 8.8588257e+00, 2.5001748e+00,  1.9079940e+03, 1.9128140e+00,  0.5, 0.5, 9.2223175e-02]]
    return tf.keras.layers.Normalization(name="Normalization",mean=norm_means, variance=norm_var)
 

# create the model, now formatted to accept the config dict
def create_model(num_features, units, layers, num_hit_categories, maxHits, preload, **kwargs):
    '''
    Current idea: mull over inputs and get the category output
                  take categories and *raw input* and do extrapolation.  
    '''
    if preload: 
        return get_model(preload)
    trackin = tf.keras.layers.Input(shape=(num_features,))
    norm = get_static_normalizer()(trackin)
    L = norm #tf.keras.layers.Dropout(0.2,name='InputReg')(norm)
    for _ in range(layers):
        L = tf.keras.layers.Dense(units, activation='relu')(L)            #stew on inputs
    c = tf.keras.layers.Dense(num_hit_categories,activation='sigmoid')(L) #first, categorize
    L2 = tf.keras.layers.Concatenate()([c, norm])                          #then use categories and inputs
    for _ in range(layers):                           
        L2 = tf.keras.layers.Dense(units+num_hit_categories, activation='relu')(L2) #stew on cat+input
    c = tf.keras.layers.Flatten(name="calo_encoded")(c)                   #OUTPUT
    p = tf.keras.layers.Dense(maxHits*3,name="position_enc")(L2)           #OUTPUT positions
    d = tf.keras.layers.Dense(maxHits*3,name="momentum_enc")(L2)           #OUTPUT momenta
    #build the model. The next line must be consistent with the input/output layers above and with the loss!
    model = tf.keras.models.Model(trackin, [c,p,d])
    return model

def get_model(path_to_model):
    return keras.models.load_model(path_to_model,compile=False)

