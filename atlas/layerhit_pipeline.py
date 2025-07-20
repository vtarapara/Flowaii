
# 1. How low is the loss with 100 collisions?
import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
from filelock import FileLock

##########
# some schema stuff
##########
#calo layers go from 0 (EMB1) to 15(FCAL2), plus 16(was 999). Entry layers are calo layer + offset; highest valid value is thus 33.
OFFSET=17
NUMCHITS=16
HASGPU=False
from itertools import product
xcols = [s.format(i) for i,s in product(range(NUMCHITS),["xx_{}","xy_{}","xz_{}"])]
pcols = [s.format(i) for i,s in product(range(NUMCHITS),["xpx_{}","xpy_{}","xpz_{}"])]
lcols = ["layer{0}".format(i) for i in range(NUMCHITS)]

# parameters (tunable ones adjusted below)
parset = {'maxHits' : 16, #maximum per track, not in detector
          'num_hit_categories' : OFFSET*2, #number of detector hit categories      
          "num_features" : 8,
          #### tuned parameters ####
          "batch_size":32,            
          "units" : 64,          
          "layers" : 3,
          #### learning rate ####   
          "learning_rate":0.01,      
          "cycle" : 20,             
          "alph"  : 0.0,            
          "timemult" : 1.1,         
          "ratemult" : 0.95,       
          #### loss issues ####
          'weights' : {'calo':4., 'pos':0.004, 'mom':0.001},  #should these be tuned?
          ### testing
          'nTracks' : 4096,
          "num_epochs":20,  
         }


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
def bt_classes(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        '''requires Imputed and Concatted data. returns a batch augmented with the multi-hot encoded calo layer target'''
        layerEnc = tf.keras.layers.CategoryEncoding(num_tokens=OFFSET*2+1, output_mode="multi_hot", sparse=False)
        batch['calo_encoded'] = layerEnc(batch['calo_samps'])[:,:-1] #chop off the last column, which is NAN
        del batch["calo_samps"]
        return batch

def get_preprocessor():
    """Make a chain of content-independent preprocessors.  Feature normalization is not included because we want the model to do that.  
       Category encoding is not done here since the batch_mapper is deprecated. 
    """
    noNoneLayers = SimpleImputer(columns=lcols,strategy="constant",fill_value=OFFSET*2)
    noNoneValues = SimpleImputer(columns=xcols+pcols,strategy="constant",fill_value=0.) #is there a better training value for this?  
    #these guys drop columns on materialize()
    groupFeatures  = Concatenator(include=['q','pt','d0','z0','eta','sinphi','cosphi','ipt'],output_column_name="features")
    groupPositions = Concatenator(include=xcols,output_column_name="positions")
    groupMomenta   = Concatenator(include=pcols,output_column_name="momenta")
    groupLayers    = Concatenator(include=lcols,output_column_name="calo_samps")

    chained_pp = Chain(noNoneLayers,noNoneValues,groupFeatures,groupPositions,groupLayers,groupMomenta)#,BatchMapper(bt_classes, batch_format="numpy"))
    return chained_pp


### this is content-dependent input processing, pre-tuned in the following way:
# ds=pandas.read_csv('texas_layers_as.csv')[['q','pt','d0','z0','eta','sinphi','cosphi','ipt']]
# normalizer=tf.keras.layers.Normalization(name="Normalization",axis=-1)
# normalizer.adapt(ds.values); print(normalizer.mean, normalizer.variance)
def get_static_normalizer():
    norm_means = [[ 0., 1.8665882e+00, -1.5916437e-02, 1.8371961e-01, 0,              0.,  0.,  7.6952296e-01]]
    norm_var   = [[ 1., 8.8588257e+00, 2.5001748e+00,  1.9079940e+03, 1.9128140e+00,  0.5, 0.5, 9.2223175e-02]]
    return tf.keras.layers.Normalization(name="Normalization",mean=norm_means, variance=norm_var)
 

from ray.data import read_csv
def load_global_data(inputfile): 
  full_ds=read_csv(inputfile)
  return full_ds.train_test_split(test_size=0.2) # splitting is materializing. 


# create the model
def create_model(nFeatures, nUnits, nLayers, num_hit_categories, maxHits):
    trackin = tf.keras.layers.Input(shape=(nFeatures,))
    norm = get_static_normalizer()(trackin)
    L = norm #tf.keras.layers.Dropout(0.2,name='InputReg')(norm)
    for _ in range(nLayers):
        L = tf.keras.layers.Dense(nUnits, activation='relu')(L)
    c = tf.keras.layers.Dense(num_hit_categories,activation='sigmoid')(L) 
    L = tf.keras.layers.Concatenate()([c, L])                 #let positions and momenta see which layers are predicted
    c = tf.keras.layers.Flatten(name="calo_encoded")(c)       #OUTPUT
    p = tf.keras.layers.Dense(maxHits*3,name="position_enc")(L)  #OUTPUT
    d = tf.keras.layers.Dense(maxHits*3,name="momentum_enc")(L)    #OUTPUT
    #build the model. The next line must be consistent with the input/output layers above and with the loss!
    model = tf.keras.models.Model(trackin, [c,p,d])
    return model


from ray import train
from ray.train.tensorflow.keras import ReportCheckpointCallback
from tensorflow import keras
from keras.callbacks import TensorBoard
def train_on_worker(config: dict):
    '''Use this in ray distributed training. Consumes shards.'''
    batch_size = config.get("batch_size",16)
    batch_size = batch_size * train.get_context().get_world_size()
    lr = config.get("lr",0.05)
    epochs = config.get("num_epochs",5)
    nFeatures = config.get("num_features",8)
    nUnits = config.get("units",64)
    nLayers = config.get("layers",3)
    num_hit_categories = config.get("num_hit_categories",34)
    maxHits = config.get("maxHits",16)

    # Get the Dataset shard for this data parallel worker
    train_data = train.get_dataset_shard("train")
    val_data = train.get_dataset_shard("val")
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    with strategy.scope():
        multi_worker_model = create_model(nFeatures,nUnits,nLayers=nLayers,num_hit_categories=num_hit_categories,maxHits=maxHits)
        multi_worker_model.compile(
           optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
           loss=[tf.keras.losses.BinaryCrossentropy(),tf.keras.losses.MeanSquaredError(),tf.keras.losses.MeanSquaredError()],
           loss_weights={'calo_encoded': config['weights']['calo'], 'positions': config['weights']['pos'], 'momenta': config['weights']['mom']},
           metrics={"calo_encoded": tf.keras.losses.BinaryCrossentropy(), "positions": tf.keras.losses.MeanSquaredError(),  "momenta": tf.keras.losses.MeanSquaredError() }
        )
    # load data. setting local shuffle-buffer size enables TFData shuffle on each iteration. 
    tf_dataset = train_data.to_tf(feature_columns="features", label_columns=["positions","momenta","calo_encoded"], batch_size=batch_size, local_shuffle_buffer_size=2*batch_size) 
    tf_val = val_data.to_tf(feature_columns="features", label_columns=["positions","momenta","calo_encoded"], batch_size=batch_size, local_shuffle_buffer_size=2*batch_size) 
    # run the training loop.
    hist = multi_worker_model.fit(
            tf_dataset,
            validation_data=tf_val,
            epochs=epochs,
            callbacks=[ReportCheckpointCallback(report_metrics_on="epoch_end"), TensorBoard(log_dir='~/ray_results/tblogs',write_graph=False,update_freq='epoch')],
            verbose=0,
    )
    results = hist.history
    return results

# simple tool that sends training jobs to workers
from ray.train import ScalingConfig
from ray.train.tensorflow import TensorflowTrainer
def just_train_model(nTracks=1024,nWorkers=4):
    train_ds, test_ds = load_global_data("../texas_layers_as.csv")  
    preprocessor = get_preprocessor()
    train_ds = preprocessor.fit_transform(train_ds).limit(nTracks).map_batches(bt_classes) 
    test_ds = preprocessor.fit_transform(test_ds).limit(int(nTracks/4)).map_batches(bt_classes) 

    trainer = TensorflowTrainer(train_loop_per_worker=train_on_worker, train_loop_config=parset,
                                scaling_config=ScalingConfig( num_workers=nWorkers, use_gpu=HASGPU, trainer_resources={"CPU": 1}),
                                datasets={"train": train_ds, "val": test_ds})
    best_result = trainer.fit()
    print(f"Last result: {best_result.metrics}")
    return best_result
    pass


class TuneReporterCallback(tf.keras.callbacks.Callback):
    """Custom callback"""
    def __init__(self, logs={}):
        self.iteration = 0
        super(TuneReporterCallback, self).__init__()
    def on_epoch_end(self, batch, logs={}):
        self.iteration += 1
        tune.report(keras_info=logs, mean_loss=logs.get("loss"))


import os
def train_with_tf(config: dict):
    '''Use this in tuning jobs. only difference to worker job is that it does not shard.'''
    batch_size = config.get("batch_size",16)
    lr = config.get("lr",0.05)
    epochs = config.get("num_epochs",5)
    nFeatures = config.get("num_features",8)
    nUnits = config.get("units",64)
    nLayers = config.get("layers",3)
    num_hit_categories = config.get("num_hit_categories",34)
    maxHits = config.get("maxHits",16)

    #Tuning uses all the data.
    with FileLock(os.path.expanduser("~/.data.lock")):
        train_ds, test_ds = load_global_data("../texas_layers_as.csv")
    preprocessor = get_preprocessor()
    train_ds = preprocessor.fit_transform(train_ds).limit(config.get("nTracks",100)) #short for testing 
    train_ds = train_ds.map_batches(bt_classes) 

    smodel = create_model(nFeatures,nUnits,nLayers,num_hit_categories,maxHits)
    smodel.compile( optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                    loss=[tf.keras.losses.BinaryCrossentropy(),tf.keras.losses.MeanSquaredError(),tf.keras.losses.MeanSquaredError()],
                    loss_weights={'calo_encoded': config['weights']['calo'], 'positions': config['weights']['pos'], 'momenta': config['weights']['mom']},
                    metrics={"calo_encoded": tf.keras.losses.BinaryCrossentropy(),
                             "positions": tf.keras.losses.MeanSquaredError(),  
                             "momenta": tf.keras.losses.MeanSquaredError() }
                )           
    # run it  

    tf_dataset=train_ds.to_tf(feature_columns="features", label_columns=["positions","momenta","calo_encoded"]) #this looks like a slow start on each worker
    smodel.fit(tf_dataset, batch_size=batch_size, epochs=epochs, verbose=0, callbacks=[ReportCheckpointCallback(report_metrics_on="epoch_end"),TuneReporterCallback])
    pass



def train_with_tftrainer(config: dict):
    '''Use this in tuning jobs.'''
    batch_size = config.get("batch_size",16)
    lr = config.get("lr",0.05)
    epochs = config.get("num_epochs",5)
    nFeatures = config.get("num_features",8)
    nUnits = config.get("units",64)
    nLayers = config.get("layers",3)
    num_hit_categories = config.get("num_hit_categories",34)
    maxHits = config.get("maxHits",16)

    #Tuning uses all the data.
    with FileLock(os.path.expanduser("~/.data.lock")):
        train_ds, test_ds = load_global_data("../texas_layers_as.csv")
    preprocessor = get_preprocessor()    
    train_ds = preprocessor.fit_transform(train_ds).limit(config.get("nTracks",1024)).map_batches(bt_classes) 
    test_ds = preprocessor.fit_transform(test_ds).limit(config.get("nTracks",1024)).map_batches(bt_classes) 

    trainer = TensorflowTrainer(train_loop_per_worker=train_on_worker, train_loop_config=parset,
                                scaling_config=ScalingConfig( num_workers=2, use_gpu=HASGPU, trainer_resources={"CPU": 1}),
                                datasets={"train": train_ds, "val": test_ds})
    best_result = trainer.fit()
    pass


from ray import tune
### parameters for tuning 
tparset = parset.copy()
#tparset['learning_rate']=tune.uniform(0.001, 0.1)
tparset['units']=tune.choice([16,32,64,128])

from ray.tune.schedulers import AsyncHyperBandScheduler
def tune_layers(samples=2):
    sched = AsyncHyperBandScheduler(
        time_attr="training_iteration", max_t=20, grace_period=5
    )

    tuner = tune.Tuner(
        tune.with_resources(train_with_tftrainer, resources={"cpu": 1, "gpu": 0}),
        tune_config=tune.TuneConfig(
            metric="positions_loss",
            mode="min",
            scheduler=sched,
            num_samples=samples,
        ),
        run_config=train.RunConfig(
            name="exp",
            stop={"positions_loss": 1000.},
        ),
        param_space=tparset
    )
    results = tuner.fit()
    print("Best hyperparameters found were: ", results.get_best_result().config)
    return results
