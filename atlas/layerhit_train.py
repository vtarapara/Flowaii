from typing import Dict, Optional
from ray import train
from ray.train.tensorflow.keras import ReportCheckpointCallback
from tensorflow import keras, distribute
from keras.callbacks import TensorBoard

from layerhit_model import get_preprocessor, create_model, bt_classes, get_model
from layerhit_schema import LABELCOLUMNS

#def skip_invalid(row): return 'skip' #would be good to count'em
#from pyarrow import csv
#po = {'parse_options': csv.ParseOptions(invalid_row_handler=skip_invalid)}
# TODO: clean_csv

def load_global_data(inputfile,nTracks=-1,split=0.25): 
  from ray.data import read_csv 
  full_ds=read_csv(inputfile)
  if nTracks < 0:  return full_ds.train_test_split(split) # splitting is materializing. 
  return full_ds.limit(nTracks).train_test_split(split)

def load_global_data_nosplit(inputfile,nTracks=-1):
    from ray.data import read_csv 
    full_ds=read_csv(inputfile)
    if nTracks < 0:  return full_ds
    return full_ds.limit(nTracks)

def get_configured(config):
    batch_size = config.get("batch_size",16)
    lr = config.get("lr",0.05)
    epochs = config.get("num_epochs",50)
    return batch_size,lr,epochs


#### USE ME FOR TRAINING!
def train_on_worker(config: dict):
    '''Use this in ray distributed training. Consumes shards.'''
    batch_size,lr,epochs = get_configured(config)
    batch_size = batch_size * train.get_context().get_world_size()
    # Get the Dataset shard for this data parallel worker
    train_data = train.get_dataset_shard("train")
    val_data = train.get_dataset_shard("val")
    strategy = distribute.MultiWorkerMirroredStrategy()
    with strategy.scope():
        multi_worker_model = create_model(**config)
        multi_worker_model.compile(
           optimizer=keras.optimizers.Adam(learning_rate=lr),
           loss=[keras.losses.BinaryCrossentropy(),keras.losses.MeanSquaredError(),keras.losses.Huber()],
           loss_weights={'calo_encoded': config['weights']['calo'], 'position_enc': config['weights']['pos'], 'momentum_enc': config['weights']['mom']},
           metrics={"calo_encoded": [keras.losses.BinaryCrossentropy(), keras.metrics.BinaryAccuracy()],
                             "position_enc": keras.losses.MeanSquaredError(),  
                             "momentum_enc": keras.losses.MeanSquaredError() }
        )
    # load data. setting local shuffle-buffer size enables TFData shuffle on each iteration. 
    # tbCallBack = TensorBoard(log_dir='TBLOGS',update_freq='epoch',write_graph=True, write_images=False)
    tf_dataset = train_data.to_tf(feature_columns="features", label_columns=LABELCOLUMNS, 
                                  batch_size=batch_size, local_shuffle_buffer_size=2*batch_size) 
    tf_val = val_data.to_tf(feature_columns="features", label_columns=LABELCOLUMNS, 
                            batch_size=batch_size, local_shuffle_buffer_size=2*batch_size) 
    # run the training loop.
    hist = multi_worker_model.fit(tf_dataset,validation_data=tf_val,epochs=epochs,verbose=0,
                                  callbacks=[ReportCheckpointCallback(checkpoint_on="train_end",report_metrics_on="epoch_end")])
    results = hist.history
    return results

#### USE ME FOR TUNING!
def train_in_job(config: dict):
    '''Like train_on_worker, but for parallel tuning
        - does not consume shards
        - uses lock to access input file
        - does not checkpoint every time
    '''
    batch_size,lr,epochs = get_configured(config)
    #Tuning uses all the data.
    with FileLock(os.path.expanduser("~/.data.lock")): 
        train_ds, test_ds = load_global_data(config['filename'],config.get("nTracks",128))
    preprocessor = get_preprocessor()
    train_ds = preprocessor.fit_transform(train_ds).limit(config.get("nTracks",128)) 
    train_ds = train_ds.map_batches(bt_classes) 
    smodel = create_model(**config)
    smodel.compile( optimizer=keras.optimizers.Adam(learning_rate=lr),
                    loss=[keras.losses.BinaryCrossentropy(),keras.losses.MeanSquaredError(),keras.losses.Huber()],
                    loss_weights={'calo_encoded': config['weights']['calo'], 'position_enc': config['weights']['pos'], 'momentum_enc': config['weights']['mom']},
                    metrics={"calo_encoded": [keras.losses.BinaryCrossentropy(), keras.metrics.BinaryAccuracy()],
                             "position_enc": keras.losses.MeanSquaredError(),  
                             "momentum_enc": keras.losses.MeanSquaredError() }
                )           
    tf_dataset=train_ds.to_tf(feature_columns="features", label_columns=LABELCOLUMNS) #this looks like a slow start on each worker
    
    for epoch in range(epochs):
        hist = smodel.fit(tf_dataset, batch_size=batch_size, epochs=1, verbose=0, callbacks=[ReportCheckpointCallback(checkpoint_on="train_end")])
        train.report({"classification_x" : hist.history["calo_encoded_loss"][0], 
                      "accuracy_x"  : hist.history["calo_encoded_binary_accuracy"][0],
                      "momentum_x"  : hist.history["momentum_enc_loss"][0],
                      "position_x"  : hist.history["position_enc_loss"][0],
                      "totalloss_x" : hist.history["loss"][0]})
    #print("HISTORY\n\n",hist.history) #debugging line 
    train.report(metrics={"classification_x" : hist.history["calo_encoded_loss"][0], 
                          "accuracy_x"  : hist.history["calo_encoded_binary_accuracy"][0],
                          "momentum_x"  : hist.history["momentum_enc_loss"][0],
                          "position_x"  : hist.history["position_enc_loss"][0],
                          "loss" : hist.history["loss"][0]
                          },
                 checkpoint=train.Checkpoint.from_directory(train.get_context().get_trial_dir()))    
    pass

import layerhit_schema as lhs
# simple tool that sends training jobs to workers
from ray.train import ScalingConfig
from ray.train.tensorflow import TensorflowTrainer
def run_tensorflowtrainer(config: dict):
    '''calls train_on_worker using ray scaling'''
    train_ds, test_ds = load_global_data(config['filename'])  
    preprocessor = get_preprocessor()
    nTracks = config.get("nTracks",1024)
    train_ds = preprocessor.fit_transform(train_ds).limit(nTracks).map_batches(bt_classes) 
    test_ds = preprocessor.fit_transform(test_ds).limit(int(nTracks/4)).map_batches(bt_classes) 
    trainer = TensorflowTrainer(train_loop_per_worker=train_on_worker, train_loop_config=config,
                                scaling_config=ScalingConfig( num_workers=2, use_gpu=lhs.HASGPU, trainer_resources={"CPU": 1}),
                                datasets={"train": train_ds, "val": test_ds})
    best_result = trainer.fit()
    print(f"Last result: {best_result.metrics}")
    return best_result
    pass

import os
from filelock import FileLock
def run_tensorflowtrainer_tune(config: dict):
    '''Like run_tensorflowtrainer, but made for parallel tuning jobs:
       - no extra args; filename passed by config
       - does not consume shards
       - expects to compete with other workers for the input file'''
    with FileLock(os.path.expanduser("~/.data.lock")):
        train_ds, test_ds = load_global_data(config['filename'])
    preprocessor = get_preprocessor()    
    train_ds = preprocessor.fit_transform(train_ds).limit(config.get("nTracks",1024)).map_batches(bt_classes) 
    test_ds = preprocessor.fit_transform(test_ds).limit(config.get("nTracks",1024)).map_batches(bt_classes) 
    trainer = TensorflowTrainer(train_loop_per_worker=train_in_job, train_loop_config=config,
                                scaling_config=ScalingConfig( num_workers=2, use_gpu=lhs.HASGPU, trainer_resources={"CPU": 1}),
                                datasets={"train": train_ds, "val": test_ds})
    best_result = trainer.fit()
    print(f"Last result: {best_result.metrics}")
    return best_result
    pass


from ray import tune
class TuneReporterCallback(keras.callbacks.Callback):
    """Custom callback"""
    def __init__(self, logs={}):
        self.iteration = 0
        super(TuneReporterCallback, self).__init__()
    def on_epoch_end(self, batch, logs={}):
        self.iteration += 1
        tune.report(keras_info=logs, mean_loss=logs.get("loss"))


def zeroloss(y_true,y_pred):
    return 0*y_true

class flowaii_trainable(tune.Trainable):
    # class to use for bigger trainings
    def _get_tf_dataset(config:dict):
        with FileLock(os.path.expanduser("~/.data.lock")): 
            ds = load_global_data_nosplit(config['filename'],config.get("nTracks",128))
        preprocessor = get_preprocessor()
        ds = preprocessor.fit_transform(ds) #.limit(config.get("nTracks",128)) why limit again?
        ds = ds.map_batches(bt_classes) 
        train_ds,test_ds = ds.train_test_split(0.125)
        return train_ds.to_tf(feature_columns="features", label_columns=LABELCOLUMNS),test_ds.to_tf(feature_columns="features", label_columns=LABELCOLUMNS)
    def setup(self, config: dict):
        # assign self.train and test data (test data not used yet)
        # get a compiled model in self.model
        self.batch_size,lr,epochs = get_configured(config)
        #Tuning uses all the data.
        self.tf_dataset,self.tf_test = flowaii_trainable._get_tf_dataset(config)
        smodel = create_model(**config)
        losses =  [keras.losses.BinaryCrossentropy(),keras.losses.MeanSquaredError(),keras.losses.Huber()]
        if config.get('justclass'): losses = [keras.losses.BinaryCrossentropy(),zeroloss,zeroloss ]

        smodel.compile( optimizer=keras.optimizers.Adam(learning_rate=lr),
                        loss=losses,
                        loss_weights={'calo_encoded': config['weights']['calo'], 'position_enc': config['weights']['pos'], 'momentum_enc': config['weights']['mom']},
                        metrics={"calo_encoded": [keras.losses.BinaryCrossentropy(), keras.metrics.BinaryAccuracy()],
                             "position_enc": keras.losses.MeanSquaredError(),  
                             "momentum_enc": keras.losses.MeanSquaredError()}
                    )        
        self.model = smodel
        pass
    def reset_config(self,config:dict):
        # reset the model (without having to reload the data) if reusing actors.

        self.batch_size,lr,epochs = get_configured(config)
        smodel = create_model(**config)
        losses =  [keras.losses.BinaryCrossentropy(),keras.losses.MeanSquaredError(),keras.losses.Huber()]
        if config.get('justclass'): losses = [keras.losses.BinaryCrossentropy(),zeroloss,zeroloss ]
        smodel.compile( optimizer=keras.optimizers.Adam(learning_rate=lr),
                        loss=losses,
                        loss_weights={'calo_encoded': config['weights']['calo'], 'position_enc': config['weights']['pos'], 'momentum_enc': config['weights']['mom']},
                        metrics={"calo_encoded": [keras.losses.BinaryCrossentropy(), keras.metrics.BinaryAccuracy()],
                             "position_enc": keras.losses.MeanSquaredError(),  
                             "momentum_enc": keras.losses.MeanSquaredError()}
                    )        
        self.model = smodel
        pass
    def save_checkpoint(self, checkpoint_dir: str):
        file_path = checkpoint_dir + "/model.keras"
        self.model.save(file_path)
    def load_checkpoint(self, checkpoint_dir: str):
        del self.model
        file_path = checkpoint_dir + "/model.keras"
        self.model = get_model(file_path)
    def step(self):
        hist=self.model.fit(self.tf_dataset, 
                       batch_size=self.batch_size, 
                       epochs=1, 
                       verbose=0, 
                       validation_data=self.tf_test,
                       callbacks=[])
        return {"classification_x" : hist.history["calo_encoded_loss"][0], 
                "accuracy_x"  : hist.history["calo_encoded_binary_accuracy"][0],
                "momentum_x"  : hist.history["momentum_enc_loss"][0],
                "position_x"  : hist.history["position_enc_loss"][0],
                "totalloss_x" : hist.history["loss"][0],
                "classification_v" : hist.history["val_calo_encoded_loss"][0], 
                "totalloss_v" : hist.history["val_loss"][0],
        }

    

