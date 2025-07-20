from ray import train,tune
import layerhit_schema as lhs
from tensorflow import keras
from layerhit_train import run_tensorflowtrainer_tune,train_in_job

lastmodel='/Users/ath11/ray_results/two_minds/flowaii_trainable_8a2e9_00001_1_learning_rate=0.0050_2023-11-28_22-44-03/checkpoint_000000/model.keras'
### parameters for tuning 
tparset = lhs.parset.copy()
tparset['preload']=None #None or not tuning model!
tparset['nTracks']=8192 #65536
tparset['layers']=2
tparset['justclass']=True
#tparset['units']=64#tune.choice([64,128])
#tparset['learning_rate']=tune.choice([0.004, 0.005])
tparset['learning_rate']=0.01
tparset['cycle']=15 # tune.choice([10, 30])
tparset['alpha']=tune.choice([0.3, 0])
tparset['weights']=lhs.parset['weights'].copy()
tparset['weights']['calo']=100
#tparset['weights']['pos']=tune.grid_search([0.0001, 0.0005])
#tparset['weights']['mom']=tune.grid_search([0.0001, 0.0005])
from ray.tune.schedulers import AsyncHyperBandScheduler
from layerhit_train import flowaii_trainable
def tune_layers(samples=4):
    tfdata = flowaii_trainable._get_tf_dataset(tparset)
    sched = AsyncHyperBandScheduler(time_attr="training_iteration", max_t=tparset['num_epochs'], grace_period=50)
    tuner = tune.Tuner(
        flowaii_trainable,
        #tune.with_resources(train_in_job, resources={"cpu": 0.5, "gpu": 0}),
        run_config=train.RunConfig(name='classy',
                                   stop=tune.stopper.TrialPlateauStopper(metric='totalloss_x', num_results=8, grace_period=30),
                                   checkpoint_config=train.CheckpointConfig(checkpoint_at_end=True)),                                
        tune_config=tune.TuneConfig(metric="totalloss_x", mode="min",scheduler=sched, num_samples=samples,
                                    reuse_actors=True),
        param_space=tparset,
        )
    results = tuner.fit()
    print("Best hyperparameters found were: ", results.get_best_result().config)
    return results
