##########
# This defines the formatting of the input files, the model parameters, and some training parameters.
##########
# calo layers go from 0 (EMB1) to 15(FCAL2), plus 16(was 999). Entry layers are calo layer + offset; highest valid value is thus 33.
# need similar colors for C, C+17
# need grouped colors for 0,1,2 ; 3,4,5,; 6,7,8,9'; 10,11,12 (like012)

  
# Also useful to know that units are millimeters and MeV. Thus position uncertainty of 1 meter is 1e6 in loss.


### INPUT AND LABEL SCHEMA
OFFSET=17
NUMCHITS=16
LAYERCODES=OFFSET*2
HASGPU=False
from itertools import product
xcols = [s.format(i) for i,s in product(range(NUMCHITS),["xx_{}","xy_{}","xz_{}"])]
pcols = [s.format(i) for i,s in product(range(NUMCHITS),["xpx_{}","xpy_{}","xpz_{}"])]
lcols = ["layer{}".format(i) for i in range(NUMCHITS)]

#### VISUALIZATION CONSTANTs
SMALLWHENWRONG=True
import matplotlib as mpl
import numpy as np
CNORM=mpl.colors.Normalize(vmin=0, vmax=LAYERCODES)

# nice bluish and reddish rainbows, arranged to nearly repeat after 16.
_bco = (mpl.colormaps['viridis'].resampled(34))(np.linspace(0,1,34))[np.hstack((np.arange(17)*2,np.arange(17)*2+1))]
_eco = (mpl.colormaps['spring'].resampled(34))(np.linspace(0,1,34))[np.hstack((np.arange(17)*2,np.arange(17)*2+1))]
_bco[3:10,:] = _eco[3:10,:] #EME,HEC
_bco[20:27,:] = _eco[20:27,:] #EME,HEC
CALCMAP = mpl.colors.ListedColormap(_bco)
getcolor = lambda v: CALCMAP(CNORM(v))


### DATATYPES FOR CONVERSION
import numpy as np
tf_old_dtypes = {'calo_encoded': np.dtype('34f4'), 'positions': np.dtype('48f8'), 'momenta':np.dtype('48f8')}
tf_dtypes = {'calo_encoded': np.dtype('34f4'), 'position_enc': np.dtype('102f8'), 'momentum_enc':np.dtype('102f8')}
numpifyTFTarget = lambda d,k: np.fromiter(d.map(lambda a:a[k]).as_numpy_iterator(),dtype=tf_dtypes[k])

LABELCOLUMNS=["position_enc","momentum_enc","calo_encoded"]

import tensorflow as tf
#PRELOADPATH="/Users/ath11/ray_results/TensorflowTrainer_2023-10-06_12-50-49/TensorflowTrainer_80a96_00000_0_2023-10-06_12-50-49/"
#PRELOADPATH="/Users/ath11/ray_results/TensorflowTrainer_2023-10-12_13-58-12/TensorflowTrainer_e8e23_00000_0_2023-10-12_13-58-12/checkpoint_000000/model.keras"
#PRELOADPATH='/Users/ath11/ray_results/TensorflowTrainer_2023-10-31_09-13-20/TensorflowTrainer_436ff_00000_0_2023-10-31_09-13-20/checkpoint_000000/model.keras'
PRELOADPATH='/Users/ath11/ray_results/TensorflowTrainer_2023-10-31_09-13-20/TensorflowTrainer_436ff_00000_0_2023-10-31_09-13-20/checkpoint_000000/model.keras'
#/Users/ath11/ray_results/two_minds/train_in_job_2b762_00000_0_learning_rate=0.0040_2023-11-28_15-17-35/
# parameters (tunable ones adjusted below)
parset = {'maxHits' : LAYERCODES, #maximum per track, not in detector. Now identical.
          'num_hit_categories' : LAYERCODES, #number of detector hit categories      
          "num_features" : 8,
          'filename' : '../texas_layers_1245.35360200_fixed.csv',
          'preload' : PRELOADPATH,
          #### tuned parameters ####
          "batch_size":32,          
          "num_epochs":200,            
          "units" : 128,          
          "layers" : 2,
          #### learning rate ####   
          "learning_rate":0.006,      
          "cycle" : 50,             
          "alph"  : 0.0,            
          "timemult" : 3,         
          "ratemult" : 0.25,       
          #### loss issues ####
          'justclass': False,
          'weights' : {'calo':1., 'pos':1.e-6, 'mom':1.e-3},  #should these be tuned?
          ### testing
          'nTracks' : 8192,
          #'weightlayers' : [],         #no loss weights for now
          #'extraweight'  : 2,          #weight for tracks hitting these layers is three times the default
          #"verbose" : False
         }
