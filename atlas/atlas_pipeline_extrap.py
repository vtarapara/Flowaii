# A machine learning pipeline for learning where the ATLAS track extrapolator predicts that a charged particle track goes in the calorimeter.
# Tracks bend in the magnetic field depending on their transverse momentum and charge.
#    also, the field is not entirely uniform, so we hope for a model that depends on initial position, charge, momentum, and the extrapolation radius
# Track features currently used:
#   x[0] = layer (integer label for the layer of the calorimeter, related to the extrapolation radius)
#   x[1] = isEntry (is the track entering at that layer? not sure what this means or if it is useful)
#   x[2] = eta (initial eta direction)
#   x[3] = phi (initial phi direction)
#   x[4] = pT  (track transverse momentum, signed by the charge of the track)
# Targets:
#    y[0] = eta_extrap (eta extrapolated to the specified calorimeter layer)
#    y[1] = phi_extrap (phi exrapolated to the specified calorimeter layer)

# notes
#  - CANNOT LEARN IF BATCHED. Batching requires a smaller learning rate to work.
#

# parameters to play with
filename='/Users/ayana/Downloads/ote.txt'
batch_size=1
num_epochs=500
learning_rate=0.08
phionly=False

# Import needed libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

### functions
from math import pi


def etaphi_loss(y_true, y_pred,dloss=[999,pi]):
  '''Loss function for calorimeter coordinate predictions (eta,phi) 
     - Eta (pseudorapidity) ranges from -infty to infty (although practically from -2.5 to 2.5 for tracks)
     - phi (azimuthal angle) ranges from -pi to pi, differences must be modded down to the range (-pi,pi)
  '''
  #use floormod to remove cyclic ambiguity in phi, but don't change eta (which should have absolute value less than 999 always!)
  # note this doesn't behave physically, like np.fmod! Only works for positive arguments. 
  abs_difference = tf.math.floormod(tf.math.abs(y_true - y_pred),dloss)
  return tf.reduce_mean(abs_difference, axis=-1)

def plot_loss(history):
  '''Plot the training history'''
  plt.figure()
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 3])
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend()
  plt.grid(True)
  plt.savefig('training.png')

def plot_hists(px,y,outname="result.png"):
  '''see how well predictions match the target'''
  def plot_corr(_x,_y,label=None):
    plt.plot(_y,_x,"b.",label=label)
    tru =np.linspace(np.min(_x),np.max(_x))
    plt.plot(tru,tru,'k-',label="ideal")  
    plt.xlim([-3.142, 3.142])
    plt.ylabel('predicted')
    plt.xlabel('extrapolated')
    plt.legend()
  plt.figure()
  if y.ndim >1:  
    plt.subplot(121) 
    plot_corr(y[:,0],px[:,0],'eta')
    plt.subplot(122)
    plot_corr(y[:,1],px[:,1],'phi')  
  else:
    plot_corr(y,px,'phi')  
  plt.savefig(outname)

#{'names': ('layer', 'entryQ', 'eta', 'phi', 'pt'),'formats': ('u1', 'u1', 'f4', 'f4', 'f4')}
#'names': ('eta', 'phi'), 'formats': ('f4','f4')
def load_data(filename,batch_size=32,train_split=0.8,seed=42,use_tengev=True,invert_pt=True,use_vec=True):
  '''read a file and make tensorflow datasets from it'''
  features = np.loadtxt(filename,dtype='f4',usecols=(2,3,4,5,8)) #this works for both versions of files
  targets = np.loadtxt(filename,dtype='f4',usecols=(6,7)) 
  if (use_tengev): features = np.divide(features,[1,1,1,1,10000.])
  if (invert_pt) : features[:,4] = 1./features[:,4]
  if (use_vec)   : 
    sc = np.stack((np.sin(features[:,3]),np.cos(features[:,3])),axis=1)
    features = np.concatenate((features,sc),axis=1)
    targets  = np.stack((targets[:,0],np.sin(targets[:,1]),np.cos(targets[:,1])),axis=1)
  dssize=features.shape[0]
  print("Loading",dssize,"tracks.")
  ds=tf.data.Dataset.from_tensor_slices((features,targets))
  train_size=int(train_split*dssize)
  train_ds = ds.take(train_size)
  test_ds = ds.skip(train_size)
  if batch_size is not None:
    train_ds = train_ds.shuffle(buffer_size=batch_size * 8, seed=seed)
    train_ds = train_ds.batch(batch_size)
    test_ds =  test_ds.batch(batch_size)
  else:
    train_ds=train_ds.shuffle(1024,seed=seed) #not sure about need to shuffle here  
    pass
  return train_ds, test_ds   


### here is the training/evaluation of the model
print('\n\n welcome! \n\n')

### get the datasets, massage them into features (x) and regression targets (y)
xy_train, xy_test = load_data(filename,batch_size=batch_size)
x_train,y_train = np.concatenate([x for x, y in xy_train], axis=0),np.concatenate([y for x, y in xy_train], axis=0)
x_test, y_test  = np.concatenate([x for x, y in xy_test], axis=0),np.concatenate([y for x, y in xy_test], axis=0)
print("Size of training data:",y_train.shape[0])

### create a normalization layer (although input ranges should be similar; consider skipping it)
normalizer = tf.keras.layers.Normalization(axis=-1,name="Normalization")
normalizer.adapt(np.array(x_train)) 
print("Normalized feature means:",normalizer.mean.numpy())

#see how well a do-nothing (no extrapolation) model performs
y_etaphi = np.stack((y_test[:,0],np.arctan2(y_test[:,1],y_test[:,2])),axis=1)
plot_hists(x_test[:,2:4],y_etaphi,outname="no_extrap.png") #compare an idiot no-extrapolation model
print(">>> Bias in delta-eta and delta-phi, unextrapolated:",tf.reduce_mean(x_test[:,2:4]-y_etaphi,axis=0).numpy())
print(">>>  RMS of delta-eta and delta-phi, unextrapolated:",tf.math.reduce_std(x_test[:,2:4]-y_etaphi,axis=0).numpy())

### define the model
model = tf.keras.models.Sequential([
  normalizer,
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(3,name="Output")
])
model.summary()
#compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate= tf.keras.optimizers.schedules.CosineDecayRestarts(learning_rate, 10,alpha=0.05)),
              loss='mean_absolute_error',
              metrics=['mse'])

#fit the model!
history = model.fit(x_train, y_train, verbose=0, validation_split=0.2, epochs=num_epochs)              
plot_loss(history)

#how did we do?
print("\n\nEvaluation on test: \n\n")
predictions= model.predict(x_test)
predictions[:,1]=np.arctan2(predictions[:,1],predictions[:,2])
y_etaphi = np.stack((y_test[:,0],np.arctan2(predictions[:,1],predictions[:,2])),axis=1)
plot_hists(predictions,y_etaphi)
model.evaluate(x_test,  y_test, verbose=2)
print(">>> Biases, fitted:",tf.reduce_mean(predictions[:,:2]-y_etaphi,axis=0).numpy())
print(">>>    RMS, fitted:",tf.math.reduce_std(predictions[:,:2]-y_etaphi,axis=0).numpy())