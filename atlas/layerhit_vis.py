import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import pandas as pd
import tensorflow_datasets as tfds


def compare_to_athena(px,y,outname="result",images_dir=".",names=[]):
  '''Compare any column(s) to the ATHENA prediction.
     - py: ai, 
     - y:  athena
     - outname: output file tag
     - names: can be ?
  '''
  twocols = ['tab:purple','tab:orange']
  def plot_corr(_x,_y,label,squeeze,ax,fig):
    if "r_" in label:  ax.scatter(_y,_x,label=label,alpha=0.1,c=twocols['p' in label]) 
    else: h = ax.hist2d(_y,_x, norm=colors.LogNorm(vmin=1, vmax=len(y))) #,'.',label=label,alpha=0.1,c=twocols['p' in label]) #switching these around
    tru =np.linspace(np.min(_y),np.max(_y))
    ax.plot(tru,tru,'k-',label="ideal")
    if not squeeze: 
      ax.set_ylabel('athena')
      if not "r_" in label: fig.colorbar(h[3],ax=ax)
    ax.set_xlabel('flowaii')
    #ax.legend()
  def plot_hist(_x,_y,label,squeeze,ax):
     d= _y-_x
     N=d.shape[0]
     ax.hist(d,bins=int(N/40)+10,color=twocols['p' in label], lw=0,density=True)
     m= np.mean(d)
     s= np.std(d)
     ax.set_title(label)   
     ax.set_yscale("log")  
     if not squeeze: ax.set_ylabel('tracks/{}'.format(N))
     ax.set_xlabel('$\mu$:{:.1%}; $\sigma$:{:0.2f}'.format(m/(s/np.sqrt(N)),s))
     #print(label,":    bias={:0.2f} \pm {:0.2f}; std={:0.2f}".format(m,s/np.sqrt(N),s) )
  nc=7
  nplots=y.shape[1] 
  nrows = nplots//nc
  if (nplots % nc != 0) : nrows = nrows + 1
  fig = plt.figure(figsize=(nc*4,nrows*5),layout="constrained") 
  sfigs = fig.subfigures(2,1)
  ax1 = sfigs[0].subplots(nrows,nc) 
  ax2 = sfigs[1].subplots(nrows,nc)   
  for n,ax in enumerate(ax1.flat):
    if n<nplots:  plot_corr(y[:,n],px[:,n],names[n],bool(n),ax,sfigs[0])
  for n,ax in enumerate(ax2.flat):
    if n<nplots: plot_hist(y[:,n],px[:,n],names[n],bool(n),ax)
  plt.savefig(f"{images_dir}/eval_{outname}.png")       
  

import pandas as pd
import layerhit_schema as lhs

def get_df(pars = lhs.parset):
  '''Read file, make a dataframe, adding the radius of layers.
     When read from a file the encoding is indexed, not 1hot
  '''
  df = pd.read_csv(pars['filename'], delimiter=',')
  expressions= ['r_{0} = xx_{0}**2 + xy_{0}**2'.format(i) for i in range(lhs.NUMCHITS)]
  df.eval('\n'.join(expressions),inplace=True)  
  #print(df.schema())
  for i in range(lhs.NUMCHITS):
      r='r_{}'.format(i)
      df[r]= df[r].apply(np.sqrt)
  return df


def plot_detector(df,xyview=True,axi=None):
    '''
    in the read-csv, the nth calolayer traversed has its ID *stored in* layerN.  nans are not plotted.
    '''
    for i in range(lhs.NUMCHITS):
        l = 'l_{}'.format(i) 
        if xyview : df.plot.scatter(x='xx_{}'.format(i),y='xy_{}'.format(i),s=0.1,c='layer{}'.format(i),colormap=lhs.CALCMAP,ax=axi,subplots=True,colorbar=False)
        else : df.plot.scatter(x='xz_{}'.format(i),y='r_{}'.format(i),s=0.1,c='layer{}'.format(i),colormap=lhs.CALCMAP,ax=axi,subplots=True,colorbar=False)

def plot_learned_detector(pdf,xyview=True,axi=None):
  '''
  in the prediction, the calolayers are stored by index.
  the meaning of layer
  '''
  import matplotlib.pyplot as plt
  for i in range(lhs.LAYERCODES):
    if lhs.SMALLWHENWRONG:
      ess,alf=0.1*pdf['l{}'.format(i)],1
    else:
      ess,alf=0.1,np.abs(pdf['l{}'.format(i)]-1)  
    if xyview : 
      axi.scatter(x=pdf['x{}'.format(i)], y=pdf['y{}'.format(i)],s=ess, alpha=alf, color=lhs.getcolor(i))
      axi.set_xlim(-4000, 4000)
      axi.set_ylim(-4000, 4000)
    else : 
      axi.scatter(x=pdf['z{}'.format(i)],y=pdf['r{}'.format(i)],s=ess, alpha=alf, color=lhs.getcolor(i))
      axi.set_xlim(-6300,6300)
      axi.set_ylim(0,4000)
  pass

def plot_atlas(df, learned=False):
    if not learned: plotter = plot_detector
    else: plotter= plot_learned_detector
    fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(12,4))
    plotter(df,True,axi=axes[0])
    plotter(df,False,axi=axes[1])

    pass

#import sys
#print(sys.path)
from layerhit_model import get_model,get_preprocessor,bt_classes
from layerhit_train import load_global_data
from ray.data.preprocessors import Concatenator

# make names in prediction slightly different from in truth (truth has underscore)
from itertools import product
xcs = [s.format(i) for i,s in product(range(lhs.LAYERCODES),["x{}","y{}","z{}"])]
pcs = [s.format(i) for i,s in product(range(lhs.LAYERCODES),["px{}","py{}","pz{}"])]
lcs = ["l{0}".format(i) for i in range(lhs.LAYERCODES)]

def get_predicted_frame(test_ds = None, pars = lhs.parset):
  '''
  note that this frame has layer positions/momenta represented in a per-layer array. Plotting is different!
  ''' 
  texas = get_model(pars['preload'])
  if (test_ds == None):
    _, test_ds = load_global_data(pars['filename'])
    preprocessor = get_preprocessor()
    test_ds = preprocessor.fit_transform(test_ds).limit(pars['nTracks']).map_batches(bt_classes) 
  tf_dataset=test_ds.to_tf(feature_columns="features", label_columns=["calo_encoded","position_enc","momentum_enc"])
  #map to breakup the options dataset
  #feature_dataset = tf_dataset.map(lambda a,_: a)
  ptem = texas.predict(tf_dataset)
  d = pd.concat((pd.DataFrame(ptem[0],columns=lcs ),
        pd.DataFrame(ptem[1],columns=xcs ),  
        pd.DataFrame(ptem[2],columns=pcs )),axis=1)
  #produce radius layers
  expressions= ['r{0} = x{0}**2 + y{0}**2'.format(i) for i in range(lhs.LAYERCODES)]
  d.eval('\n'.join(expressions),inplace=True)  
  for i in range(lhs.LAYERCODES):
      r='r{}'.format(i)
      d[r]= d[r].apply(np.sqrt)
  return d






