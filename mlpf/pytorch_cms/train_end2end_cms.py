import sys
import os
import math

from comet_ml import Experiment

#Check if the GPU configuration has been provided
try:
    if not ("CUDA_VISIBLE_DEVICES" in os.environ):
        import setGPU
except Exception as e:
    print("Could not import setGPU, running CPU-only")

import torch
use_gpu = torch.cuda.device_count()>0
multi_gpu = torch.cuda.device_count()>1

#define the global base device
if use_gpu:
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

import torch_geometric

import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import EdgeConv, MessagePassing, EdgePooling, GATConv, GCNConv, JumpingKnowledge, GraphUNet, DynamicEdgeConv, DenseGCNConv
from torch_geometric.nn import TopKPooling, SAGPooling, SGConv
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_scatter import scatter_mean
from torch_geometric.nn.inits import reset
from torch_geometric.data import Data, DataLoader, DataListLoader, Batch
from gravnet import GravNetConv

import torch_cluster

from glob import glob
import numpy as np
import os.path as osp
import pickle

import math
import time
import numba
import tqdm
import sklearn
import pandas

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import graph_data_cms
from graph_data_cms import PFGraphDataset, elem_to_id, class_to_id, class_labels

#Ignore divide by 0 errors
np.seterr(divide='ignore', invalid='ignore')

def onehot(a):
    b = np.zeros((a.size, len(class_labels)))
    b[np.arange(a.size),a] = 1
    return b

#Creates the dataframe of predictions given a trained model and a data loader
def prepare_dataframe(model, loader, multi_gpu, device):
    model.eval()
    dfs = []
    dfs_edges = []
    eval_time = 0

    for i, data in enumerate(loader):
        if not multi_gpu:
            data = data.to(device)

        pred_id_onehot, pred_momentum, new_edges = model(data)
        _, pred_id = torch.max(pred_id_onehot, -1)
        pred_momentum[pred_id==0] = 0
        if not multi_gpu:
            data = [data]

        x = torch.cat([d.x.to("cpu") for d in data])
        gen_id = torch.cat([d.y_gen_id.to("cpu") for d in data])
        gen_p4 = torch.cat([d.ygen[:, :4].to("cpu") for d in data])
        cand_id = torch.cat([d.y_candidates_id.to("cpu") for d in data])
        cand_p4 = torch.cat([d.ycand[:, :4].to("cpu") for d in data])

        df = pandas.DataFrame()

        df["elem_type"] = [int(graph_data_cms.elem_labels[i]) for i in torch.argmax(x[:, :len(graph_data_cms.elem_labels)], axis=-1).numpy()]
        for ifeat, feat in enumerate([
                'pt', 'eta', 'phi', 'e',
                'layer', 'depth', 'charge', 'trajpoint',
                'eta_ecal', 'phi_ecal', 'eta_hcal', 'phi_hcal',
                'muon_dt_hits', 'muon_csc_hits']):
            df["elem_{}".format(feat)] = x[:, len(graph_data_cms.elem_labels)+ifeat].numpy()

        df["elem_type"] = [int(graph_data_cms.elem_labels[i]) for i in torch.argmax(x[:, :len(graph_data_cms.elem_labels)], axis=-1).numpy()]
        df["gen_pid"] = [int(graph_data_cms.class_labels[i]) for i in gen_id.numpy()]
        df["gen_eta"] = gen_p4[:, 0].numpy()
        df["gen_phi"] = gen_p4[:, 1].numpy()
        df["gen_e"] = gen_p4[:, 2].numpy()
        df["gen_charge"] = gen_p4[:, 3].numpy()

        df["cand_pid"] = [int(graph_data_cms.class_labels[i]) for i in cand_id.numpy()]
        df["cand_eta"] = cand_p4[:, 0].numpy()
        df["cand_phi"] = cand_p4[:, 1].numpy()
        df["cand_e"] = cand_p4[:, 2].numpy()
        df["cand_charge"] = cand_p4[:, 3].numpy()
        df["pred_pid"] = [int(graph_data_cms.class_labels[i]) for i in pred_id.detach().cpu().numpy()]

        df["pred_eta"] = pred_momentum[:, 0].detach().cpu().numpy()
        df["pred_phi"] = pred_momentum[:, 1].detach().cpu().numpy()
        df["pred_e"] = pred_momentum[:, 2].detach().cpu().numpy()
        df["pred_charge"] = pred_momentum[:, 3].detach().cpu().numpy()

        dfs.append(df)
        #df_edges = pandas.DataFrame()
        #df_edges["edge0"] = edges[0].to("cpu")
        #df_edges["edge1"] = edges[1].to("cpu")
        #dfs_edges += [df_edges]

    df = pandas.concat(dfs, ignore_index=True)
    #df_edges = pandas.concat(dfs_edges, ignore_index=True)
    return df#, df_edges

#Get a unique directory name for the model
def get_model_fname(dataset, model, n_train, lr, target_type):
    model_name = type(model).__name__
    model_params = sum(p.numel() for p in model.parameters())
    import hashlib
    model_cfghash = hashlib.blake2b(repr(model).encode()).hexdigest()[:10]
    model_user = os.environ['USER']

    model_fname = '{}_{}_{}__npar_{}__cfg_{}__user_{}__ntrain_{}__lr_{}__{}'.format(
        model_name,
        dataset.split("/")[-1],
        target_type,
        model_params,
        model_cfghash,
        model_user,
        n_train,
        lr, int(time.time()))
    return model_fname

#Model with gravnet clustering
class PFNet7(nn.Module):
    def __init__(self,
        input_dim=3, hidden_dim=32, encoding_dim=256,
        output_dim_id=len(class_to_id),
        output_dim_p4=4,
        convlayer="gravnet-radius",
        convlayer2="none",
        space_dim=2, nearest=3, dropout_rate=0.0, activation="leaky_relu", return_edges=False, radius=0.1, input_encoding=0):

        super(PFNet7, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.return_edges = return_edges
        self.convlayer = convlayer
        self.input_encoding = input_encoding

        if activation == "leaky_relu":
            self.act = nn.LeakyReLU
            self.act_f = torch.nn.functional.leaky_relu
        elif activation == "selu":
            self.act = nn.SELU
            self.act_f = torch.nn.functional.selu
        elif activation == "relu":
            self.act = nn.ReLU
            self.act_f = torch.nn.functional.relu

        # if you want to add an initial encoding of the input
        conv_in_dim = input_dim
        if self.input_encoding>0:
            self.nn1 = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                self.act(),
                nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
                nn.Linear(hidden_dim, hidden_dim),
                self.act(),
                nn.Linear(hidden_dim, encoding_dim),
            )
            conv_in_dim = encoding_dim

        # (1) GNN layer
        if convlayer == "gravnet-knn":
            self.conv1 = GravNetConv(conv_in_dim, encoding_dim, space_dim, hidden_dim, nearest, neighbor_algo="knn")
        elif convlayer == "gravnet-radius":
            self.conv1 = GravNetConv(conv_in_dim, encoding_dim, space_dim, hidden_dim, nearest, neighbor_algo="radius", radius=radius)
        else:
            raise Exception("Unknown convolution layer: {}".format(convlayer))

        #decoding layer receives the raw inputs and the gravnet output
        num_decode_in = input_dim + encoding_dim

        # (2) another GNN layer if you want
        self.convlayer2 = convlayer2
        if convlayer2 == "none":
            self.conv2_1 = None
            self.conv2_2 = None
        elif convlayer2 == "sgconv":
            self.conv2_1 = SGConv(num_decode_in, hidden_dim, K=1)
            self.conv2_2 = SGConv(num_decode_in, hidden_dim, K=1)
            num_decode_in += hidden_dim
        elif convlayer2 == "graphunet":
            self.conv2_1 = GraphUNet(num_decode_in, hidden_dim, hidden_dim, 2, pool_ratios=0.1)
            self.conv2_2 = GraphUNet(num_decode_in, hidden_dim, hidden_dim, 2, pool_ratios=0.1)
            num_decode_in += hidden_dim
        elif convlayer2 == "gatconv":
            self.conv2_1 = GATConv(num_decode_in, hidden_dim, 4, concat=False, dropout=dropout_rate)
            self.conv2_2 = GATConv(num_decode_in, hidden_dim, 4, concat=False, dropout=dropout_rate)
            num_decode_in += hidden_dim
        else:
            raise Exception("Unknown convolution layer: {}".format(convlayer2))

        # (3) dropout layer if you want
        self.dropout1 = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

        # (4) DNN layer: classifying PID
        self.nn2 = nn.Sequential(
            nn.Linear(num_decode_in, hidden_dim),
            self.act(),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            self.act(),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            self.act(),
            nn.Linear(hidden_dim, output_dim_id),
        )

        # (5) DNN layer: regressing p4
        self.nn3 = nn.Sequential(
            nn.Linear(num_decode_in + output_dim_id, hidden_dim),
            self.act(),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            self.act(),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            self.act(),
            nn.Linear(hidden_dim, output_dim_p4),
        )


    def forward(self, data):

        #encode the inputs
        x = data.x

        if self.input_encoding:
            x = self.nn1(x)

        #Run a clustering of the inputs that returns the new_edge_index
        new_edge_index, x = self.conv1(x)
        x1 = self.act_f(x)

        #run a second convolution
        if self.convlayer2 != "none":
            conv2_input = torch.cat([data.x, x1], axis=-1)
            x2_1 = self.act_f(self.conv2_1(conv2_input, new_edge_index))
            x2_2 = self.act_f(self.conv2_2(conv2_input, new_edge_index))
            nn2_input = torch.cat([data.x, x1, x2_1], axis=-1)
        else:
            nn2_input = torch.cat([data.x, x1], axis=-1)

        #Decode convolved graph nodes to pdgid and p4
        cand_ids = self.nn2(self.dropout1(nn2_input))

        if self.convlayer2 != "none":
            nn3_input = torch.cat([data.x, x1, x2_2, cand_ids], axis=-1)
        else:
            nn3_input = torch.cat([data.x, x1, cand_ids], axis=-1)

        cand_p4 = data.x[:, len(elem_to_id):len(elem_to_id)+4] + self.nn3(self.dropout1(nn3_input))
        return cand_ids, cand_p4, new_edge_index

model_classes = {
    "PFNet7": PFNet7,
}

def mse_loss(input, target):
    return torch.sum((input - target) ** 2)

def weighted_mse_loss(input, target, weight):
    return torch.sum(weight * (input - target).sum(axis=1) ** 2)

@torch.no_grad()
def test(model, loader, epoch, l1m, l2m, l3m, target_type):
    with torch.no_grad():
        ret = train(model, loader, epoch, None, l1m, l2m, l3m, target_type, None)
    return ret

def compute_weights(target_ids, device):
    vs, cs = torch.unique(target_ids, return_counts=True)
    weights = torch.zeros(len(class_to_id)).to(device=device)
    for k, v in zip(vs, cs):
        weights[k] = 1.0/math.sqrt(float(v))
    return weights


def train(model, loader, epoch, optimizer, l1m, l2m, l3m, target_type, scheduler):

    is_train = not (optimizer is None)

    if is_train:
        model.train()
    else:
        model.eval()

    #loss values for each batch: classification, regression
    losses = np.zeros((len(loader), 3))

    #accuracy values for each batch (monitor classification performance)
    accuracies_batch = np.zeros(len(loader))

    #correlation values for each batch (monitor regression performance)
    corrs_batch = np.zeros(len(loader))

    #epoch confusion matrix
    conf_matrix = np.zeros((len(class_labels), len(class_labels)))

    #keep track of how many data points were processed
    num_samples = 0
    for i, data in enumerate(loader):
        t0 = time.time()

        if not multi_gpu:
            data = data.to(device)

        if is_train:
            optimizer.zero_grad()

        cand_id_onehot, cand_momentum, new_edge_index = model(data)

        nelem = data.x.shape[0]
        assert(len(new_edge_index[0])>0)
        mat1 = torch_geometric.utils.to_dense_adj(data.edge_index, max_num_nodes=nelem).flatten()
        mat2 = torch_geometric.utils.to_dense_adj(new_edge_index, max_num_nodes=nelem).flatten()

        _dev = cand_id_onehot.device
        _, indices = torch.max(cand_id_onehot, -1)
        if not multi_gpu:
            data = [data]
        num_samples += len(cand_id_onehot)

        if args.target == "gen":
            target_ids = torch.cat([d.y_gen_id for d in data]).to(_dev)
            target_p4 = torch.cat([d.ygen[:, :4] for d in data]).to(_dev)
        elif args.target == "cand":
            target_ids = torch.cat([d.y_candidates_id for d in data]).to(_dev)
            target_p4 = torch.cat([d.ycand[:, :4] for d in data]).to(_dev)

        #Predictions where both the predicted and true class label was nonzero
        #In these cases, the true candidate existed and a candidate was predicted
        msk = ((indices != 0) & (target_ids != 0)).detach().cpu()
        msk2 = ((indices != 0) & (indices == target_ids))

        accuracies_batch[i] = accuracy_score(target_ids[msk].detach().cpu().numpy(), indices[msk].detach().cpu().numpy())

        weights = compute_weights(target_ids, _dev)

        #Loss for output candidate id (multiclass)
        if l1m > 0.0:
            l1 = l1m * torch.nn.functional.cross_entropy(cand_id_onehot, target_ids, weight=weights)
        else:
            l1 = torch.tensor(0.0).to(device=_dev)

        #Loss for candidate p4 properties (regression)
        l2 = torch.tensor(0.0).to(device=_dev)
        if l2m > 0.0:
            l2 = l2m*torch.nn.functional.mse_loss(cand_momentum[msk2], target_p4[msk2])
        else:
            l2 = torch.tensor(0.0).to(device=_dev)

        l3 = l3m*torch.nn.functional.binary_cross_entropy(mat2, mat1)

        batch_loss = l1 + l2
        losses[i, 0] = l1.item()
        losses[i, 1] = l2.item()
        losses[i, 2] = l3.item()

        if is_train:
            batch_loss.backward()

        batch_loss_item = batch_loss.item()
        t1 = time.time()

        print('{}/{} batch_loss={:.2f} dt={:.1f}s'.format(i, len(loader), batch_loss_item, t1-t0), end='\r', flush=True)
        if is_train:
            optimizer.step()
            if not scheduler is None:
                scheduler.step()

        #Compute correlation of predicted and true pt values for monitoring
        corr_pt = 0.0
        if msk.sum()>0:
            corr_pt = np.corrcoef(
                cand_momentum[msk, 0].detach().cpu().numpy(),
                target_p4[msk, 0].detach().cpu().numpy())[0,1]

        corrs_batch[i] = corr_pt

        conf_matrix += confusion_matrix(target_ids.detach().cpu().numpy(),
                                        np.argmax(cand_id_onehot.detach().cpu().numpy(),axis=1), labels=range(8))

        i += 1

    corr = np.mean(corrs_batch)
    acc = np.mean(accuracies_batch)
    losses = losses.mean(axis=0)
    return num_samples, losses, corr, acc, conf_matrix

def train_loop():
    t0_initial = time.time()

    losses_train = np.zeros((args.n_epochs, 3))
    losses_val = np.zeros((args.n_epochs, 3))

    corrs = []
    corrs_v = []
    accuracies = []
    accuracies_v = []
    best_val_loss = 99999.9
    stale_epochs = 0

    print("Training over {} epochs".format(args.n_epochs))
    for j in range(args.n_epochs):
        t0 = time.time()

        if stale_epochs > patience:
            print("breaking due to stale epochs")
            break

        with experiment.train():
            model.train()
            num_samples_train, losses, c, acc, conf_matrix = train(model, train_loader, j, optimizer,
                                                                   args.l1, args.l2, args.l3, args.target, scheduler)
            experiment.log_metric('lr', optimizer.param_groups[0]['lr'], step=j)
            l = sum(losses)
            losses_train[j] = losses
            corrs += [c]
            accuracies += [acc]
            experiment.log_metric('loss',l, step=j)
            experiment.log_metric('loss1',losses[0], step=j)
            experiment.log_metric('loss2',losses[1], step=j)
            experiment.log_metric('loss3',losses[2], step=j)
            experiment.log_metric('corrs',c, step=j)
            experiment.log_metric('accuracy',acc, step=j)
            experiment.log_confusion_matrix(matrix=conf_matrix, step=j,
                                            title='Confusion Matrix Full',
                                            file_name='confusion-matrix-full-train-%03d.json' % j,
                                            labels = [str(c) for c in class_labels])


        with experiment.validate():
            model.eval()
            num_samples_val, losses_v, c_v, acc_v, conf_matrix_v = test(model, val_loader, j,
                                                                        args.l1, args.l2, args.l3, args.target)
            l_v = sum(losses_v)
            losses_val[j] = losses_v
            corrs_v += [c_v]
            accuracies_v += [acc_v]
            experiment.log_metric('loss',l_v, step=j)
            experiment.log_metric('loss1',losses_v[0], step=j)
            experiment.log_metric('loss2',losses_v[1], step=j)
            experiment.log_metric('loss3',losses_v[2], step=j)
            experiment.log_metric('corrs',c_v, step=j)
            experiment.log_metric('accuracy',acc_v, step=j)
            experiment.log_confusion_matrix(matrix=conf_matrix_v, step=j,
                                            title='Confusion Matrix Full',
                                            file_name='confusion-matrix-full-val-%03d.json' % j,
                                            labels = [str(c) for c in class_labels])

        if l_v < best_val_loss:
            best_val_loss = l_v
            stale_epochs = 0
        else:
            stale_epochs += 1

        t1 = time.time()
        epochs_remaining = args.n_epochs - j
        time_per_epoch = (t1 - t0_initial)/(j + 1)
        experiment.log_metric('time_per_epoch', time_per_epoch, step=j)
        eta = epochs_remaining*time_per_epoch/60

        spd = (num_samples_val+num_samples_train)/time_per_epoch
        losses_str = "[" + ",".join(["{:.4f}".format(x) for x in losses_v]) + "]"

        torch.save(model.state_dict(), "{0}/epoch_{1}_weights.pth".format(outpath, j))

        print("epoch={}/{} dt={:.2f}s l={:.5f}/{:.5f} c={:.2f}/{:.2f} a={:.6f}/{:.6f} partial_losses={} stale={} eta={:.1f}m spd={:.2f} samples/s lr={}".format(
            j, args.n_epochs,
            t1 - t0, l, l_v, c, c_v, acc, acc_v,
            losses_str, stale_epochs, eta, spd, optimizer.param_groups[0]['lr']))

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_train", type=int, default=80, help="number of training events")
    parser.add_argument("--n_val", type=int, default=20, help="number of validation events")
    parser.add_argument("--n_epochs", type=int, default=100, help="number of training epochs")
    parser.add_argument("--patience", type=int, default=100, help="patience before early stopping")
    parser.add_argument("--hidden_dim", type=int, default=64, help="hidden dimension")
    parser.add_argument("--encoding_dim", type=int, default=256, help="encoded element dimension")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of .pt files to load in parallel")
    parser.add_argument("--model", type=str, choices=sorted(model_classes.keys()), help="type of model to use", default="PFNet6")
    parser.add_argument("--target", type=str, choices=["cand", "gen"], help="Regress to PFCandidates or GenParticles", default="cand")
    parser.add_argument("--dataset", type=str, help="Input dataset", required=True)
    parser.add_argument("--outpath", type=str, default = 'data/', help="Output folder")
    parser.add_argument("--activation", type=str, default='leaky_relu', choices=["selu", "leaky_relu", "relu"], help="activation function")
    parser.add_argument("--optimizer", type=str, default='adam', choices=["adam", "adamw"], help="optimizer to use")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--l1", type=float, default=1.0, help="Loss multiplier for pdg-id classification")
    parser.add_argument("--l2", type=float, default=1.0, help="Loss multiplier for momentum regression")
    parser.add_argument("--l3", type=float, default=1.0, help="Loss multiplier for clustering")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    parser.add_argument("--radius", type=float, default=0.1, help="Radius-graph radius")
    parser.add_argument("--convlayer", type=str, choices=["gravnet-knn", "gravnet-radius", "sgconv", "gatconv"], help="Convolutional layer", default="gravnet")
    parser.add_argument("--convlayer2", type=str, choices=["sgconv", "graphunet", "gatconv", "none"], help="Convolutional layer", default="none")
    parser.add_argument("--space_dim", type=int, default=2, help="Spatial dimension for clustering in gravnet layer")
    parser.add_argument("--nearest", type=int, default=3, help="k nearest neighbors in gravnet layer")
    parser.add_argument("--overwrite", action='store_true', help="overwrite if model output exists")
    parser.add_argument("--disable_comet", action='store_true', help="disable comet-ml")
    parser.add_argument("--input_encoding", type=int, help="use an input encoding layer", default=0)
    parser.add_argument("--load", type=str, help="Load the weight file", required=False, default=None)
    parser.add_argument("--scheduler", type=str, help="LR scheduler", required=False, default="none", choices=["none", "onecycle"])
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()

    # # the next part initializes some args values (to run the script not from terminal)
    # class objectview(object):
    #     def __init__(self, d):
    #         self.__dict__ = d
    #
    # args = objectview({'n_train': 3, 'n_val': 2, 'n_epochs': 2, 'patience': 100, 'hidden_dim':64, 'encoding_dim': 256,
    # 'batch_size': 1, 'model': 'PFNet7', 'target': 'cand', 'dataset': '../../test_tmp/data/TTbar_14TeV_TuneCUETP8M1_cfi',
    # 'outpath': 'data/', 'activation': 'leaky_relu', 'optimizer': 'adam', 'lr': 1e-4, 'l1': 1, 'l2': 1, 'l3': 1, 'dropout': 0.5,
    # 'radius': 0.1, 'convlayer': 'gravnet-radius', 'convlayer2': 'none', 'space_dim': 2, 'nearest': 3, 'overwrite': True,
    # 'disable_comet': True, 'input_encoding': 0, 'load': None, 'scheduler': 'none'})

    # define the dataset
    full_dataset = PFGraphDataset(args.dataset)

    #one-hot encoded element ID + element parameters
    input_dim = 26

    #one-hot particle ID and momentum
    output_dim_id = len(class_to_id)
    output_dim_p4 = 4

    edge_dim = 1

    patience = args.patience

    train_dataset = torch.utils.data.Subset(full_dataset, np.arange(start=0, stop=args.n_train))
    val_dataset = torch.utils.data.Subset(full_dataset, np.arange(start=args.n_train, stop=args.n_train+args.n_val))
    print("train_dataset", len(train_dataset))
    print("val_dataset", len(val_dataset))

    #hack for multi-gpu training
    if not multi_gpu:
        def collate(items):
            l = sum(items, [])
            return Batch.from_data_list(l)
    else:
        def collate(items):
            l = sum(items, [])
            return l

    train_loader = DataListLoader(train_dataset, batch_size=args.batch_size, pin_memory=False, shuffle=False)
    train_loader.collate_fn = collate
    val_loader = DataListLoader(val_dataset, batch_size=args.batch_size, pin_memory=False, shuffle=False)
    val_loader.collate_fn = collate

    model_class = model_classes[args.model]
    model_kwargs = {'input_dim': input_dim,
                    'hidden_dim': args.hidden_dim,
                    'encoding_dim': args.encoding_dim,
                    'output_dim_id': output_dim_id,
                    'output_dim_p4': output_dim_p4,
                    'dropout_rate': args.dropout,
                    'convlayer': args.convlayer,
                    'convlayer2': args.convlayer2,
                    'radius': args.radius,
                    'space_dim': args.space_dim,
                    'activation': args.activation,
                    'nearest': args.nearest,
                    'input_encoding': args.input_encoding}


    #instantiate the model
    model = model_class(**model_kwargs)
    if args.load:
        s1 = torch.load(args.load, map_location=torch.device('cpu'))
        s2 = {k.replace("module.", ""): v for k, v in s1.items()}
        model.load_state_dict(s2)

    if multi_gpu:
        model = torch_geometric.nn.DataParallel(model)

    model.to(device)

    model_fname = get_model_fname(args.dataset, model, args.n_train, args.lr, args.target)

    # need your api key in a .comet.config file: see https://www.comet.ml/docs/python-sdk/advanced/#comet-configuration-variables
    experiment = Experiment(project_name="particleflow", disabled=args.disable_comet)
    experiment.set_model_graph(repr(model))
    experiment.log_parameters(dict(model_kwargs, **{'model': args.model, 'lr':args.lr, 'model_fname': model_fname,
                                                    'l1': args.l1, 'l2':args.l2,
                                                    'n_train':args.n_train, 'target':args.target, 'optimizer': args.optimizer}))
    outpath = osp.join(args.outpath, model_fname)
    if osp.isdir(outpath):
        if args.overwrite:
            print("model output {} already exists, deleting it".format(outpath))
            import shutil
            shutil.rmtree(outpath)
        else:
            print("model output {} already exists, please delete it".format(outpath))
            sys.exit(0)
    try:
        os.makedirs(outpath)
    except Exception as e:
        pass

    with open('{}/model_kwargs.pkl'.format(outpath), 'wb') as f:
        pickle.dump(model_kwargs, f,  protocol=pickle.HIGHEST_PROTOCOL)

    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    scheduler = None
    if args.scheduler == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            steps_per_epoch=int(len(train_loader)),
            epochs=args.n_epochs + 1,
            anneal_strategy='linear',
        )

    loss = torch.nn.MSELoss()
    loss2 = torch.nn.BCELoss()

    print(model)
    print(model_fname)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("params", params)

    model.train()

    train_loop()
    # with torch.autograd.profiler.profile(use_cuda=True) as prof:
    #     train_loop()

    # print(prof.key_averages().table(sort_by="cuda_time_total"))
