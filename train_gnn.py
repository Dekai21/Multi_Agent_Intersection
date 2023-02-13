import argparse
import copy
import json
import os
import pdb
import pickle
import pickle as pkl
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import GRU, LSTM, BatchNorm1d, Linear, ReLU, Sequential
from torch.nn.functional import smooth_l1_loss
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (CGConv, EdgeConv, GATConv, GATv2Conv, GCNConv,
                                GENConv, GeneralConv, GINEConv, GMMConv)
from torch_geometric.nn import GraphConv
from torch_geometric.nn import GraphConv as GNNConv
from torch_geometric.nn import (PDNConv, RGCNConv, SplineConv, TAGConv,
                                TransformerConv, global_add_pool,
                                global_max_pool, global_mean_pool)
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import from_networkx
from tqdm import tqdm

from utils.feature_utils import get_intention_from_vehicle_id

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from dataset import CarDataset, CarDataset2, CarDataset4, CarDataset5
from utils.config import DT, OBS_LEN, PRED_LEN

obs_len, pred_len, dt = OBS_LEN, PRED_LEN, DT

# pred_len
exp_id = "sumo_0913"
model_path = f"trained_params/{exp_id}"
os.makedirs(model_path, exist_ok=True)

parser = argparse.ArgumentParser(description="")
parser.add_argument('--train_folder', type=str, help='path to the training set', default='csv/train')
parser.add_argument('--val_folder', type=str, help='path to the validation set', default='csv/val')
parser.add_argument('--epoch', type=int, help='number of total training epochs', default=2000)
args = parser.parse_args()

train_folder = args.train_folder
val_folder = args.val_folder

mlp = False
collision_penalty = True
batch_size = 8000   # 8000
train_dataset = CarDataset5(preprocess_folder=train_folder, mlp=False, mpc_aug=False)
val_dataset = CarDataset5(preprocess_folder=val_folder, mlp=False, mpc_aug=False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

class GNN_mtl_gnn(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(21)
        self.conv1 = GNNConv(hidden_channels, hidden_channels)
        self.conv2 = GNNConv(hidden_channels, hidden_channels)
        self.linear1 = nn.Linear(5, 64)
        self.linear2 = nn.Linear(64, hidden_channels)
        self.linear3 = nn.Linear(hidden_channels, hidden_channels)
        self.linear4 = nn.Linear(hidden_channels, hidden_channels)
        self.linear5 = nn.Linear(hidden_channels, 30*2)

    def forward(self, x, edge_index):
        x = self.linear1(x).relu()
        x = self.linear2(x).relu()
        x = self.linear3(x).relu() + x
        x = self.linear4(x).relu() + x
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.linear5(x)
        return x  # mtl

class GNN_mtl_mlp(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(21)
        self.conv1 = nn.Linear(hidden_channels, hidden_channels)
        self.conv2 = nn.Linear(hidden_channels, hidden_channels)
        self.linear1 = nn.Linear(5, 64)
        self.linear2 = nn.Linear(64, hidden_channels)
        self.linear3 = nn.Linear(hidden_channels, hidden_channels)
        self.linear4 = nn.Linear(hidden_channels, hidden_channels)
        self.linear5 = nn.Linear(hidden_channels, 30*2)

    def forward(self, x, edge_index):
        x = self.linear1(x).relu()
        x = self.linear2(x).relu()
        x = self.linear3(x).relu() + x
        x = self.linear4(x).relu() + x
        x = self.conv1(x).relu()
        x = self.conv2(x).relu()
        x = self.linear5(x)
        return x  # mtl

model = GNN_mtl_mlp(hidden_channels=128).to(device) if mlp else GNN_mtl_gnn(hidden_channels=128)
print(model)

def rotation_matrix_back(yaw):
    """
    Rotate back. 
    https://en.wikipedia.org/wiki/Rotation_matrix#Non-standard_orientation_of_the_coordinate_system
    """
    rotation = np.array([[np.cos(-np.pi/2+yaw), -np.sin(-np.pi/2+yaw)],[np.sin(-np.pi/2+yaw), np.cos(-np.pi/2+yaw)]])
    rotation = torch.tensor(rotation).float()
    return rotation

def train(model, device, data_loader, optimizer, collision_penalty=False):
    """ Performs an epoch of model training.

    Parameters:
    model (nn.Module): Model to be trained.
    loss_fn (nn.Module): Loss function for training.
    device (torch.Device): Device used for training.
    data_loader (torch.utils.data.DataLoader): Data loader containing all batches.
    optimizer (torch.optim.Optimizer): Optimizer used to update model.

    Returns:
    float: Total loss for epoch.
    """
    model.train()
    total_loss = 0

    step_weights = torch.ones(30, device=device)
    step_weights[:5] *= 5
    step_weights[0] *= 5
    dist_threshold = 4

    for batch in data_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x[:,[0,1,4,5,6]], batch.edge_index)
        out = out.reshape(-1,30,2)  # [v, 30, 2]
        if not collision_penalty:
            gt = batch.y.reshape(-1,30,8)[:,:,[4,5]]
            error = ((gt-out).square().sum(-1) * step_weights).sum(-1)
            loss = (batch.weights * error).nanmean()
        else:
            out = out.permute(0,2,1)    # [v, 2, pred]
            yaw = batch.x[:,3].detach().cpu().numpy()
            rotations = torch.stack([rotation_matrix_back(yaw[i])  for i in range(batch.x.shape[0])]).to(out.device)
            out = torch.bmm(rotations, out).permute(0,2,1)       # [v, pred, 2]
            out += batch.x[:,[0,1]].unsqueeze(1)
            gt = batch.y.reshape(-1,30,8)[:,:,[0,1]]
            error = ((gt-out).square().sum(-1) * step_weights).sum(-1)
            loss = (batch.weights * error).nanmean()
            
            mask = (batch.edge_index[0,:] < batch.edge_index[1,:])
            _edge = batch.edge_index[:, mask].T   # [edge',2]
            dist = torch.linalg.norm(out[_edge[:,0]] - out[_edge[:,1]], dim=-1)
            dist = dist_threshold - dist[dist < dist_threshold]
            collision_penalty = dist.square().mean()
            # collision_penalty = smooth_l1_loss(dist, torch.zeros_like(dist), reduction='sum')
            # loss += collision_penalty * (loss.item()/collision_penalty.item()/10)
            # print(f"loss: {loss.item()}, collision penalty: {collision_penalty.item()}")
            loss += collision_penalty * 20

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(data_loader)

def evaluate(model, device, data_loader):
    """ Performs an epoch of model training.

    Parameters:
    model (nn.Module): Model to be trained.
    loss_fn (nn.Module): Loss function for training.
    device (torch.Device): Device used for training.
    data_loader (torch.utils.data.DataLoader): Data loader containing all batches.
    optimizer (torch.optim.Optimizer): Optimizer used to update model.

    Returns:
    float: Total loss for epoch.
    """
    step_weights = torch.ones(30, device=device)
    step_weights[:5] *= 5
    step_weights[0] *= 5
    dist_threshold = 4
    mr_threshold = 4
    model.eval()
    ade, fde = [], []
    n_edge, n_collision = [], []
    val_losses, collision_penalties = [], []
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            out = model(batch.x[:,[0,1,4,5,6]], batch.edge_index)
            out = out.reshape(-1,30,2)  # [v, 30, 2]
            gt = batch.y.reshape(-1,30,6)[:,:,[4,5]]
            _error = (gt-out).square().sum(-1)
            error = _error.clone() ** 0.5
            _error = (_error * step_weights).sum(-1)
            val_loss = (batch.weights * _error).nanmean()
            val_losses.append(val_loss)
            fde.append(error[:,-1])
            ade.append(error.mean(dim=-1))

            out = out.permute(0,2,1)    # [v, 2, pred]
            yaw = batch.x[:,3].detach().cpu().numpy()
            rotations = torch.stack([rotation_matrix_back(yaw[i])  for i in range(batch.x.shape[0])]).to(out.device)
            out = torch.bmm(rotations, out).permute(0,2,1)       # [v, pred, 2]
            out += batch.x[:,[0,1]].unsqueeze(1)

            mask = (batch.edge_index[0,:] < batch.edge_index[1,:])
            _edge = batch.edge_index[:, mask].T   # [edge',2]
            dist = torch.linalg.norm(out[_edge[:,0]] - out[_edge[:,1]], dim=-1) # [edge, 30]
            collision_penalty = dist_threshold - dist[dist < dist_threshold]
            collision_penalty = collision_penalty.square().mean() * 20
            collision_penalties.append(collision_penalty)

            # out = out.permute(0,2,1)    # [v, 2, pred]
            # yaw = batch.x[:,3].detach().cpu().numpy()
            # rotations = torch.stack([rotation_matrix_back(yaw[i])  for i in range(batch.x.shape[0])]).to(out.device)
            # out = torch.bmm(rotations, out).permute(0,2,1)       # [v, pred, 2]
            # out += batch.x[:,[7,8]].unsqueeze(1)
            # # gt = batch.y.reshape(-1,50,6)[:,:,[0,1]]
            # # error = ((gt-out).square().sum(-1) * step_weights).sum(-1)
            # # loss = (batch.weights * error).nanmean()
            
            # mask = (batch.edge_index[0,:] < batch.edge_index[1,:])
            # _edge = batch.edge_index[:, mask].T   # [edge',2]
            # # pos1 = torch.stack([out[_edge[i, 0]] for i in range(_edge.shape[0])])
            # # pos2 = torch.stack([out[_edge[i, 1]] for i in range(_edge.shape[0])])
            # # dist = torch.linalg.norm(pos1 - pos2, dim=-1)
            # dist = torch.linalg.norm(out[_edge[:,0]] - out[_edge[:,1]], dim=-1) # [edge, 50]
            dist = torch.min(dist, dim=-1)[0]
            n_edge.append(len(dist))
            n_collision.append((dist<2).sum().item())
    
    ade = torch.cat(ade).mean()
    fde = torch.cat(fde)
    mr = ((fde>mr_threshold).sum() / len(fde)).item()
    fde = fde.mean()
    collision_rate = sum(n_collision) / sum(n_edge)
    collision_penalties = torch.tensor(collision_penalties).mean()
    val_losses = torch.tensor(val_losses).mean()
    
    return ade.item(), fde.item(), mr, collision_rate, val_losses.item(), collision_penalties.item()


# def evaluate(model, device, data_loader):
#     """ Performs an epoch of model training.

#     Parameters:
#     model (nn.Module): Model to be trained.
#     loss_fn (nn.Module): Loss function for training.
#     device (torch.Device): Device used for training.
#     data_loader (torch.utils.data.DataLoader): Data loader containing all batches.
#     optimizer (torch.optim.Optimizer): Optimizer used to update model.

#     Returns:
#     float: Total loss for epoch.
#     """
#     model.eval()
#     ade, fde = [], []
#     with torch.no_grad():
#         for batch in data_loader:
#             batch = batch.to(device)
#             out = model(batch.x[:,[0,1,4,5,6]], batch.edge_index)
#             out = out.reshape(-1,30,2)  # [v, 30, 2]
#             gt = batch.y.reshape(-1,30,8)[:,:,[4,5]]
#             error = torch.linalg.norm(gt-out, dim=-1)   # [v, 30]
#             fde.append(error[:,-1])
#             ade.append(error.mean(dim=-1))
#     ade = torch.cat(ade).mean()
#     fde = torch.cat(fde).mean()
#     return ade, fde

# losses = list()
# print(checkpoint_dir)
# checkpoint_dir = 'D:/codes/tmp/dekai_intellisys_ws2021/gnn_theta/trained_params/sumo_0911/model_rot_gnn_mtl_wp400_sumo_0911_e3_0990.pth'
# if checkpoint_dir:
#     # load_checkpoint(checkpoint_dir, model)
#     checkpoint = torch.load(checkpoint_dir, map_location=torch.device('cpu'))
#     model.load_state_dict(checkpoint)

model = model.to(device)

# min_ade = 1e6
# min_fde = 1e6
# best_epoch = 0
# patience = 100
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# for epoch in tqdm(range(0, 80000)):
#     loss = train(model, device, train_loader, optimizer, collision_penalty)
#     if epoch % 20 == 0:
#         ade, fde = evaluate(model, device, val_loader)
#         print(f"Epoch {epoch}: Train Loss: {loss}, ADE: {ade}, FDE: {fde}, lr: {optimizer.param_groups[0]['lr']}.")
#         if fde < min_fde:
#             min_ade, min_fde = ade, fde
#             best_epoch = epoch
#             print(f'Epoch {epoch}: save the model.')
#             torch.save(model.state_dict(), model_path + f"/model_{'mlp' if mlp else 'gnn'}_mtl_{'wp' if collision_penalty else 'np'}_sumo_0910_e3.pth")
#             # torch.save(model.state_dict(), model_path + f"/model_rot_tx_64_InD_0906.pth")
#         elif (epoch - best_epoch) > patience:
#             if patience > 1600: # x16
#                 print(f"{'MLP' if mlp else 'GNN'} earlier stops, Best Epoch: {best_epoch}, Min ADE: {min_ade}, Min FDE: {min_fde}.")
#                 break
#             else:
#                 optimizer.param_groups[0]['lr'] *= 0.5
#                 patience *= 2

min_ade = 1e6
min_fde = 1e6
best_epoch = 0
patience = 100
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
record = []
for epoch in tqdm(range(0, args.epoch)):
    loss = train(model, device, train_loader, optimizer)
    if epoch % 10 == 0:
        ade, fde, mr, collision_rate, val_losses, collision_penalties = evaluate(model, device, val_loader)
        record.append([ade, fde, mr, collision_rate, val_losses, collision_penalties])
        print(f"Epoch {epoch}: Train Loss: {loss}, ADE: {ade}, FDE: {fde}, MR: {mr}, CR:{collision_rate}, Val_loss: {val_losses}, CP: {collision_penalties}, lr: {optimizer.param_groups[0]['lr']}.")
        torch.save(model.state_dict(), model_path + f"/model_{'mlp' if mlp else 'gnn'}_{'wp' if collision_penalty else 'np'}_{exp_id}_e3_{str(epoch).zfill(4)}.pth")
        if fde < min_fde:
            min_ade, min_fde = ade, fde
            best_epoch = epoch
            print(" !!! New smallest FDE !!! ")
        elif (epoch - best_epoch) > patience:
            if patience > 1600: # x16
                print(f"{'MLP' if mlp else 'GNN'} earlier stops, Best Epoch: {best_epoch}, Min ADE: {min_ade}, Min FDE: {min_fde}, MR: {mr}, CR:{collision_rate}.")
                break
            else:
                optimizer.param_groups[0]['lr'] *= 0.5
                patience *= 2
pkl_file = f"model_{'mlp' if mlp else 'gnn'}_{'wp' if collision_penalty else 'np'}_{exp_id}_e3.pkl"
# pkl_file = f"model_{'mlp' if mlp else 'gnn'}_mtl_sumo_0911_e3.pkl"
with open(f'{model_path}/{pkl_file}', 'wb') as handle:
    pickle.dump(record, handle, protocol=pickle.HIGHEST_PROTOCOL)