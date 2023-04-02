import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from dataset import (MPC_Block, adjust_future_deltas, rotation_matrix,
                     transform_sumo2carla)
from utils.config import DT, NUM_PREDICT, OBS_LEN, PRED_LEN
from utils.feature_utils import get_intention_from_vehicle_id

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
obs_len, pred_len, dt = OBS_LEN, PRED_LEN, DT

parser = argparse.ArgumentParser(description="")
parser.add_argument('--csv_folder', type=str, help='path to the data set (*.csv)', default='csv/train')
parser.add_argument('--pkl_folder', type=str, help='path to the preprocessed data (*.pkl)', default='csv/train_pre')
parser.add_argument('--num_mpc_aug', type=int, help='number of MPC augmentation', default=2)
args = parser.parse_args()

csv_folder = args.csv_folder
preprocess_folder = args.pkl_folder
os.makedirs(preprocess_folder, exist_ok=True)
n_mpc_aug = args.num_mpc_aug
csv_files = os.listdir(csv_folder)
csv_files.sort()
graphs = list()
plot = False
if plot:
    fig, ax = plt.subplots()    # plot the mcp augmented data
    fig2, ax2 = plt.subplots()  # plot the rotated gt
    os.makedirs('images', exist_ok=True)

for csv_file in tqdm(csv_files):
    if os.path.splitext(csv_file)[1] != '.csv':
        continue
    print(csv_file)
    df = pd.read_csv(os.path.join(csv_folder, csv_file))
    all_features = list()
    for track_id, remain_df in df.groupby('TRACK_ID'):
        if(len(remain_df) >= (obs_len + pred_len)):
            coords = remain_df[['X', 'Y', 'speed', 'yaw']].values   # [obs+pred, 4]
            coords[:, 3] = np.deg2rad(coords[:, 3])
            transform_sumo2carla(coords)
            intention = get_intention_from_vehicle_id(track_id)[:3]
            features = np.hstack((coords, intention * np.ones((coords.shape[0], 3))))   # [obs+pred, 7]
            all_features.append(features)

    num_rows = features.shape[0]    # obs+pred
    all_features = np.array(all_features)   # [vehicle, steps(obs+pred), 7]: [x, y, speed, yaw, intent, intent, intent]
    acc_delta_padding = np.empty((all_features.shape[0], all_features.shape[1], 2))
    acc_delta_padding[:] = np.NaN
    all_features = np.concatenate((all_features, acc_delta_padding), axis=-1) 
        # [vehicle, steps, 9]: [x, y, speed, yaw, intent, intent, intent, acc, delta]
    num_cars = len(all_features)
    edges = [[x,y] for x in range(num_cars) for y in range(num_cars)]
    edge_index = torch.tensor(edges, dtype=torch.long).T    # [2, edge]
    noise_range = 3.0

    # for each timestep, create an interaction graph
    for row in range(0, num_rows - NUM_PREDICT):
        print(row)
        x = all_features[:, row, :7]   # [vehicle, 7]
        
        # translate and then rotate Gt
        y = (all_features[:, row+1:row+1+NUM_PREDICT, :2] - all_features[:, row:row+1, :2]).transpose(0,2,1) 
            # [vehicle, pred_len, 2] -> [vehicle, 2, pred_len]
        rotations = np.array([rotation_matrix(x[i][3])  for i in range(x.shape[0])])    # [vehicle, 2, 2]
        y = (rotations @ y)         # [vehicle, 2, pred_len], transform y into local coordinate system
        y = y.transpose(0, 2, 1)    # [vehicle, pred_len, 2]
        
        # use MPC to compute acc and delta
        curr_states = all_features[:, row, :4]  # [vehicle, 4]
        future_states = all_features[:, row+1:row+1+NUM_PREDICT, :4]    # [vehicle, pred_len, 4], [x, y, speed, yaw]
        adjust_future_deltas(curr_states, future_states)
        if plot:
            ax.set_xlim([-75, 75])
            ax.set_ylim([75, -75])  # invert y axis because of left-handed Cartesian coordinate system
            ax.set_aspect('equal')
            ax2.set_xlim([-75, 75])
            ax2.set_ylim([75, -75])
            ax2.set_aspect('equal')
            ax.scatter(x=curr_states[:,0], y=curr_states[:,1], s=8.0, c='blue')
            ax.scatter(x=future_states[:,:,0].reshape(-1), y=future_states[:,:,1].reshape(-1), s=2.0, c='green')
            ax2.scatter(x=y[:,:,0].reshape(-1), y=y[:,:,1].reshape(-1), s=2.0, c='green')
        acc_delta_old = all_features[:, row+1:row+1+NUM_PREDICT, -2:] # [vehicle, pred_len, 2], [acc, delta]
        shifted_curr, mpc_output = MPC_Block(curr_states, future_states, acc_delta_old, noise_range=0)    
            # [vehicle, 4], [vehicle, pred_len, 6]: [x, y, v, yaw, acc, delta]
        all_features[:, row+1:row+1+NUM_PREDICT, -2:] = mpc_output[:, :, -2:]   # store the control vector to accelerate future MPC opt
        speed = all_features[:, row+1:row+1+NUM_PREDICT, 2:3]   # [vehicle, pred_len, 1]
        yaw = all_features[:, row+1:row+1+NUM_PREDICT, 3:4] - all_features[:, row:row+1, 3:4] + np.pi/2     
            # [vehicle, pred_len, 1], align the initial direction to +y
        y = np.concatenate((y, speed, yaw, mpc_output[:, :, -2:]), axis=2).reshape(num_cars, -1) # [vehicle, pred_len*6]
        # data = Data(x=torch.tensor(x, dtype=torch.float), y=torch.tensor(y, dtype=torch.float) , edge_index=edge_index, path = csv_file, row  = torch.tensor([row]))
        data = (torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float), edge_index, torch.tensor([row]))
        # graphs.append(data)
        # x: [vehicle, 7]: [x_0, y_0, speed_0, yaw_0, intent, intent, intent]
        # y: [vehicle, pred_len * 6]: [[x_1, y_1, v_1, yaw_1, acc_1, delta_1, x_2, y_2...], ...]
        with open(f'{preprocess_folder}/{os.path.splitext(csv_file)[0]}-{str(row).zfill(3)}-0.pkl', 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        for a in range(n_mpc_aug):
            shifted_curr, mpc_output = MPC_Block(curr_states, future_states, acc_delta_old, noise_range=noise_range)    
                # [vehicle, 4], [vehicle, pred_len, 6]: [x, y, v, yaw, acc, delta]
            x = x.copy()    # no in-place modify
            x[:, :2] = shifted_curr[:, :2]
            y = (mpc_output[:, :, :2] - np.expand_dims(shifted_curr[:, :2], axis=1)).transpose(0,2,1)    # [vehicle, 2, pred_len]
            y = (rotations @ y)
            y = y.transpose(0, 2, 1)    # [vehicle, pred_len, 2]
            if plot:
                ax2.scatter(x=y[:,:,0].reshape(-1), y=y[:,:,1].reshape(-1), s=0.5, c='red')
            mpc_output[:, :, 3:4] = mpc_output[:, :, 3:4] - all_features[:, row:row+1, 3:4] + np.pi/2
            y = np.concatenate((y, mpc_output[:,:,2:]), axis=-1)    # [vehicle, pred_len, 6]
            y = y.reshape(num_cars, -1)
            # data = Data(x=torch.tensor(x, dtype=torch.float), y=torch.tensor(y, dtype=torch.float) , edge_index=edge_index, path = csv_file, row  = torch.tensor([row]))
            data = (torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float), edge_index, torch.tensor([row]))
            # graphs.append(data)     # num of graph per csv: (num_rows - NUM_PREDICT) * (1 + n_mpc_aug)
            with open(f'{preprocess_folder}/{os.path.splitext(csv_file)[0]}-{str(row).zfill(3)}-{int(a+1)}.pkl', 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            if plot:
                ax.scatter(x=shifted_curr[:,0], y=shifted_curr[:,1], s=2.0, c='red')
                ax.scatter(x=mpc_output[:,:,0].reshape(-1), y=mpc_output[:,:,1].reshape(-1), s=0.5, c='red')

        if plot:
            fig.savefig(f'images/{os.path.splitext(csv_file)[0]}-{str(row).zfill(3)}.png')
            fig2.savefig(f'images/{os.path.splitext(csv_file)[0]}-{str(row).zfill(3)}-y.png')
            ax.clear()
            ax2.clear()
            # break
