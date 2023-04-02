# %%

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

from sumo_integration.MPC_XY_Frame import linear_mpc_control_data_aug
from utils.config import DT, OBS_LEN, PRED_LEN


obs_len, pred_len, dt = OBS_LEN, PRED_LEN, DT


def rotation_matrix(yaw):
    """
    Make the current direction aligns to +y axis. 
    https://en.wikipedia.org/wiki/Rotation_matrix#Non-standard_orientation_of_the_coordinate_system
    """
    rotation = np.array([[np.cos(np.pi/2-yaw), -np.sin(np.pi/2-yaw)],[np.sin(np.pi/2-yaw), np.cos(np.pi/2-yaw)]])
    return rotation


class CarDataset(InMemoryDataset):  
    # read from preprocessed data
    def __init__(self, preprocess_folder, plot=False, mlp=False, mpc_aug=True):
        self.preprocess_folder = preprocess_folder
        self.plot = plot
        self.mlp = mlp
        self.mpc_aug = mpc_aug        
        super().__init__(preprocess_folder)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        pt_name = 'data'
        pt_name += '_mlp' if self.mlp else '_gnn'
        if self.mpc_aug:
            pt_name += '_aug'
        return [f'{pt_name}.pt']
    
    def process(self):
        """ 
        Converts raw data into GNN-readable format by constructing
        graphs out of connectivity matrices.
        """
        preprocess_files = os.listdir(self.preprocess_folder)
        preprocess_files.sort()
        graphs = list()
        
        if self.plot:
            fig, ax = plt.subplots()    # plot the mcp augmented data
            fig2, ax2 = plt.subplots()  # plot the rotated gt
            os.makedirs(f'images/{self.preprocess_folder}', exist_ok=True)

        for file in tqdm(preprocess_files):
            if os.path.splitext(file)[1] != '.pkl':
                continue
            if not self.mpc_aug:
                if os.path.splitext(file)[0].split('-')[1] != '0':
                    continue
            data = pickle.load(open(os.path.join(self.preprocess_folder, file), "rb"))  
            # x: [v, 7], [x, y, v, yaw, intention(3-bit)],
            # y: [v, pred_len*6], [x, y, v, yaw, acc, steering],
            # edge_indices: [2, edge], 
            # t: [], row index in a csv file
            n_v = data[0].shape[0]
            weights = torch.ones(n_v)
            turn_index = (data[0][:, 4] + data[0][:, 6]).bool() # left- and right-turn cases with higher weights
            center_index1 = (data[0][:,0].abs() < 30) * (data[0][:,1].abs() < 30)   # vehicles in the central area with higher weights
            center_index2 = (data[0][:,0].abs() < 40) * (data[0][:,1].abs() < 40)
            weights[turn_index] *= 1.5
            weights[center_index1] *= 4
            weights[center_index2] *= 4

            if self.mlp:
                self_loop_index = (data[2][0,:] == data[2][1,:])
                graph = Data(x=data[0], y=data[1], edge_index=data[2][:,self_loop_index], t=data[3], weights=weights)
            else:
                graph = Data(x=data[0], y=data[1], edge_index=data[2], t=data[3], weights=weights)
            # [v,7], [v, pred_len*6], [2, edge], []
            graphs.append(graph)

        data, slices = self.collate(graphs)
        torch.save((data, slices), self.processed_paths[0])


def adjust_future_deltas(curr_states, future_states)-> None:
    """
    The range of delta angle is [-90, 270], in order to avoid the jump, adjust the future delta angles.
    :param curr_states: [vehicle, 4]
    :param future_states: [vehicle, pred_len, 4]
    """

    assert curr_states.shape[0] == future_states.shape[0]
    num_vehicle = curr_states.shape[0]
    num_step = future_states.shape[1]

    for i_vehicle in range(num_vehicle):
        for i_step in range(num_step):
            if (future_states[i_vehicle, i_step, 3] - curr_states[i_vehicle, 3]) < -np.pi:
                future_states[i_vehicle, i_step, 3] += 2*np.pi
            elif (future_states[i_vehicle, i_step, 3] - curr_states[i_vehicle, 3]) > np.pi:
                future_states[i_vehicle, i_step, 3] -= 2*np.pi

    return None


def MPC_Block(curr_states: np.ndarray, target_states: np.ndarray, acc_delta_old: np.ndarray, noise_range: float = 0.0):
    """
    :param curr_states: [vehicle, 4], [[x, y, speed, yaw], ...]
    :param target_states: [vehicle, pred_len, 4]
    :param acc_delta_old: [vehicle, pred_len, 2]
    :param noise_range: noise on the lateral direction

    :return shifted_curr: [vehicle, 4]
    :return mpc_output: [vehicle, pred_len, 6], [x, y, speed, yaw, acc, delta]
    """

    # acc_delta_new = np.zeros_like(acc_delta_old)
    num_vehicles = curr_states.shape[0]
    pred_len = target_states.shape[1]
    shifted_curr = np.zeros((num_vehicles, 4))
    mpc_output = np.zeros((num_vehicles, pred_len, 6))
    for v in range(num_vehicles):
        shifted_curr[v], mpc_output[v] = MPC_module(curr_states[v], target_states[v], acc_delta_old[v], noise_range)
    return shifted_curr, mpc_output


def MPC_module(curr_state_v: np.ndarray, target_states_v: np.ndarray, acc_delta_old_v: np.ndarray, noise_range: float = 0.0):
    """
    :param curr_state_v: [4], [x_0, y_0, speed_0, yaw_0]
    :param target_states_v: [pred_len, 4], [[x_1, y_1, speed_1, yaw_1], ...]
    :param acc_delta_old_v: [pred_len, 2], [[acc_1, delta_1], ...]
    :param noise_range: noise on the lateral direction

    :return shifted_curr: [4]
    :return mpc_output: [pred_len, 6]
    """
    
    acc_delta_old_v[np.isnan(acc_delta_old_v)] = 0.0    # [pred_len, 2]
    a_old = acc_delta_old_v[:, 0].tolist()
    delta_old = acc_delta_old_v[:, 1].tolist()

    if noise_range > 0:
        curr_state_v = curr_state_v.copy()  # avoid add noise in-place
        noise_direction = curr_state_v[3] - np.deg2rad(90)
        noise_length = np.random.uniform(low=-1, high=1) * noise_range  # TODO: uniform or Gaussian distribution?
        # noise = np.array([np.cos(noise_direction), np.sin(noise_direction)]) * noise_length
        noise = np.array([np.cos(noise_direction), np.sin(noise_direction)]) * noise_length
        curr_state_v[:2] += noise

    curr_state_v = curr_state_v.reshape(1, 4)

    target_states_v = np.concatenate((curr_state_v, target_states_v), axis=0) # [pred_len+1, 4]
    _curr_state_v = curr_state_v.reshape(-1).tolist()

    target_states_v = target_states_v.T
    a_opt, delta_opt, x_opt, y_opt, v_opt, yaw_opt = linear_mpc_control_data_aug(target_states_v, _curr_state_v, a_old, delta_old)

    mpc_output = np.concatenate((x_opt[1:].reshape(-1, 1), y_opt[1:].reshape(-1, 1), v_opt[1:].reshape(-1, 1), yaw_opt[1:].reshape(-1, 1), \
        a_opt.reshape(-1, 1), delta_opt.reshape(-1, 1)), axis=1)
    # assert acc_delta_new_v.shape[1] == 2

    return curr_state_v.reshape(-1), mpc_output


def transform_sumo2carla(states: np.ndarray):
    """
    In-place transform from sumo to carla: [x_carla, y_carla, yaw_carla] = [x_sumo, -y_sumo, yaw_sumo-90].
    Note: 
        - the coordinate system in Carla is more convenient since the angle increases in the direction of rotation from +x to +y, while in sumo this is from +y to +x. 
        - the coordinate system in Carla is a left-handed Cartesian coordinate system.
    """
    if states.ndim == 1:
        states[1] = -states[1]
        states[3] -= np.deg2rad(90)
    elif states.ndim == 2:
        states[:, 1] = -states[:, 1]
        states[:, 3] -= np.deg2rad(90)
    else:
        raise NotImplementedError


if __name__ == "__main__":

    train_folder = 'csv/train_pre'
    train_dataset = CarDataset(preprocess_folder=train_folder, mlp=False, mpc_aug=True)
