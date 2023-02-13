# %%

import os
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data, DataLoader
import torch
from utils.feature_utils import compute_feature_for_one_seq, encoding_features, save_features, get_intention_from_vehicle_id
from utils.config import DATA_DIR, LANE_RADIUS, OBJ_RADIUS, OBS_LEN, INTERMEDIATE_DATA_DIR, PRED_LEN, DT
from tqdm import tqdm
import pdb
from MPC_XY_Frame import linear_mpc_control
import pickle
from scipy.spatial import distance


# %%
def get_fc_edge_index(num_nodes, start=0):
    """
    return a tensor(2, edges), indicing edge_index
    """
    to_ = np.arange(num_nodes, dtype=np.int64)
    edge_index = np.empty((2, 0))
    for i in range(num_nodes):
        from_ = np.ones(num_nodes, dtype=np.int64) * i
        # FIX BUG: no self loop in ful connected nodes graphs
        edge_index = np.hstack((edge_index, np.vstack((np.hstack([from_[:i], from_[i+1:]]), np.hstack([to_[:i], to_[i+1:]])))))
    edge_index = edge_index + start

    return edge_index.astype(np.int64), num_nodes + start
# %%


class GraphData(Data):
    """
    override key `cluster` indicating which polyline_id is for the vector
    """

    def __inc__(self, key, value):
        if key == 'edge_index':
            return self.x.size(0)
        elif key == 'cluster':
            return int(self.cluster.max().item()) + 1
        else:
            return 0

# %%


class GraphDataset(InMemoryDataset):
    """
    dataset object similar to `torchvision` 
    """

    def __init__(self, root, transform=None, pre_transform=None):
        super(GraphDataset, self).__init__(root, transform, pre_transform)
        # pdb.set_trace()
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['dataset.pt']

    def download(self):
        pass

    def process(self):

        def get_data_path_ls(dir_):
            return [os.path.join(dir_, data_path) for data_path in os.listdir(dir_)]
        
        # make sure deterministic results
        data_path_ls = sorted(get_data_path_ls(self.root))

        valid_len_ls = []
        data_ls = []
        for data_p in tqdm(data_path_ls):
            
            if not data_p.endswith('pkl'):
                continue
            
            scene_name = os.path.splitext(os.path.split(data_p)[-1])[0].split('_')[-1]  
            # e.g. 'csv/val_intermediate/features_00020-00024.pkl' -> '00020-00024'
            
            x_ls = []
            y = None
            cluster = None  # indicate the vector-format feature belongs to which subgraph
            edge_index_ls = []  # fully-connected edge within each subgraph
            data = pd.read_pickle(data_p)
            
            all_in_features = data['POLYLINE_FEATURES'].values[0]   # shape=[(pred_len-1)*n, 10], [[xs, ys, xe, ye, timestamp, left, straight, right, stop, polyline_id]]
            add_len = data['TARJ_LEN'].values[0]
            cluster = all_in_features[:, -1].reshape(-1).astype(np.int32)   # cluster.shape = [vector]
            valid_len_ls.append(cluster.max())
            num_tgt_agent = data['GT'].values[0].shape[0]
            y = data['GT'].values[0].reshape(num_tgt_agent, -1).astype(np.float32)
            traj_mask = data["TRAJ_ID_TO_MASK"].values[0]   
            agent_id = 0
            edge_index_start = 0
            # assert all_in_features[agent_id][-1] == 0, f"agent id is wrong. id {agent_id}: type {all_in_features[agent_id][4]}"

            for id_, mask_ in traj_mask.items():
                data_ = all_in_features[mask_[0]:mask_[1]]  # data.shape = [vector, 10]
                edge_index_, edge_index_start = get_fc_edge_index(data_.shape[0], start=edge_index_start)   # edge_index_.shape = [2, edge]
                x_ls.append(data_)
                edge_index_ls.append(edge_index_)

            edge_index = np.hstack(edge_index_ls)
            x = np.vstack(x_ls)
            data_ls.append([x, y, cluster, edge_index, scene_name]) 
            # x.shape = [vector, 10], y.shape = [tgt_agent, pred_len*2], cluster.shape = [vector], edge_index = [2, edge]

        # [x, y, cluster, edge_index, valid_len]
        g_ls = []
        padd_to_index = np.max(valid_len_ls)    # max agent number in a scene (for the whole dataset)
        feature_len = data_ls[0][0].shape[1]    # 10
        for ind, tup in enumerate(data_ls):
            tup[0] = np.vstack([tup[0], np.zeros((padd_to_index - tup[2].max(), feature_len), dtype=tup[0].dtype)])   
            tup[2] = np.hstack([tup[2], np.arange(tup[2].max()+1, padd_to_index+1)])  # for the new added nodes, each one is a subgraph 
            g_data = GraphData(
                x = torch.from_numpy(tup[0]),
                y = torch.from_numpy(tup[1]),
                cluster = torch.from_numpy(tup[2]),
                edge_index = torch.from_numpy(tup[3]),
                valid_len = torch.tensor([valid_len_ls[ind]]),    # number of agent in each scenario (actual)
                padding_len = torch.tensor([padd_to_index + 1]), # number of agent in each scenario (padding), each scene has the same padding_len for bmm in self-atten in vectornet
                scene_name = tup[4]
            )
            g_ls.append(g_data)
        data, slices = self.collate(g_ls)
        torch.save((data, slices), self.processed_paths[0])


obs_len, pred_len, dt = OBS_LEN, PRED_LEN, DT
from utils.config import NUM_PREDICT

# def rotation_matrix(yaw):
#     rotation = np.array([[np.cos(yaw), -np.sin(yaw)],[np.sin(yaw), np.cos(yaw)]])
#     return rotation

def rotation_matrix(yaw):
    """
    Make the current direction aligns to +y axis. 
    https://en.wikipedia.org/wiki/Rotation_matrix#Non-standard_orientation_of_the_coordinate_system
    """
    rotation = np.array([[np.cos(np.pi/2-yaw), -np.sin(np.pi/2-yaw)],[np.sin(np.pi/2-yaw), np.cos(np.pi/2-yaw)]])
    return rotation

def collision_weight(x:np.ndarray, threshold=10.0, weight=5.0):
    """
    :param x: [vehicle, 7]
    :return collision_weight: [vehicle]
    """
    dists = distance.cdist(x[:,:2], x[:,:2], 'euclidean') # [vehicle, vehicle]
    np.fill_diagonal(dists, 1e6)
    v = np.unique(np.where(dists<threshold)[0])
    weights = np.ones(x.shape[0])
    weights[v] *= weight
    return weights

class CarDataset2(InMemoryDataset):  
    # read from preprocessed data
    def __init__(self, preprocess_folder, plot=False, mlp=False, mpc_aug=True):
        self.preprocess_folder = preprocess_folder
        self.plot = plot
        self.mlp = mlp
        self.mpc_aug = mpc_aug
        # self.n_mpc_aug = 1
        
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
        """ Converts raw data into GNN-readable format by constructing
        graphs out of connectivity matrices.

        """
        
        preprocess_files = os.listdir(self.preprocess_folder)
        preprocess_files.sort()
        graphs = list()
        
        if self.plot:
            fig, ax = plt.subplots()    # plot the mcp augmented data
            fig2, ax2 = plt.subplots()  # plot the rotated gt
            os.makedirs(f'images/{self.csv_folder}', exist_ok=True)

        for file in tqdm(preprocess_files):
            if os.path.splitext(file)[1] != '.pkl' or (os.path.splitext(file)[0].split('-')[-1] != '0' if not self.mpc_aug else False):
                continue
            # print(file)
            data = pickle.load(open(os.path.join(self.preprocess_folder, file), "rb"))  # tuple
            weights = collision_weight(data[0]) # [vehicle]
            if self.mlp:
                self_loop_index = (data[2][0,:] == data[2][1,:])
                graph = Data(x=data[0], y=data[1], edge_index=data[2][:,self_loop_index], path=data[3], row=data[4], weights=weights)
            else:
                # data = (x, y, edge_index, csv_file, row)
                graph = Data(x=data[0], y=data[1], edge_index=data[2], path=data[3], row=data[4], weights=weights)
            graphs.append(graph)

        data, slices = self.collate(graphs)
        torch.save((data, slices), self.processed_paths[0])


class CarDataset3(InMemoryDataset):  
    # read from preprocessed data
    def __init__(self, preprocess_folder, plot=False, mlp=False, mpc_aug=True):
        self.preprocess_folder = preprocess_folder
        self.plot = plot
        self.mlp = mlp
        self.mpc_aug = mpc_aug
        # self.n_mpc_aug = 1
        
        super().__init__(preprocess_folder)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        pt_name = 'data_0830'
        pt_name += '_mlp' if self.mlp else '_gnn'
        if self.mpc_aug:
            pt_name += '_aug'
        return [f'{pt_name}.pt']
    
    def process(self):
        """ Converts raw data into GNN-readable format by constructing
        graphs out of connectivity matrices.

        """
        
        preprocess_files = os.listdir(self.preprocess_folder)
        preprocess_files.sort()
        graphs = list()
        
        if self.plot:
            fig, ax = plt.subplots()    # plot the mcp augmented data
            fig2, ax2 = plt.subplots()  # plot the rotated gt
            os.makedirs(f'images/{self.csv_folder}', exist_ok=True)

        for file in tqdm(preprocess_files):
            if os.path.splitext(file)[1] != '.pkl':
                continue
            data = pickle.load(open(os.path.join(self.preprocess_folder, file), "rb"))  # [v, 7], [v, pred_len*6], [2, edge], t: []
            n_v = data[0].shape[0]
            weights = torch.ones(n_v)
            turn_index = (data[0][:,4] + data[0][:,6]).bool()
            center_index = (data[0][:,0].abs() < 20) * (data[0][:,1].abs() < 20)
            weights[turn_index] *= 2.5
            weights[center_index] *= 15

            if self.mlp:
                self_loop_index = (data[2][0,:] == data[2][1,:])
                graph = Data(x=data[0], y=data[1], edge_index=data[2][:,self_loop_index], t=data[3], weights=weights)
            else:
                graph = Data(x=data[0], y=data[1], edge_index=data[2], t=data[3], weights=weights)
            graphs.append(graph)

        data, slices = self.collate(graphs)
        torch.save((data, slices), self.processed_paths[0])


class CarDataset4(InMemoryDataset):  # bosch-2
    # read from preprocessed data
    def __init__(self, preprocess_folder, plot=False, mlp=False, mpc_aug=True):
        self.preprocess_folder = preprocess_folder
        self.plot = plot
        self.mlp = mlp
        self.mpc_aug = mpc_aug
        # self.n_mpc_aug = 1
        
        super().__init__(preprocess_folder)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        pt_name = 'data_0831'
        pt_name += '_mlp' if self.mlp else '_gnn'
        if self.mpc_aug:
            pt_name += '_aug'
        return [f'{pt_name}.pt']
    
    def process(self):
        """ Converts raw data into GNN-readable format by constructing
        graphs out of connectivity matrices.

        """
        
        preprocess_files = os.listdir(self.preprocess_folder)
        preprocess_files.sort()
        graphs = list()
        
        if self.plot:
            fig, ax = plt.subplots()    # plot the mcp augmented data
            fig2, ax2 = plt.subplots()  # plot the rotated gt
            os.makedirs(f'images/{self.csv_folder}', exist_ok=True)

        for file in tqdm(preprocess_files):
            if os.path.splitext(file)[1] != '.pkl':
                continue
            if not self.mpc_aug:
                if os.path.splitext(file)[0].split('-')[1] != '0':
                    continue
            data = pickle.load(open(os.path.join(self.preprocess_folder, file), "rb"))  # [v, 7], [v, pred_len*8], [2, edge], t: []
            n_v = data[0].shape[0]
            weights = torch.ones(n_v)
            turn_index = (data[0][:,4] + data[0][:,6]).bool()
            center_index = (data[0][:,0].abs() < 20) * (data[0][:,1].abs() < 20)
            weights[turn_index] *= 2.5
            weights[center_index] *= 15

            if self.mlp:
                self_loop_index = (data[2][0,:] == data[2][1,:])
                graph = Data(x=data[0], y=data[1], edge_index=data[2][:,self_loop_index], t=data[3], weights=weights)
            else:
                graph = Data(x=data[0], y=data[1], edge_index=data[2], t=data[3], weights=weights)
            # [v,7], [v, pred_len*8], [2, edge], []
            graphs.append(graph)

        data, slices = self.collate(graphs)
        torch.save((data, slices), self.processed_paths[0])


class CarDataset5(InMemoryDataset):  
    # read from preprocessed data
    def __init__(self, preprocess_folder, plot=False, mlp=False, mpc_aug=True):
        self.preprocess_folder = preprocess_folder
        self.plot = plot
        self.mlp = mlp
        self.mpc_aug = mpc_aug
        # self.n_mpc_aug = 1
        
        super().__init__(preprocess_folder)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        pt_name = 'data_0910'
        pt_name += '_mlp' if self.mlp else '_gnn'
        if self.mpc_aug:
            pt_name += '_aug'
        return [f'{pt_name}.pt']
    
    def process(self):
        """ Converts raw data into GNN-readable format by constructing
        graphs out of connectivity matrices.

        """
        
        preprocess_files = os.listdir(self.preprocess_folder)
        preprocess_files.sort()
        graphs = list()
        
        if self.plot:
            fig, ax = plt.subplots()    # plot the mcp augmented data
            fig2, ax2 = plt.subplots()  # plot the rotated gt
            os.makedirs(f'images/{self.csv_folder}', exist_ok=True)

        for file in tqdm(preprocess_files):
            if os.path.splitext(file)[1] != '.pkl':
                continue
            if not self.mpc_aug:
                if os.path.splitext(file)[0].split('-')[1] != '0':
                    continue
            data = pickle.load(open(os.path.join(self.preprocess_folder, file), "rb"))  # [v, 7], [v, pred_len*8], [2, edge], t: []
            n_v = data[0].shape[0]
            weights = torch.ones(n_v)
            turn_index = (data[0][:,4] + data[0][:,6]).bool()
            center_index1 = (data[0][:,0].abs() < 30) * (data[0][:,1].abs() < 30)
            center_index2 = (data[0][:,0].abs() < 40) * (data[0][:,1].abs() < 40)
            weights[turn_index] *= 1.5
            weights[center_index1] *= 4
            weights[center_index2] *= 4

            if self.mlp:
                self_loop_index = (data[2][0,:] == data[2][1,:])
                graph = Data(x=data[0], y=data[1], edge_index=data[2][:,self_loop_index], t=data[3], weights=weights)
            else:
                graph = Data(x=data[0], y=data[1], edge_index=data[2], t=data[3], weights=weights)
            # [v,7], [v, pred_len*8], [2, edge], []
            graphs.append(graph)

        data, slices = self.collate(graphs)
        torch.save((data, slices), self.processed_paths[0])


class CarDataset_Interaction(InMemoryDataset):  
    # read from preprocessed data
    def __init__(self, preprocess_folder, plot=False, mlp=False, mpc_aug=True):
        self.preprocess_folder = preprocess_folder
        self.plot = plot
        self.mlp = mlp
        self.mpc_aug = mpc_aug
        # self.n_mpc_aug = 1
        
        super().__init__(preprocess_folder)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        pt_name = 'data_0903'
        pt_name += '_mlp' if self.mlp else '_gnn'
        if self.mpc_aug:
            pt_name += '_aug'
        return [f'{pt_name}.pt']
    
    def process(self):
        """ Converts raw data into GNN-readable format by constructing
        graphs out of connectivity matrices.

        """
        
        preprocess_files = os.listdir(self.preprocess_folder)
        preprocess_files.sort()
        graphs = list()
        
        if self.plot:
            fig, ax = plt.subplots()    # plot the mcp augmented data
            fig2, ax2 = plt.subplots()  # plot the rotated gt
            os.makedirs(f'images/{self.csv_folder}', exist_ok=True)

        for file in tqdm(preprocess_files):
            if os.path.splitext(file)[1] != '.pkl':
                continue
            if not self.mpc_aug:
                if os.path.splitext(file)[0].split('-')[-1] != '0':
                    continue
            data = pickle.load(open(os.path.join(self.preprocess_folder, file), "rb"))  # [v, 7], [v', pred_len*7], [2, edge], mask: [v]
            n_v = data[0].shape[0]
            # weights = torch.ones(n_v)
            # turn_index = (data[0][:,4] + data[0][:,6]).bool()
            # center_index1 = (data[0][:,0].abs() < 20) * (data[0][:,1].abs() < 20)
            # center_index2 = (data[0][:,0].abs() < 40) * (data[0][:,1].abs() < 40)
            # weights[turn_index] *= 2.5
            # weights[center_index1] *= 4
            # weights[center_index2] *= 4

            if self.mlp:
                self_loop_index = (data[2][0,:] == data[2][1,:])
                graph = Data(x=data[0], y=data[1], edge_index=data[2][:,self_loop_index], mask=torch.tensor(data[3], dtype=torch.bool))
            else:
                graph = Data(x=data[0], y=data[1], edge_index=data[2], mask=torch.tensor(data[3], dtype=torch.bool))
            # [v,7], [v, pred_len*7], [2, edge], [v]
            graphs.append(graph)

        data, slices = self.collate(graphs)
        torch.save((data, slices), self.processed_paths[0])

class CarDataset_InD(InMemoryDataset):  
    # read from preprocessed data
    def __init__(self, preprocess_folder, plot=False, mlp=False, mpc_aug=True):
        self.preprocess_folder = preprocess_folder
        self.plot = plot
        self.mlp = mlp
        self.mpc_aug = mpc_aug
        # self.n_mpc_aug = 1
        
        super().__init__(preprocess_folder)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        # pt_name = 'data_rot_track8'
        # pt_name = 'data_rot_0906'
        pt_name = 'data_rot_015'
        pt_name += '_mlp' if self.mlp else '_gnn'
        if self.mpc_aug:
            pt_name += '_aug'
        return [f'{pt_name}.pt']
    
    def process(self):
        """ Converts raw data into GNN-readable format by constructing
        graphs out of connectivity matrices.
        """
        
        preprocess_files = os.listdir(self.preprocess_folder)
        preprocess_files.sort()
        graphs = list()
        
        if self.plot:
            fig, ax = plt.subplots()    # plot the mcp augmented data
            fig2, ax2 = plt.subplots()  # plot the rotated gt
            os.makedirs(f'images/{self.csv_folder}', exist_ok=True)

        for file in tqdm(preprocess_files):
            if os.path.splitext(file)[1] != '.pkl':
                continue
            # if os.path.splitext(file)[0].split('_')[0] != '08':
            #     continue
            # if not self.mpc_aug:
            #     if os.path.splitext(file)[0].split('-')[-1] != '0':
            #         continue
            data = pickle.load(open(os.path.join(self.preprocess_folder, file), "rb"))  # # [v, 9], [v', pred_len*6], [2, edge], [v]
            data[0][:,3][data[0][:,3]<0] += 2*np.pi
            n_v = data[0].shape[0]
            weights = torch.ones(n_v)

            # turn_index = (data[0][:,4] + data[0][:,6]).bool()
            # # direction_index = (data[0][:,3] > 3/4*np.pi) * (data[0][:,3] < 5/4*np.pi) + (data[0][:,3] < 1/4*np.pi) + (data[0][:,3] > 7/4*np.pi)
            # direction_index = ((data[0][:,3] < 3/4*np.pi) * (data[0][:,3] > 1/4*np.pi)) + ((data[0][:,3] > 5/4*np.pi) * (data[0][:,3] < 7/4*np.pi)) # 2
            # # center_index1 = (data[0][:,0].abs() < 20) * (data[0][:,1].abs() < 20)
            # # center_index2 = (data[0][:,0].abs() < 40) * (data[0][:,1].abs() < 40)
            # # weights[turn_index] *= 10
            # weights[turn_index] *= 3   # 2
            # weights[direction_index] *= 10

            # weights[center_index1] *= 4
            # weights[center_index2] *= 4

            if self.mlp:
                self_loop_index = (data[2][0,:] == data[2][1,:])
                graph = Data(x=data[0], y=data[1], edge_index=data[2][:,self_loop_index], mask=torch.tensor(data[3], dtype=torch.bool), weights=weights)
            else:
                graph = Data(x=data[0], y=data[1], edge_index=data[2], mask=torch.tensor(data[3], dtype=torch.bool), weights=weights)
            # [v,5], [v', pred_len*4], [2, edge], [v]
            graphs.append(graph)

        data, slices = self.collate(graphs)
        torch.save((data, slices), self.processed_paths[0])


class CarDataset_0908(InMemoryDataset):  # bosch-2
    # read from preprocessed data
    def __init__(self, preprocess_folder, plot=False, mlp=False, mpc_aug=True):
        self.preprocess_folder = preprocess_folder
        self.plot = plot
        self.mlp = mlp
        self.mpc_aug = mpc_aug
        # self.n_mpc_aug = 1
        
        super().__init__(preprocess_folder)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        pt_name = 'data_0908'
        pt_name += '_mlp' if self.mlp else '_gnn'
        if self.mpc_aug:
            pt_name += '_aug'
        return [f'{pt_name}.pt']
    
    def process(self):
        """ Converts raw data into GNN-readable format by constructing
        graphs out of connectivity matrices.

        """
        
        preprocess_files = os.listdir(self.preprocess_folder)
        preprocess_files.sort()
        graphs = list()
        
        if self.plot:
            fig, ax = plt.subplots()    # plot the mcp augmented data
            fig2, ax2 = plt.subplots()  # plot the rotated gt
            os.makedirs(f'images/{self.csv_folder}', exist_ok=True)

        for file in tqdm(preprocess_files):
            if os.path.splitext(file)[1] != '.pkl':
                continue
            if not self.mpc_aug:
                if os.path.splitext(file)[0].split('-')[1] != '0':
                    continue
            data = pickle.load(open(os.path.join(self.preprocess_folder, file), "rb"))  # [v, 7], [v, pred_len*8], [2, edge], t: []
            n_v = data[0].shape[0]
            weights = torch.ones(n_v)
            turn_index = (data[0][:,4] + data[0][:,6]).bool()
            center_index = (data[0][:,0].abs() < 20) * (data[0][:,1].abs() < 20)
            weights[turn_index] *= 2.5
            weights[center_index] *= 15

            if self.mlp:
                self_loop_index = (data[2][0,:] == data[2][1,:])
                graph = Data(x=data[0], y=data[1], edge_index=data[2][:,self_loop_index], t=data[3], weights=weights)
            else:
                graph = Data(x=data[0], y=data[1], edge_index=data[2], t=data[3], weights=weights)
            # [v,7], [v, pred_len*8], [2, edge], []
            graphs.append(graph)

        data, slices = self.collate(graphs)
        torch.save((data, slices), self.processed_paths[0])


class CarDataset(InMemoryDataset):
    def __init__(self, root, csv_folder ,transform=None, pre_transform=None, plot=False):
        self.csv_folder = csv_folder
        self.plot = plot
        self.n_mpc_aug = 1
        
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['data.pt']
    

    def process(self):
        """ Converts raw data into GNN-readable format by constructing
        graphs out of connectivity matrices.

        """
        
        csv_files = os.listdir(self.csv_folder)
        csv_files.sort()
        graphs = list()
        
        if self.plot:
            fig, ax = plt.subplots()    # plot the mcp augmented data
            fig2, ax2 = plt.subplots()  # plot the rotated gt
            os.makedirs(f'images/{self.csv_folder}', exist_ok=True)

        for csv_file in tqdm(csv_files):
            if os.path.splitext(csv_file)[1] != '.csv':
                continue
            print(csv_file)
            df = pd.read_csv(os.path.join(self.csv_folder, csv_file))
            all_features = list()
            for track_id, remain_df in df.groupby('TRACK_ID'):
                if(len(remain_df) >= (obs_len + pred_len)):
                    coords = remain_df[['X', 'Y', 'speed', 'yaw']].values   # [40, 4]
                    coords[:, 3] = np.deg2rad(coords[:, 3])
                    transform_sumo2carla(coords)
                    intention = get_intention_from_vehicle_id(track_id)[:3]
                    features = np.hstack((coords, intention * np.ones((coords.shape[0], 3))))   # [40, 7]
                    all_features.append(features)

            num_rows = features.shape[0]    # 40
            all_features = np.array(all_features)   # [vehicle, steps(40), 7]: [x, y, speed, yaw, intent, intent, intent]
            acc_delta_padding = np.empty((all_features.shape[0], all_features.shape[1], 2))
            acc_delta_padding[:] = np.NaN
            all_features = np.concatenate((all_features, acc_delta_padding), axis=-1) # [vehicle, 41, 9]: [x, y, speed, yaw, intent, intent, intent, acc, delta]
            num_cars = len(all_features)
            edges = [[x,y] for x in range(num_cars) for y in range(num_cars)]
            edge_index = torch.tensor(edges, dtype=torch.long).T    # [2, edge]
            noise_range = 3.0

            # for each timestep, create an interaction graph
            for row in range(0, num_rows - NUM_PREDICT):
                print(row)
                x = all_features[:, row, :7]   # [vehicle, 7]
                
                # translate and then rotate Gt
                y = (all_features[:, row+1:row+1+NUM_PREDICT, :2] - all_features[:, row:row+1, :2]).transpose(0,2,1) # [vehicle, pred_len, 2] -> [vehicle, 2, pred_len]
                rotations = np.array([rotation_matrix(x[i][3])  for i in range(x.shape[0])])    # [vehicle, 2, 2]
                y = (rotations @ y)         # [vehicle, 2, pred_len], transform y into local coordinate system
                y = y.transpose(0, 2, 1)    # [vehicle, pred_len, 2]
                
                # use MPC to compute acc and delta
                curr_states = all_features[:, row, :4]  # [vehicle, 4]
                future_states = all_features[:, row+1:row+1+NUM_PREDICT, :4]    # [vehicle, pred_len, 4], [x, y, speed, yaw]
                adjust_future_deltas(curr_states, future_states)
                if self.plot:
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
                shifted_curr, mpc_output = MPC_Block(curr_states, future_states, acc_delta_old, noise_range=0)    # [vehicle, 4], [vehicle, pred_len, 6]: [x, y, v, yaw, acc, delta]
                all_features[:, row+1:row+1+NUM_PREDICT, -2:] = mpc_output[:, :, -2:]   # store the control vector to accelerate future MPC opt
                speed = all_features[:, row+1:row+1+NUM_PREDICT, 2:3]   # [vehicle, pred_len, 1]
                yaw = all_features[:, row+1:row+1+NUM_PREDICT, 3:4] - all_features[:, row:row+1, 3:4] + np.pi/2     # [vehicle, pred_len, 1], align the initial direction to +y
                y = np.concatenate((y, speed, yaw, mpc_output[:, :, -2:]), axis=2).reshape(num_cars, -1) # [vehicle, pred_len*6]
                data = Data(x=torch.tensor(x, dtype=torch.float), y=torch.tensor(y, dtype=torch.float) , edge_index=edge_index, path = csv_file, row  = torch.tensor([row]))
                graphs.append(data)
                # x: [vehicle, 7]: [x_0, y_0, speed_0, yaw_0, intent, intent, intent]
                # y: [vehicle, pred_len * 6]: [[x_1, y_1, v_1, yaw_1, acc_1, delta_1, x_2, y_2...], ...]
                
                for a in range(self.n_mpc_aug):
                    shifted_curr, mpc_output = MPC_Block(curr_states, future_states, acc_delta_old, noise_range=noise_range)    # [vehicle, 4], [vehicle, pred_len, 6]: [x, y, v, yaw, acc, delta]
                    x = x.copy()    # no in-place modify
                    x[:, :2] = shifted_curr[:, :2]
                    y = (mpc_output[:, :, :2] - all_features[:, row:row+1, :2]).transpose(0,2,1)    # [vehicle, 2, pred_len]
                    y = (rotations @ y)
                    y = y.transpose(0, 2, 1)    # [vehicle, pred_len, 2]
                    if self.plot:
                        ax2.scatter(x=y[:,:,0].reshape(-1), y=y[:,:,1].reshape(-1), s=0.5, c='red')
                    mpc_output[:, :, 3:4] = mpc_output[:, :, 3:4] - all_features[:, row:row+1, 3:4] + np.pi/2
                    y = np.concatenate((y, mpc_output[:,:,2:]), axis=-1)    # [vehicle, pred_len, 6]
                    y = y.reshape(num_cars, -1)
                    data = Data(x=torch.tensor(x, dtype=torch.float), y=torch.tensor(y, dtype=torch.float) , edge_index=edge_index, path = csv_file, row  = torch.tensor([row]))
                    graphs.append(data)     # num of graph per csv: (num_rows - NUM_PREDICT) * (1 + n_mpc_aug)
                    if self.plot:
                        ax.scatter(x=shifted_curr[:,0], y=shifted_curr[:,1], s=2.0, c='red')
                        ax.scatter(x=mpc_output[:,:,0].reshape(-1), y=mpc_output[:,:,1].reshape(-1), s=0.5, c='red')

                if self.plot:
                    fig.savefig(f'images/{self.csv_folder}/{os.path.splitext(csv_file)[0]}-{str(row).zfill(3)}.png')
                    fig2.savefig(f'images/{self.csv_folder}/{os.path.splitext(csv_file)[0]}-{str(row).zfill(3)}-y.png')
                    ax.clear()
                    ax2.clear()
                    break

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
    a_opt, delta_opt, x_opt, y_opt, v_opt, yaw_opt = linear_mpc_control(target_states_v, _curr_state_v, a_old, delta_old)

    mpc_output = np.concatenate((x_opt[1:].reshape(-1, 1), y_opt[1:].reshape(-1, 1), v_opt[1:].reshape(-1, 1), yaw_opt[1:].reshape(-1, 1), \
        a_opt.reshape(-1, 1), delta_opt.reshape(-1, 1)), axis=1)
    # assert acc_delta_new_v.shape[1] == 2

    return curr_state_v.reshape(-1), mpc_output


# def MPC_Block(curr_states: np.ndarray, target_states: np.ndarray, acc_delta_old: np.ndarray, noise_range: float = 0.0):
#     """
#     :param curr_states: [vehicle, 4], [[x, y, speed, yaw], ...]
#     :param target_states: [vehicle, pred_len, 4]
#     :param acc_delta_old: [vehicle, pred_len, 2]
#     :param noise_range: noise on the lateral direction

#     :return: [vehicle, pred_len, 2], [acc, delta]
#     """

#     acc_delta_new = np.zeros_like(acc_delta_old)
#     num_vehicles = curr_states.shape[0]
#     for v in range(num_vehicles):
#         acc_delta_new[v] = MPC_module(curr_states[v], target_states[v], acc_delta_old[v], noise_range)
#     return acc_delta_new

# def MPC_module(curr_state_v: np.ndarray, target_states_v: np.ndarray, acc_delta_old_v: np.ndarray, noise_range: float = 0.0):
#     """
#     :param curr_state_v: [4], [x_0, y_0, speed_0, yaw_0]
#     :param target_states_v: [pred_len, 4], [[x_1, y_1, speed_1, yaw_1], ...]
#     :param acc_delta_old_v: [pred_len, 2], [[acc_1, delta_1], ...]
#     :param noise_range: noise on the lateral direction

#     :return: [pred_len, 2]
#     """
    
#     acc_delta_old_v[np.isnan(acc_delta_old_v)] = 0.0    # [pred_len, 2]
#     a_old = acc_delta_old_v[:, 0].tolist()
#     delta_old = acc_delta_old_v[:, 1].tolist()

#     if noise_range > 0:
#         curr_state_v = curr_state_v.copy()  # avoid add noise in-place
#         noise_direction = curr_state_v[3] - np.deg2rad(90)
#         noise_length = np.random.uniform(low=-1, high=1) * noise_range  # TODO: uniform or Gaussian distribution?
#         # noise = np.array([np.cos(noise_direction), np.sin(noise_direction)]) * noise_length
#         noise = np.array([np.sin(noise_direction), np.cos(noise_direction)]) * noise_length
#         curr_state_v[:2] += noise

#     curr_state_v = curr_state_v.reshape(1, 4)

#     target_states_v = np.concatenate((curr_state_v, target_states_v), axis=0) # [pred_len+1, 4]
#     curr_state_v = curr_state_v.reshape(-1).tolist()

#     target_states_v = target_states_v.T
#     a_opt, delta_opt, x_opt, y_opt, v_opt, yaw_opt = linear_mpc_control(target_states_v, curr_state_v, a_old, delta_old)

#     acc_delta_new_v = np.concatenate((a_opt.reshape(-1, 1), delta_opt.reshape(-1, 1)), axis=1)
#     assert acc_delta_new_v.shape[1] == 2

#     return acc_delta_new_v


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
    # for folder in os.listdir(DATA_DIR):
    #     dataset_input_path = os.path.join(
    #         INTERMEDIATE_DATA_DIR, f"{folder}_intermediate")
    #     dataset = GraphDataset(dataset_input_path)
    #     batch_iter = DataLoader(dataset, batch_size=256)
    #     batch = next(iter(batch_iter))
    # ds = CarDataset(csv_folder='csv/debug2', root='csv/debug2_intermediate', plot=True)
    # ds = CarDataset5(preprocess_folder='D:/codes/tmp/dekai_intellisys_ws2021/gnn_theta/csv/fcd_0831_aug', mlp=True, mpc_aug=True)
    # ds = CarDataset_Interaction(preprocess_folder='D:/codes/tmp/dekai_intellisys_ws2021/gnn_theta/csv/interaction_intersection_val_preprocess', mlp=True, mpc_aug=False)
    ds = CarDataset_InD(preprocess_folder='D:/codes/tmp/dekai_intellisys_ws2021/gnn_theta/csv/ind16_preprocess_rot', mlp=False, mpc_aug=False)
    # ds = CarDataset_0908(preprocess_folder='D:/codes/tmp/dekai_intellisys_ws2021/gnn_theta/csv/fcd_0831_aug', mlp=False, mpc_aug=True)
    # ds = CarDataset5(preprocess_folder='D:/codes/tmp/dekai_intellisys_ws2021/gnn_theta/csv/fcd_0831_aug', mlp=False, mpc_aug=False)
