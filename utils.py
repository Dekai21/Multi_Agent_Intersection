import numpy as np
import torch
from torch_geometric.nn import GraphConv as GNNConv
from torch import nn
from typing import Dict, List, Tuple, Union
import traci  # pylint: disable=import-error
from scipy.spatial import distance
from torch_geometric.data import Data

NORMALIZED_CENTER = np.array([0.0, 0.0])

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(42)
        self.conv1 = GNNConv(5, hidden_channels)
        self.conv2 = GNNConv(hidden_channels, hidden_channels)
        self.conv3 = GNNConv(hidden_channels, 30*2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        return x

class GNN_mtl(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(21)
        # self.conv1 = GNNConv(6, hidden_channels)
        self.conv2 = GNNConv(hidden_channels, hidden_channels)
        self.conv3 = GNNConv(hidden_channels, hidden_channels)
        # self.conv3 = GNNConv(hidden_channels, 50*2)
        self.linear1 = nn.Linear(5, 64)
        self.linear2 = nn.Linear(64, hidden_channels)
        self.linear3 = nn.Linear(hidden_channels, 30*2)
        self.linear4 = nn.Linear(hidden_channels, hidden_channels)
        self.linear5 = nn.Linear(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = self.linear1(x)
        x = x.relu()
        x = self.linear2(x)
        x = x.relu()
        x = self.linear4(x).relu() + x
        x = self.linear5(x).relu() + x
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = self.linear3(x)
        return x  # mtl

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
        # self.conv1 = GNNConv(hidden_channels, hidden_channels)
        # self.conv2 = GNNConv(hidden_channels, hidden_channels)
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
        # x = self.conv1(x, edge_index).relu()
        # x = self.conv2(x, edge_index).relu()
        x = self.conv1(x).relu()
        x = self.conv2(x).relu()
        x = self.linear5(x)
        return x  # mtl

def get_intention_from_vehicle_id(vehicle_id: str)-> str:
    """
    Parse the vehicle id to distinguish its intention.
    """

    from_path, to_path, _ = vehicle_id.split('_')
    if from_path == 'left':
        if to_path == 'right':
            return 'straight'
        elif to_path == 'up':
            return 'left'
        elif to_path == 'down':
            return 'right'

    elif from_path == 'right':
        if to_path == 'left':
            return 'straight'
        elif to_path == 'up':
            return 'right'
        elif to_path == 'down':
            return 'left'

    elif from_path == 'up':
        if to_path == 'down':
            return 'straight'
        elif to_path == 'left':
            return 'right'
        elif to_path == 'right':
            return 'left'

    elif from_path == 'down':
        if to_path == 'up':
            return 'straight'
        elif to_path == 'right':
            return 'right'
        elif to_path == 'left':
            return 'left'

    raise Exception('Wrong vehicle id')


def get_intention_vector(intention: str = 'straight')-> np.ndarray:
    """
    Return a 3-bit one-hot format intention vector.
    """

    intention_feature = np.zeros(3)   # rest
    # intention_feature = np.zeros(4) # vectornet
    if intention == 'left':
        intention_feature[0] = 1
    elif intention == 'straight':
        intention_feature[1] = 1
    elif intention == 'right':
        intention_feature[2] = 1
    # elif intention == 'stop':
    #     intention_feature[3] = 1
    elif intention == 'null':
        None
    else:
        raise NotImplementedError
    return intention_feature

def rotation_matrix_back(yaw):
    """
    Rotate back. 
    https://en.wikipedia.org/wiki/Rotation_matrix#Non-standard_orientation_of_the_coordinate_system
    """
    rotation = np.array([[np.cos(-np.pi/2+yaw), -np.sin(-np.pi/2+yaw)],[np.sin(-np.pi/2+yaw), np.cos(-np.pi/2+yaw)]])
    return rotation

def get_yaw(vehicle_id: str, pos: np.ndarray, yaw_dict: dict):
    """
    建议一个[x,y,yaw]的数据库, 存储在字典中, 字典的key为vehicle_id的前两位
    """
    # t = 1
    route = '_'.join(vehicle_id.split('_')[:-1])
    yaws = yaw_dict[route]
    # for yaw in yaws:
    #     if np.linalg.norm(yaw[:2]-pos) < 1.5:
    #         return yaw[-1]
    dists = distance.cdist(pos.reshape(1,2), yaws[:,:-1])
    return yaws[np.argmin(dists),-1]
    # return None

def update_trajs(vehicle_ids: List, time: float, trajs: Dict, yaw_dict: dict)-> Dict:
    """
    Update the dict trajs, e.g. {'left_0': [(x0, y0, speed, yaw, yaw(sumo-degree), intention(str)), (x1, y1, speed, yaw, yaw(sumo-degree), intention(str)), ...], 'left_1': [...]}
    """
    
    # add the new vehicles in the scene
    for vehicle_id in vehicle_ids:
        if 'carla' in vehicle_id:   # 上次仿真的残留
            continue
        if vehicle_id not in trajs.keys():
            trajs[vehicle_id] = []
        pos = traci.vehicle.getPosition(vehicle_id)
        speed = traci.vehicle.getSpeed(vehicle_id)
        # yaw = np.deg2rad(traci.vehicle.getAngle(vehicle_id))
        yaw_sumo_degree = traci.vehicle.getAngle(vehicle_id)
        yaw = np.deg2rad(get_yaw(vehicle_id, np.array(pos), yaw_dict))
        norm_x, norm_y = NORMALIZED_CENTER
        already_steps = len(trajs[vehicle_id])
        if already_steps == 0:
            intention = get_intention_from_vehicle_id(vehicle_id)
            trajs[vehicle_id].append((pos[0] - norm_x, pos[1] - norm_y, speed, yaw, yaw_sumo_degree, intention)) 
        else:
            intention = trajs[vehicle_id][-1][-1]
            assert isinstance(intention, str)
            trajs[vehicle_id].append((pos[0] - norm_x, pos[1] - norm_y, speed, yaw, yaw_sumo_degree, intention))    # TODO: normalize the time

    # remove the vehicles out of the scene
    for vehicle_id in list(trajs):
        if vehicle_id not in vehicle_ids:
            del trajs[vehicle_id]
    
    return trajs

def encoding_scenario_features(trajs: Dict, time: float)-> Tuple[np.ndarray, int, int, np.ndarray, List[str]]:
    """
    Args:
        - trajs: e.g. {'left_0': [(x0, y0, speed, yaw, yaw', intention(str)), (x1, y1, speed, yaw, yaw', intention(str)), ...], 'left_1': [...]}
        - time: current time (second)
    
    Returns:
        - x: [[xs, ys, xe, ye, timestamp, left, straight, right, stop, polyline_id], ...], x.shape = [N, 10]
        - num_tgt_agent
        - num_agent
        - edge_indexs: shape = [2, edges]
        - tgt_agent_ids
    """

    # num_tgt_agent, num_agent = 0, 0
    x = np.empty((0, 7))
    # edge_indexs = np.empty((2, 0))
    tgt_agent_ids = []

    for vehicle_id in trajs.keys():
        # print(vehicle_id)
        if np.linalg.norm(trajs[vehicle_id][-1][:2]) < 65:
            # begin = x.shape[0]
            # num_tgt_agent += 1
            # num_agent += 1
            _x = np.concatenate((np.array(trajs[vehicle_id][-1][:-2]), get_intention_vector(trajs[vehicle_id][-1][-1]))).reshape(1, -1) # [1, 7]
            x = np.vstack((x, _x))
            # x = encoding_tgt_agent_features(trajs[vehicle_id][-2:], x)
            # print(x.shape)
            # num_node = x.shape[0] - begin
            # edge_index = get_fc_edge_index(num_node, start=begin)
            # edge_indexs = np.hstack((edge_indexs, edge_index))
            tgt_agent_ids.append(vehicle_id)
        # elif ((time - trajs[vehicle_id][0][2]) < (OBS_LEN / 10)) and ((time - trajs[vehicle_id][0][2]) >= 0.2):
        #     begin = x.shape[0]
        #     num_agent += 1
        #     x = encoding_non_tgt_agent_features(trajs[vehicle_id], x)
        #     num_node = x.shape[0] - begin
        #     edge_index = get_fc_edge_index(num_node, start=begin)
        #     edge_indexs = np.hstack((edge_indexs, edge_index))

    # x[:, 4] -= (time - OBS_LEN / 10)    # normalize the time to the range [0.0, 1.9]

    # x = x[:, [2,3,0,1,5,6,7]].astype(np.float32)

    return x, tgt_agent_ids

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

def remove_vehicles(vehicle_ids: List, trajs: Dict):
    """
    Remove the vehicles which are getting stuck.
    """

    for vehicle_id in vehicle_ids:
        if 'carla' in vehicle_id:
            continue
        # remove the vehicle waiting too long
        if traci.vehicle.getWaitingTime(vehicle_id) > 25 and traci.vehicle.getDistance(vehicle_id) > 5:
        # if traci.vehicle.getWaitingTime(vehicle_id) > 12:    
            traci.vehicle.remove(vehicle_id)
            continue

        # x, y = traci.vehicle.getPosition(vehicle_id)
        # curr_pos = np.array([x, y])
        # dist = np.linalg.norm(curr_pos - NORMALIZED_CENTER)
        # if dist > 55 and traci.vehicle.getDistance(vehicle_id) > 50: 
        #     traci.vehicle.remove(vehicle_id)
        #     continue

        traj_len = len(trajs[vehicle_id])
        # remove the vehicle staying too long (more than 45 seconds)
        # if (trajs[vehicle_id][traj_len - 1][2] - trajs[vehicle_id][0][2]) > 45.0:
        #     traci.vehicle.remove(vehicle_id)
        #     continue
        if traj_len > 2:
            if 270 > abs(trajs[vehicle_id][traj_len - 1][-2] - trajs[vehicle_id][traj_len - 2][-2]) > 135:  # drive on the opposite lane
                traci.vehicle.remove(vehicle_id)
                continue

def get_double_tracks_and_ts(raw_feature: List[Tuple])-> np.ndarray:
    """
    Args:
        - raw_feature: [(x0, y0, timestamp0, intention), (x1, y1, timestamp1, intention), ...]
    
    Returns:
        - xys_ts: [[xs, ys, xe, ye, time], [xs, ys, xe, ye, time], ...]
    """

    _raw_feature = np.asarray(raw_feature)[:, :3].astype(np.float)
    xys = _raw_feature[:, :2]
    ts = _raw_feature[:, 2]
    xys = np.hstack((xys[:-1], xys[1:]))
    ts = (ts[:-1] + ts[1:]) / 2
    ts = ts.reshape(-1, 1)
    xys_ts = np.hstack((xys, ts))
    assert xys_ts.shape[1] == 5
    return xys_ts

def get_polyline_id(x: np.ndarray)-> int:
    """
    Args:
        - x: [xs, ys, xe, ye, timestamp, left, straight, right, stop, polyline_id], x.shape = [N, 10]
    """

    assert x.shape[1] == 10
    polyline_id = None
    if x.shape[0] == 0:
        polyline_id = 0
    else:
        polyline_id = int(x[-1, -1]) + 1
    
    return polyline_id

def encoding_agent_features(raw_feature: List[Tuple], x: np.ndarray, intention: str)-> np.array:
    """
    Stack the agent features. 

    Args:
        - raw_feature: [(x0, y0, timestamp0), (x1, y1, timestamp1), ...]
        - x: [[xs, ys, xe, ye, timestamp, left, straight, right, stop, polyline_id], ...], x.shape = [N, 10]
        - intention: e.g. 'left'

    Return:
        - x
    """

    xys_ts = get_double_tracks_and_ts(raw_feature)
    intention = get_intention_vector(intention)
    polyline_id = get_polyline_id(x)
    intention_polyline_id = np.append(intention, polyline_id).reshape(1, 5)
    intention_polyline_id = np.repeat(intention_polyline_id, xys_ts.shape[0], axis=0)
    agent_feature = np.hstack((xys_ts, intention_polyline_id))
    x = np.vstack((x, agent_feature))
    assert x.shape[1] == 10
    return x

def encoding_tgt_agent_features(raw_feature: List[Tuple], x: np.ndarray):
    """
    Args:
        - raw_feature: [(x0, y0, timestamp0, intention), (x1, y1, timestamp1, intention), ...]
        - x: [[xs, ys, xe, ye, timestamp, left, straight, right, stop, polyline_id], ...], x.shape = [N, 10]
    """

    x = encoding_agent_features(raw_feature, x, raw_feature[0][3])
    
    return x

def get_fc_edge_index(num_nodes: int, start: int = 0)-> np.ndarray:
    """
    Get the edge index (u, v) for the fully-connected subgraph.

    Return: 
        - edge_index: shape = [2, edges]
    """

    u = np.arange(num_nodes, dtype=np.int64).reshape(1, -1)
    _u = np.repeat(u, num_nodes, axis=1).reshape(-1, 1)
    _v = np.repeat(u, num_nodes, axis=0).reshape(-1, 1)
    edge_index = np.hstack((_u, _v))
    mask = (edge_index[:, 0] != edge_index[:, 1])
    edge_index = edge_index[mask]
    edge_index += start
    edge_index = edge_index.T

    return edge_index

def encoding_scenario_features_vectornet(trajs: Dict, time: float)-> Tuple[np.ndarray, int, int, np.ndarray, List[str]]:
    """
    Args:
        - trajs: e.g. {'left_0': [(x0, y0, timestamp0, intention), (x1, y1, timestamp1, intention), ...], 'left_1': [...]}
        - time: current time (second)
    
    Returns:
        - x: [[xs, ys, xe, ye, timestamp, left, straight, right, stop, polyline_id], ...], x.shape = [N, 10]
        - num_tgt_agent
        - num_agent
        - edge_indexs: shape = [2, edges]
        - tgt_agent_ids
    """

    num_tgt_agent, num_agent = 0, 0
    x = np.empty((0, 10))
    edge_indexs = np.empty((2, 0))
    tgt_agent_ids = []

    for vehicle_id in trajs.keys():
        if (time - trajs[vehicle_id][0][2]) >= (20 / 10):
            begin = x.shape[0]
            num_tgt_agent += 1
            num_agent += 1
            x = encoding_tgt_agent_features(trajs[vehicle_id][-20:], x)
            num_node = x.shape[0] - begin
            edge_index = get_fc_edge_index(num_node, start=begin)
            edge_indexs = np.hstack((edge_indexs, edge_index))
            tgt_agent_ids.append(vehicle_id)
        # elif ((time - trajs[vehicle_id][0][2]) < (OBS_LEN / 10)) and ((time - trajs[vehicle_id][0][2]) >= 0.2):
        #     begin = x.shape[0]
        #     num_agent += 1
        #     x = encoding_non_tgt_agent_features(trajs[vehicle_id], x)
        #     num_node = x.shape[0] - begin
        #     edge_index = get_fc_edge_index(num_node, start=begin)
        #     edge_indexs = np.hstack((edge_indexs, edge_index))

    x[:, 4] -= (time - 20 / 10)    # normalize the time to the range [0.0, 1.9]

    return x, num_tgt_agent, num_agent, edge_indexs, tgt_agent_ids

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

def get_graph_data_vectornet(x: np.ndarray, num_tgt_agent: int, num_agent: int, edge_indexs: np.ndarray):
    """
    Generate torch_geometric format input data.
    """

    y = np.zeros((num_tgt_agent, 30*2))

    g_data = GraphData(
        x=torch.from_numpy(x[:,:4]).float(),
        y=torch.from_numpy(y),   # y.shape[0] = #tgt_agent
        cluster=torch.from_numpy(x[:, -1]),
        edge_index=torch.from_numpy(edge_indexs).long(),
        valid_len=torch.tensor([num_agent]),    # number of agent in each scenario
        padding_len=torch.tensor([num_agent]), # padding_len is the number of agent after padding, since there is no padding here, so valid_len = padding_len
        num_graphs=1
    )

    return g_data

def get_average_offset_vectornet(out: torch.Tensor)-> Tuple[Union[float, float]]:
    
    assert len(out) == 30 * 2
    out = out[:2]   # now only the first step prediction is used, if a average value of multiple steps is needed, set the value to 4, 6, 8...
    out = out.reshape((-1, 2))
    offset = torch.mean(out, dim=0) 

    return offset

def motion_filter_vectornet(delta_x: float, delta_y: float, vehicle_id: str, trajs: Dict):
    """
    If generating a prediction of reversing, change the target motion to a slightly forward movement.

    Args:
        - delta_x
        - delta_y
        - vehicle_id
        - trajs: e.g. {'left_0': [(x0, y0, timestamp0, intention0), (x1, y1, timestamp1, intention0), ...], 'left_1': [...]}
    """

    delta_x = delta_x.detach().numpy()
    delta_y = delta_y.detach().numpy()
    pred_motion = np.array([delta_x, delta_y])

    len_traj = len(trajs[vehicle_id])
    last_delta_x = trajs[vehicle_id][len_traj - 1][0] - trajs[vehicle_id][len_traj - 2][0]
    last_delta_y = trajs[vehicle_id][len_traj - 1][1] - trajs[vehicle_id][len_traj - 2][1]
    last_motion = np.array([last_delta_x, last_delta_y])

    i = 0
    while np.linalg.norm(last_motion) < 0.0001:  # avoid the bug in sumo (last step is already a static one)
        i += 1
        last_delta_x = trajs[vehicle_id][len_traj - 1][0] - trajs[vehicle_id][len_traj - i - 2][0]
        last_delta_y = trajs[vehicle_id][len_traj - 1][1] - trajs[vehicle_id][len_traj - i - 2][1]
        last_motion = np.array([last_delta_x, last_delta_y])

    cos_sim = np.dot(pred_motion, last_motion) / (np.linalg.norm(pred_motion) * np.linalg.norm(last_motion))

    if cos_sim < 0:
        last_motion_length = np.linalg.norm(last_motion)
        last_delta_x = last_delta_x / last_motion_length * 0.01
        last_delta_y = last_delta_y / last_motion_length * 0.01
        return last_delta_x, last_delta_y
    else:
        return delta_x, delta_y

def load_checkpoint_vectornet(checkpoint_path, model, optimizer = None):
    state = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(state['state_dict'])
    # optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)

def update_trajs_vectornet(vehicle_ids: List, time: float, trajs: Dict)-> Dict:
    """
    Update the dict trajs, e.g. {'left_0': [(x0, y0, timestamp0, intention), (x1, y1, timestamp1, intention), ...], 'left_1': [...]}
    """
    
    # add the new vehicles in the scene
    for vehicle_id in vehicle_ids:
        if vehicle_id not in trajs.keys():
            trajs[vehicle_id] = []
        pos = traci.vehicle.getPosition(vehicle_id)
        norm_x, norm_y = NORMALIZED_CENTER
        already_steps = len(trajs[vehicle_id])
        if already_steps == 0:
            intention = get_intention_from_vehicle_id(vehicle_id)
            trajs[vehicle_id].append((pos[0] - norm_x, pos[1] - norm_y, time, intention)) 
        else:
            intention = trajs[vehicle_id][-1][-1]
            assert isinstance(intention, str)
            trajs[vehicle_id].append((pos[0] - norm_x, pos[1] - norm_y, time, intention))    # TODO: normalize the time

    # remove the vehicles out of the scene
    for vehicle_id in list(trajs):
        if vehicle_id not in vehicle_ids:
            del trajs[vehicle_id]
    
    return trajs