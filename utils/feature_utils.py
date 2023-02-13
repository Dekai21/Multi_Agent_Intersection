#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020-05-27 15:00
# @Author  : Xiaoke Huang
# @Email   : xiaokehuang@foxmail.com
import os
import pdb
from typing import List, Tuple, Union

# from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
# from argoverse.map_representation.map_api import ArgoverseMap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.agent_utils import get_tgt_agent_feature_ls
from utils.config import (EXIST_THRESHOLD, FRONT_ANGLE_THRESHOLD,
                          NORMALIZED_CENTER, OBS_LEN, PRED_LEN, STEPWISE_LOSS,
                          STOP_THRESHOLD, color_dict)
# from utils.lane_utils import get_halluc_lane, get_nearby_lane_feature_ls
from utils.object_utils import (get_nearby_moving_obj_feature_ls,
                                get_non_tgt_agent_feature_ls)
from utils.viz_utils import *
from utils.viz_utils import show_doubled_lane, show_traj


def compute_feature_for_one_seq(traj_df: pd.DataFrame, 
                                obs_len: int = OBS_LEN,
                                pred_len: int = PRED_LEN, 
                                exist_threshold: int = EXIST_THRESHOLD,
                                normalized_center: np.ndarray = NORMALIZED_CENTER,
                                )-> Tuple:
    """
    Args:
        - traj_df
        - obs_len: 20
        - pred_len: 30
        - exist_threshold: 5, length threshold for non_tgt_agent
    
    returns:
        tgt_agent_feature_ls: List[List[np.ndarray, str, np.ndarray, str, np.ndarray]]
            list of [doubeld_track, object_type, timetamp, track_id, not_doubled_groudtruth_feature_trajectory]
        non_tgt_agent_feature_ls: List[List[np.ndarray, str, np.ndarray, str]]
            list of [doubeld_track, object_type, timetamp, track_id]
    """

    tgt_agent_feature_ls = []
    non_tgt_agent_feature_ls = []

    # normalize timestamps
    traj_df['TIMESTAMP'] -= np.min(traj_df['TIMESTAMP'].values)

    # normalize positions
    traj_df[['X', 'Y']] -= normalized_center

    seq_ts = np.unique(traj_df['TIMESTAMP'].values)

    city_name = traj_df['CITY_NAME'].iloc[0]
    # agent_df = None
    # agent_x_end, agent_y_end, start_x, start_y, query_x, query_y, norm_center = [None] * 7
    # agent traj & its start/end point

    # norm_center = NORMALIZED_CENTER

    for track_id, remain_df in traj_df.groupby('TRACK_ID'):
        # object_type = remain_df['OBJECT_TYPE'].unique()
        # assert len(object_type) == 1, "the agent should always have the same object type"
        # if object_type[0] == 'tgt':
        if(len(remain_df) >= (obs_len + pred_len)):
            agent_feature = get_tgt_agent_feature_ls(remain_df, obs_len, pred_len)
            tgt_agent_feature_ls.append(agent_feature)
        else:
            agent_feature = get_non_tgt_agent_feature_ls(remain_df, exist_threshold)
            non_tgt_agent_feature_ls += [agent_feature] if agent_feature is not None else []

    # assert len(tgt_agent_feature_ls) > 0, "No effective tracks."
    assert len(non_tgt_agent_feature_ls) == 0, "there shouldn't be non-tgt agent anymore (which doesn't have enough long past or future trajs)"


    # prune points after "obs_len" timestamp
    # [FIXED] test set length is only `obs_len`
    # traj_df = traj_df[traj_df['TIMESTAMP'] <=
    #                   agent_df['TIMESTAMP'].values[obs_len-1]]

    # assert (np.unique(traj_df["TIMESTAMP"].values).shape[0]
    #         == obs_len), "Obs len mismatch"

    # search nearby lane from the last observed point of agent
    # FIXME: nearby or rect?
    # lane_feature_ls = get_nearby_lane_feature_ls(
    #     am, agent_df, obs_len, city_name, lane_radius, norm_center)
    # lane_feature_ls = get_nearby_lane_feature_ls(
    #     am, agent_df, obs_len, city_name, lane_radius, norm_center, mode=mode, query_bbox=query_bbox)
    # pdb.set_trace()

    # search nearby moving objects from the last observed point of agent
    # obj_feature_ls = get_nearby_moving_obj_feature_ls(
    #     agent_df, traj_df, obs_len, seq_ts, norm_center)
    # get agent features
    # print('in compute_feature_for_one_seq...')
    # pdb.set_trace()
    # agent_feature = get_tgt_agent_feature_ls(agent_df, obs_len, norm_center)
    # av_feature = get_tgt_agent_feature_ls(av_df, obs_len, norm_center)

    # # vis
    # if viz:
    #     for features in lane_feature_ls:
    #         show_doubled_lane(
    #             np.vstack((features[0][:, :2], features[0][-1, 3:5])))
    #         show_doubled_lane(
    #             np.vstack((features[1][:, :2], features[1][-1, 3:5])))
    #     for features in obj_feature_ls:
    #         show_traj(
    #             np.vstack((features[0][:, :2], features[0][-1, 2:])), features[1])
    #     show_traj(np.vstack(
    #         (agent_feature[0][:, :2], agent_feature[0][-1, 2:])), agent_feature[1])

    #     plt.plot(agent_x_end - query_x, agent_y_end - query_y, 'o',
    #              color=color_dict['AGENT'], markersize=7)
    #     plt.plot(0, 0, 'x', color='blue', markersize=4)
    #     plt.plot(start_x-query_x, start_y-query_y,
    #              'x', color='blue', markersize=4)
    #     plt.savefig(f'./images/{name}.png')
    #     # plt.show()
    #     plt.close()     # 在远端服务器没有显示出来，也就不会按q，所以需要手动退出，否则画出来的东西会一直叠加

    # return [agent_feature, av_feature, obj_feature_ls, lane_feature_ls, norm_center]    # 1, 25, 273
    return tgt_agent_feature_ls, non_tgt_agent_feature_ls

def trans_gt_offset_format(gt):
    """
    >Our predicted trajectories are parameterized as per-stepcoordinate offsets, starting from the last observed location.
    We rotate the coordinate system based on the heading of the target vehicle at the last observed location.
    
    """
    assert gt.shape == (PRED_LEN, 2) or gt.shape == (0, 2), f"{gt.shape} is wrong"

    # for test, no gt, just return a (0, 2) ndarray
    if gt.shape == (0, 2):
        return gt

    offset_gt = np.vstack((gt[0], gt[1:] - gt[:-1]))

    assert (offset_gt.cumsum(axis=0) -
            gt).sum() < 1e-6, f"{(offset_gt.cumsum(axis=0) -gt).sum()}"

    return offset_gt


def get_direction(trajectory: np.ndarray)-> np.ndarray:
    assert trajectory.shape[1] == 2 or trajectory.shape[1] == 4, 'should be double tracked (shape[1] == 4) or gt (shape[1] == 2)'
    oldest_start_pt = trajectory[0,:2]
    newest_start_pt = trajectory[-1, :2]
    return newest_start_pt - oldest_start_pt

def get_intention(traj_direction: np.ndarray, 
                  gt_direction: np.ndarray,
                  stop_threshold: float = STOP_THRESHOLD,
                  front_angle_threshold: int = FRONT_ANGLE_THRESHOLD)-> np.ndarray:
    """
    args:

    returns:
        np.ndarray:
            one-hot vector of future intention [left-turn, straight, right-turn, stop]

    """
    intention = np.zeros(4)

    cos = np.dot(traj_direction, gt_direction) / (np.linalg.norm(traj_direction)*np.linalg.norm(gt_direction))
    sin = np.cross(traj_direction, gt_direction) / (np.linalg.norm(traj_direction)*np.linalg.norm(gt_direction))

    # stop intention
    if(np.linalg.norm(gt_direction) < stop_threshold):
        intention[3] = 1
        return intention

    cos = np.abs(cos)
    angle = np.degrees(np.arccos(cos))

    if angle < front_angle_threshold:
        intention[1] = 1
    else:
        if sin > 0:
            intention[0] = 1
        else:
            intention[2] = 1
    return intention

def get_intention_from_vehicle_id(vehicle_id: str)-> np.ndarray:
    """
    Parse the vehicle id to distinguish its intention.
    """
    intention = np.zeros(4)

    from_path, to_path, _ = vehicle_id.split('_')
    if from_path == 'left':
        if to_path == 'right':
            intention_str = 'straight'
        elif to_path == 'up':
            intention_str = 'left'
        elif to_path == 'down':
            intention_str = 'right'

    elif from_path == 'right':
        if to_path == 'left':
            intention_str = 'straight'
        elif to_path == 'up':
            intention_str = 'right'
        elif to_path == 'down':
            intention_str = 'left'

    elif from_path == 'up':
        if to_path == 'down':
            intention_str = 'straight'
        elif to_path == 'left':
            intention_str = 'right'
        elif to_path == 'right':
            intention_str = 'left'

    elif from_path == 'down':
        if to_path == 'up':
            intention_str = 'straight'
        elif to_path == 'right':
            intention_str = 'right'
        elif to_path == 'left':
            intention_str = 'left'

    else:
        raise Exception('Wrong vehicle id')

    if intention_str == 'left':
        intention[0] = 1
    elif intention_str == 'straight':
        intention[1] = 1
    elif intention_str == 'right':
        intention[2] = 1
    
    return intention

def encoding_features(agent_feature, av_feature, obj_feature_ls, lane_feature_ls, intention_counter: np.ndarray):
    """
    args:
        agent_feature_ls:
            list of (doubeld_track, object_type, timestamp, track_id, not_doubled_groudtruth_feature_trajectory)
        obj_feature_ls:
            list of list of (doubled_track, object_type, timestamp, track_id)
        lane_feature_ls:
            list of list of lane a segment feature, formatted in [left_lane, right_lane, is_traffic_control, is_intersection, lane_id]
    returns:
        pd.DataFrame of (
            polyline_features: vstack[
                (xs, ys, xe, ye, timestamp, NULL, NULL, polyline_id),
                (xs, ys, xe, ye, NULL, zs, ze, polyline_id)
                ]
            offset_gt: incremental offset from agent's last obseved point,
            traj_id2mask: Dict[int, int]
            lane_id2mask: Dict[int, int]
        )
        where obejct_type = {0 - others, 1 - agent}

    """
    polyline_id = 0
    traj_id2mask, lane_id2mask = {}, {}
    agent_gt = agent_feature[-1]  # original version: gt(agent).shape = (30, 2), type = numpy.ndarray
    av_gt = av_feature[-1]
    traj_nd, lane_nd = np.empty((0, 7)), np.empty((0, 7))

    # encoding agent feature
    # pre_traj_len = traj_nd.shape[0]     # 0
    # agent_len = agent_feature[0].shape[0]   # agent_feature[0].shape = [19, 4]
    # # print(agent_feature[0].shape, np.ones(
    # # (agent_len, 1)).shape, agent_feature[2].shape, (np.ones((agent_len, 1)) * polyline_id).shape)
    # agent_nd = np.hstack((agent_feature[0], np.ones(
    #     (agent_len, 1)), agent_feature[2].reshape((-1, 1)), np.ones((agent_len, 1)) * polyline_id))
    # assert agent_nd.shape[1] == 7, "obj_traj feature dim 1 is not correct"

    # traj_nd = np.vstack((traj_nd, agent_nd))
    # traj_id2mask[polyline_id] = (pre_traj_len, traj_nd.shape[0])
    # pre_traj_len = traj_nd.shape[0]
    # polyline_id += 1

    # print('After encoding agent features...')
    # pdb.set_trace()

    # Begin: encoding av feature
    pre_traj_len = traj_nd.shape[0]
    av_len = av_feature[0].shape[0]   # agent_feature[0].shape = [19, 4]
    av_nd = np.hstack((av_feature[0], np.ones(
        (av_len, 1)), av_feature[2].reshape((-1, 1)), np.ones((av_len, 1)) * polyline_id))
    assert av_nd.shape[1] == 7, "obj_traj feature dim 1 is not correct"

    traj_nd = np.vstack((traj_nd, av_nd))
    traj_id2mask[polyline_id] = (pre_traj_len, traj_nd.shape[0])    # 大概是用来记录一个subgraph有多少节点
    pre_traj_len = traj_nd.shape[0]
    polyline_id += 1
    # End: encoding av feature

    # Compute the intention
    traj_direction = get_traj_direction(av_feature[0])
    gt_direction = get_gt_direction(av_gt)

    intention = get_intention(traj_direction, gt_direction)
    intention_counter += intention

    # encoding obj feature
    pre_traj_len = traj_nd.shape[0]
    for obj_feature in obj_feature_ls:
        obj_len = obj_feature[0].shape[0]
        # assert obj_feature[2].shape[0] == obj_len, f"obs_len of obj is {obj_len}"
        if not obj_feature[2].shape[0] == obj_len:
            from pdb import set_trace;set_trace()
        obj_nd = np.hstack((obj_feature[0], np.zeros(
            (obj_len, 1)), obj_feature[2].reshape((-1, 1)), np.ones((obj_len, 1)) * polyline_id))
        assert obj_nd.shape[1] == 7, "obj_traj feature dim 1 is not correct"
        traj_nd = np.vstack((traj_nd, obj_nd))

        traj_id2mask[polyline_id] = (pre_traj_len, traj_nd.shape[0])
        pre_traj_len = traj_nd.shape[0]
        polyline_id += 1

    # incodeing lane feature
    pre_lane_len = lane_nd.shape[0]
    for lane_feature in lane_feature_ls:
        l_lane_len = lane_feature[0].shape[0]   # 9
        l_lane_nd = np.hstack(
            (lane_feature[0], np.ones((l_lane_len, 1)) * polyline_id))  # (9, 7)
        assert l_lane_nd.shape[1] == 7, "obj_traj feature dim 1 is not correct"
        lane_nd = np.vstack((lane_nd, l_lane_nd))
        lane_id2mask[polyline_id] = (pre_lane_len, lane_nd.shape[0])
        _tmp_len_1 = pre_lane_len - lane_nd.shape[0]
        pre_lane_len = lane_nd.shape[0]
        polyline_id += 1

        r_lane_len = lane_feature[1].shape[0]
        r_lane_nd = np.hstack(
            (lane_feature[1], np.ones((r_lane_len, 1)) * polyline_id)
        )
        assert r_lane_nd.shape[1] == 7, "obj_traj feature dim 1 is not correct"
        lane_nd = np.vstack((lane_nd, r_lane_nd))
        lane_id2mask[polyline_id] = (pre_lane_len, lane_nd.shape[0])    # {17: (0, 9), 18: (9, 18), 19: (18, 27), 20: (27, 36)}
        _tmp_len_2 = pre_lane_len - lane_nd.shape[0]
        pre_lane_len = lane_nd.shape[0]
        polyline_id += 1

        # print('in encoding the lane features...')
        # pdb.set_trace()
        assert _tmp_len_1 == _tmp_len_2, f"left, right lane vector length contradict"
        # lane_nd = np.vstack((lane_nd, l_lane_nd, r_lane_nd))

    # FIXME: handling `nan` in lane_nd
    col_mean = np.nanmean(lane_nd, axis=0)
    if np.isnan(col_mean).any():
        # raise ValueError(
        # print(f"{col_mean}\nall z (height) coordinates are `nan`!!!!")
        lane_nd[:, 2].fill(.0)
        lane_nd[:, 5].fill(.0)
    else:
        inds = np.where(np.isnan(lane_nd))
        lane_nd[inds] = np.take(col_mean, inds[1])

    # traj_ls, lane_ls = reconstract_polyline(
    #     np.vstack((traj_nd, lane_nd)), traj_id2mask, lane_id2mask, traj_nd.shape[0])
    # type_ = 'AGENT'
    # for traj in traj_ls:
    #     show_traj(traj, type_)
    #     type_ = 'OTHERS'

    # for lane in lane_ls:
    #     show_doubled_lane(lane)
    # plt.show()

    # transform gt to offset_gt
    offset_gt = trans_gt_offset_format(av_gt)

    # now the features are:
    # (xs, ys, xe, ye, obejct_type, timestamp(avg_for_start_end?),polyline_id) for object
    # (xs, ys, zs, xe, ye, ze, polyline_id) for lanes

    # change lanes feature to xs, ys, xe, ye, NULL, zs, ze, polyline_id)
    lane_nd = np.hstack(
        [lane_nd, np.zeros((lane_nd.shape[0], 2), dtype=lane_nd.dtype)])
    lane_nd = lane_nd[:, [0, 1, 3, 4, 7, 8, 2, 5, 6]]
    # change object features to (xs, ys, xe, ye, timestamp, NULL, NULL, polyline_id)
    traj_nd = np.hstack(
        [traj_nd, np.zeros((traj_nd.shape[0], 4), dtype=traj_nd.dtype)])
    traj_nd = traj_nd[:, [0, 1, 2, 3, 5, 7, 8, 9, 10, 6]]
    # traj_nd.shape = [19, 8]
    # lane_nd.shape = [558, 8]

    for row in range(av_len):
        traj_nd[row, 5:9] = intention

    # don't ignore the id
    # polyline_features = np.vstack((traj_nd, lane_nd))
    polyline_features = traj_nd
    data = [[polyline_features.astype(np.float32), offset_gt, traj_id2mask, lane_id2mask, traj_nd.shape[0], lane_nd.shape[0]]]
    # polyline_features.shape = [577, 8]

    # print('in feature_utils.py...')
    # pdb.set_trace()
    return pd.DataFrame(
        data,
        columns=["POLYLINE_FEATURES", "GT",
                 "TRAJ_ID_TO_MASK", "LANE_ID_TO_MASK", "TARJ_LEN", "LANE_LEN"]
    )

def encoding_features2(tgt_agent_feature_ls: List[List[Union[np.ndarray, str, np.ndarray, str, np.ndarray]]], 
                      non_tgt_agent_feature_ls: List[List[Union[np.ndarray, str, np.ndarray, str]]], 
                      intention_counter: np.ndarray
                      )-> pd.DataFrame:
    """
    Args:
        tgt_agent_feature_ls: list of [doubeld_track, object_type, timestamp, track_id, not_doubled_groudtruth_feature_trajectory]
        non_tgt_agent_feature_ls: list of [doubled_track, object_type, timestamp, track_id]
        intention_counter: 
    
    Returns:
        pd.DataFrame of (
            polyline_features: vstack[[xs, ys, xe, ye, timestamp, left, straight, right, stop, polyline_id]], shape = [vector, 10]
            offset_gt: list of incremental offset from agent's last obseved point, [[30, 2], [30, 2], ....], shape = [tgt_agent, 30, 2]
            traj_id2mask: Dict[int, int], len = #agent (including tgt and non-tgt)
            traj_len: #vector
        )
    """
    polyline_id = 0
    traj_id2mask = {}
    traj_nd = np.empty((0, 10))
    offset_gts = np.empty((0, PRED_LEN, 2))

    for tgt_agent_feature in tgt_agent_feature_ls:  # TODO: 增加以下代码的可读性
        pre_traj_len = traj_nd.shape[0]
        agent_len = tgt_agent_feature[0].shape[0]   # 19
        assert agent_len == OBS_LEN - 1
        traj_direction = get_direction(tgt_agent_feature[0])
        gt_direction = get_direction(tgt_agent_feature[-1])
        # intention = get_intention(traj_direction, gt_direction, STOP_THRESHOLD, FRONT_ANGLE_THRESHOLD)
        intention = get_intention_from_vehicle_id(tgt_agent_feature[3])
        intention_counter += intention
        intentions = np.vstack([intention for _ in range(agent_len)])
        assert intentions.shape == (OBS_LEN - 1, 4)
        agent_nd = np.hstack((tgt_agent_feature[0], tgt_agent_feature[2].reshape(-1, 1), intentions, np.ones((agent_len, 1)) * polyline_id))
        assert agent_nd.shape == (OBS_LEN - 1, 10)
        traj_nd = np.vstack((traj_nd, agent_nd))
        traj_id2mask[polyline_id] = (pre_traj_len, traj_nd.shape[0])    # to record the begin and end row for tgt agent, e.g. 0: (0, 19), 1: (19, 38), ...
        pre_traj_len = traj_nd.shape[0]
        polyline_id += 1
        gt = tgt_agent_feature[-1]
        if STEPWISE_LOSS:
            offset_gt = np.expand_dims(trans_gt_offset_format(gt), axis=0) # shape = [1, 30, 2]
        else:
            offset_gt = np.expand_dims(gt, axis=0) # shape = [1, 30, 2]
        assert offset_gt.shape == (1, PRED_LEN, 2)
        offset_gts = np.concatenate((offset_gts, offset_gt), axis=0)

    for non_tgt_agent_feature in non_tgt_agent_feature_ls:
        pre_traj_len = traj_nd.shape[0]
        agent_len = non_tgt_agent_feature[0].shape[0]   
        agent_nd = np.hstack((non_tgt_agent_feature[0], non_tgt_agent_feature[2].reshape(-1, 1), np.zeros((agent_len, 4)), np.ones((agent_len, 1)) * polyline_id))
        traj_nd = np.vstack((traj_nd, agent_nd))
        traj_id2mask[polyline_id] = (pre_traj_len, traj_nd.shape[0])
        pre_traj_len = traj_nd.shape[0]
        polyline_id += 1


    polyline_features = traj_nd
    data = [[polyline_features.astype(np.float32), offset_gts, traj_id2mask, traj_nd.shape[0]]]

    return pd.DataFrame(
        data,
        columns=["POLYLINE_FEATURES", "GT", "TRAJ_ID_TO_MASK", "TARJ_LEN"]
    )


def save_features(df, name:str, dir_=None):
    """
    Args:
        - df
        - name: f'{begin_time}-{end_time}', e.g. '400-406'
    """

    if dir_ is None:
        dir_ = './input_data'
    os.makedirs(dir_, exist_ok=True)

    begin_time, end_time = name.split('-')
    begin_time, end_time = int(begin_time), int(end_time)

    name = "features_%05d-%05d.pkl"%(begin_time, end_time)
    df.to_pickle(
        os.path.join(dir_, name)
    )
