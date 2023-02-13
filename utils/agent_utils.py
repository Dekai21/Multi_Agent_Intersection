#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020-05-27 15:00
# @Author  : Xiaoke Huang
# @Email   : xiaokehuang@foxmail.com

import numpy as np
import pandas as pd
from typing import List, Union
from utils.config import PRED_LEN, NORMALIZED_CENTER, OBS_LEN


def get_tgt_agent_feature_ls(agent_df: pd.DataFrame, 
                            obs_len: int = OBS_LEN,
                            pred_len: int = PRED_LEN, 
                            )-> List[Union[np.ndarray, str, np.ndarray, str, np.ndarray]]:
    """
    args:
    
    returns: 
        list of [doubeld_track, object_type, timetamp, track_id, not_doubled_groudtruth_feature_trajectory]
        xys: shape = [obs_len - 1, 4]
        ts: shape = [obs_len - 1]
        gt_xys: shape = [pred_len, 2]
    """
    
    xys = agent_df[["X", "Y"]].values[:obs_len]
    current_xy = agent_df[["X", "Y"]].values[obs_len - 1].reshape(1, 2)
    gt_xys = agent_df[["X", "Y"]].values[obs_len : obs_len + pred_len] - current_xy
    # xys -= norm_center  # normalize to the center of the intersection
    # gt_xys -= norm_center
    xys = np.hstack((xys[:-1], xys[1:]))    # generate double track (vector format in each row)

    ts = agent_df['TIMESTAMP'].values[:obs_len]
    ts = (ts[:-1] + ts[1:]) / 2

    return [xys, 'tgt', ts, agent_df['TRACK_ID'].iloc[0], gt_xys]
