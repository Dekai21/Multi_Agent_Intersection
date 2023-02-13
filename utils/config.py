#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020-05-27 15:00
# @Author  : Xiaoke Huang
# @Email   : xiaokehuang@foxmail.com

import numpy as np

RAW_DATA_FORMAT = {
    "TIMESTAMP": 0,
    "TRACK_ID": 1,
    "OBJECT_TYPE": 2,
    "X": 3,
    "Y": 4,
    "CITY_NAME": 5,
}
LANE_WIDTH = {'MIA': 3.84, 'PIT': 3.97}
VELOCITY_THRESHOLD = 1.0

# Number of timesteps the track should exist to be considered in social context
# EXIST_THRESHOLD = (50)
EXIST_THRESHOLD = (5)

# index of the sorted velocity to look at, to call it as stationary
STATIONARY_THRESHOLD = (13)
color_dict = {"AGENT": "#d33e4c", "OTHERS": "#d3e8ef", "AV": "#007672"}
# LANE_RADIUS = 30
# OBJ_RADIUS = 30
LANE_RADIUS = 150
OBJ_RADIUS = 30000

DATA_DIR = './csv'
OBS_LEN = 20
PRED_LEN = 30
NUM_PREDICT = 30
COLLECT_DATA_RADIUS = 35    # once the current pos of the vehicle is within this radius, the track of this vehicle will be written in csv
HIDDEN_SIZE = 20
# INTERMEDIATE_DATA_DIR = './interm_data'
# INTERMEDIATE_DATA_DIR = './intermediate_data'
INTERMEDIATE_DATA_DIR = './csv'

# NORMALIZED_CENTER = np.array([7065.50, 4536.30])    # coordinate of the traffic light 
NORMALIZED_CENTER = np.array([0.0, 0.0])
STOP_THRESHOLD = 0.1 * PRED_LEN     # 1 m/s
FRONT_ANGLE_THRESHOLD = 20

SCENARIO_IDS = [0, 25, 50, 75, 100, 125, 150, 175]
PLT_SHOW = False
PLT_WRITE = True

FEATURE_COLUMN = 10

CONTROL_AREA = ['511308706#1','511308635','511308639#1','511308724#0','511308714#1','28672098#0','23031879#1','307704875#0']

FOCUS_AREA = ['511308639#1', '511308639#0', '511308635', '28672096#0', '511308706#1', '511308706#0', '307704875#0', \
              '307704875#1', '23031879#1', '23031879#0', '28672098#0', '28672098#1', '511308714#1', '511308714#0', \
              '511308724#0', '511308724#1']

STEPWISE_LOSS = False

UNICYCLE_MODEL = False

SAMPLE_RATE = 10

DT = float(1 / SAMPLE_RATE)    # delta_time 