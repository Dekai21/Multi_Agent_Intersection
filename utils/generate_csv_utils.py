import os
import pdb
import random
import sys
import xml.dom.minidom
from typing import List

import pandas as pd
from tqdm import trange

from utils.config import *

if sys.platform == 'win32':
    sys.path.append(os.path.join('C:/Users/ZHD1ABT/Downloads/sumo-1.13.0', 'tools'))
    sys.path.append(os.path.join('C:/Users/ZHD1ABT/Downloads/sumo-1.13.0', 'bin'))


def delete_first_two_lines(filename: str):
    """
    Delete the first two lines of the omnet files, otherwise there would be syntax error. 
    """

    with open(filename, "r+") as f:
        d = f.readlines()
        f.seek(0)
        for i, line in enumerate(d):
            if i != 1 and i !=2:
                f.write(line)
        f.truncate()


def save_csv(df: pd.DataFrame, name: str, dir: str = None):
    if dir is None:
        dir = './csv'
    if not os.path.exists(dir):
        os.makedirs(dir)

    name = f"{name}.csv"
    df.to_csv(os.path.join(dir, name))


def generate_fcd(sumocfg_path: str, 
                fcd_path: str, 
                edge_filter_path: str,
                begin_time: int, 
                offset_time: int, 
                total_time: int, 
                step_length: float = 0.1, 
                traffic_scale: float = 0.5)-> None:

    # old version: use filter_edges file to collect data from the central part of the intersection
    # cmd = f"sumo -c {sumocfg_path} --fcd-output {fcd_path} --begin {begin_time} --end {begin_time+offset_time+total_time} \
    #         --step-length {step_length} --scale {traffic_scale} --fcd-output.filter-edges.input-file {edge_filter_path}"

    # new version: collect the data from the whole scene, and filter out the vehicle whose curr_pos is outside the central part in generate_csv_from_fcd
    if sys.platform == 'win32':
        cmd = f"C:/Users/ZHD1ABT/Downloads/sumo-1.13.0/bin/sumo -c {sumocfg_path} --fcd-output {fcd_path} \
            --begin {begin_time} --end {begin_time+offset_time+total_time} \
            --step-length {step_length} --scale {traffic_scale}"
    else:
        cmd = f"sumo -c {sumocfg_path} --fcd-output {fcd_path} --begin {begin_time} --end {begin_time+offset_time+total_time} \
            --step-length {step_length} --scale {traffic_scale}"
    
    os.system(cmd)
    return None


def fcd_to_omnet(fcd_path: str, 
                begin_time: int, 
                offset_time: int, 
                total_time: int, 
                length_per_scene: int, 
                boundary: float)-> List[str]:
    """
    Generate omnet files (.xml) under 'omnet' folder.

    Returns:
        - omnet_files: e.g. ['omnet/sumoTrace_brunswick_300_306.xml', ...]
    """

    omnet_files = []
    for i_time in range(begin_time + offset_time, begin_time + offset_time + total_time):
        j_time = i_time + length_per_scene
        # name = f"sumoTrace_brunswick_{i_time}_{j_time}.xml"
        name = f"sumoTrace_simple_{i_time}_{j_time}.xml"
        output_file = os.path.join('omnet', name)
        omnet_files.append(output_file)
        cmd = f"python /usr/share/sumo/tools/traceExporter.py --fcd-input {fcd_path} \
                --omnet-output {output_file} --boundary {boundary} --begin {i_time} --end {j_time}"
        os.system(cmd)
        delete_first_two_lines(output_file)
    return omnet_files


def omnet_to_csv(omnet_file: str, target_folder: str):
    """
    Generate csv files under csv/train or csv/val folder.

    Args:
        - omnet_file: e.g. './omnet/sumoTrace_brunswick_300_306.xml'
        - target_folder: 'train' or 'val'
    """
    
    omnet_filename, extension = os.path.splitext(os.path.basename(omnet_file))
    oment_ids = omnet_filename.split('_')
    csv_name = f"{oment_ids[-2]}-{oment_ids[-1]}"   # begin_time - end_time
    DOMTree = xml.dom.minidom.parse(omnet_file)
    collection = DOMTree.documentElement
    if collection.hasAttribute("shelf"):
        print("Root element : %s" % collection.getAttribute("shelf"))

    tracks = collection.getElementsByTagName("waypoint")
    df = pd.DataFrame()

    for id, track in enumerate(tracks):
        track_id = track.getElementsByTagName('nodeid')[0].firstChild.nodeValue
        timestamp = float(track.getElementsByTagName('time')[0].firstChild.nodeValue)
        x = float(track.getElementsByTagName('xpos')[0].firstChild.nodeValue)
        y = float(track.getElementsByTagName('ypos')[0].firstChild.nodeValue)
        df = df.append({'TIMESTAMP': timestamp, 'TRACK_ID': track_id, 'OBJECT_TYPE': 'non-tgt', 'X': x, 'Y': y, 'CITY_NAME': 'SUMO'}, ignore_index=True)

    if df.empty:
        return None

    flag = 0
    for track_id, remain_df in df.groupby('TRACK_ID'):
        if(len(remain_df) >= (OBS_LEN + PRED_LEN)):
            # remain_df["OBJECT_TYPE"].replace({'non-tgt': 'tgt'}, inplace=True)    # FIXME: the modification after groupby doesn't work
            flag = 1

    if flag == 1:
        save_csv(df, csv_name, os.path.join('csv', target_folder))
    return None


def get_routes(from_path: str, straight_prob: float = 0.5)-> str:
    """
    Args:
        - from_path: e.g. 'left', 'right', 'up', 'down'
        - straight_prob: the probability of a vehicle to go straight

    Return:
        - route name: e.g. 'left_right', 'up_right'...
    """

    left_turn_prob = right_turn_prob = (1 - straight_prob) / 2.0

    random_value = random.uniform(0, 1)
    if random_value < straight_prob:    # straight
        if from_path == 'left':
            return 'left_right'
        elif from_path == 'right':
            return 'right_left'
        elif from_path == 'up':
            return 'up_down'
        elif from_path == 'down':
            return 'down_up'
    elif random_value < straight_prob + left_turn_prob:     # left-turn
        if from_path == 'left':
            return 'left_up'
        elif from_path == 'right':
            return 'right_down'
        elif from_path == 'up':
            return 'up_right'
        elif from_path == 'down':
            return 'down_left'
    else:       # right-turn
        if from_path == 'left':
            return 'left_down'
        elif from_path == 'right':
            return 'right_up'
        elif from_path == 'up':
            return 'up_left'
        elif from_path == 'down':
            return 'down_right'


def generate_routefile(rou_xml_filename: str = '04-16-22-01-00800-0.08-val-4', 
                        num_seconds: int = 2000,
                        create_new_vehicle_prob: float = 0.08, 
                        straight_prob: float = 0.4,
                        random_seed: int = 3):
    """
    Generate *.rou.xml file. (for the separated road net)

    Args:
        - rou_xml_filename
        - num_seconds
        - create_new_vehicle_prob: the prob of generating a new vehicle at a start point per second, e.g. 0.08 (normal), 0.12 (slightly busy)
        - straight_prob
    """

    random.seed(random_seed)  # make tests reproducible
    num_vehicles = 0
    os.makedirs('sumo/route', exist_ok=True)
    
    with open(f"sumo/route/{rou_xml_filename}.rou.xml", "w") as routes:
        print("""<routes>
        <vType id="typeWE" accel="2.5" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="40" guiShape="passenger"/>

        <route id="left_up" edges="E4 -E2 -E1 -E5"/>
        <route id="left_right" edges="E4 -E2 -E3 -E6"/>
        <route id="left_down" edges="E4 -E2 E0 -E7"/>

        <route id="right_up" edges="E6 E3 -E1 -E5"/>
        <route id="right_left" edges="E6 E3 E2 -E4"/>
        <route id="right_down" edges="E6 E3 E0 -E7"/>

        <route id="down_up" edges="E7 -E0 -E1 -E5"/>
        <route id="down_left" edges="E7 -E0 E2 -E4"/>
        <route id="down_right" edges="E7 -E0 -E3 -E6"/>

        <route id="up_down" edges="E5 E1 E0 -E7"/>
        <route id="up_left" edges="E5 E1 E2 -E4"/>
        <route id="up_right" edges="E5 E1 -E3 -E6"/>""", file=routes)

        for i_second in range(num_seconds):

            if i_second % 50 == 0:
                # create_new_vehicle_prob = np.random.choice([0.05,0.055,0.06,0.065,0.07,0.075,0.08,0.085,0.09,0.095,0.1])
                create_new_vehicle_prob = np.random.randint(6000,8500)/100000
                print('create_new_vehicle_prob: ', create_new_vehicle_prob)

            # from left
            random_value = random.uniform(0, 1)
            if random_value < create_new_vehicle_prob * 0.7:
                route = get_routes('left', straight_prob=straight_prob)
                print('    <vehicle id="%s_%i" type="typeWE" route="%s" depart="%i" />' % (route, num_vehicles, route, i_second), file=routes)
                num_vehicles += 1

            # from right
            random_value = random.uniform(0, 1)
            if random_value < create_new_vehicle_prob * 0.7:
                route = get_routes('right', straight_prob=straight_prob)
                print('    <vehicle id="%s_%i" type="typeWE" route="%s" depart="%i" />' % (route, num_vehicles, route, i_second), file=routes)
                num_vehicles += 1

            # from up
            random_value = random.uniform(0, 1)
            if random_value < create_new_vehicle_prob:
                route = get_routes('up', straight_prob=straight_prob)
                print('    <vehicle id="%s_%i" type="typeWE" route="%s" depart="%i" />' % (route, num_vehicles, route, i_second), file=routes)
                num_vehicles += 1

            # from down
            random_value = random.uniform(0, 1)
            if random_value < create_new_vehicle_prob:
                route = get_routes('down', straight_prob=straight_prob)
                print('    <vehicle id="%s_%i" type="typeWE" route="%s" depart="%i" />' % (route, num_vehicles, route, i_second), file=routes)
                num_vehicles += 1

        print("</routes>", file=routes)


def generate_sumocfg(rou_xml_filename: str = '04-16-22-01-00800-0.08-val-4')-> str:

    sumocfg_filename = f"sumo/sumocfg/{rou_xml_filename}.sumocfg"
    with open(sumocfg_filename, "w") as sumocfg:
        print(f"""<?xml version="1.0" encoding="UTF-8"?>

<!-- generated on 08/09/21 14:01:24 by Eclipse SUMO sumo Version v1_8_0+1925-6bf04e0fef
-->

<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">

    <input>
        <net-file value="../map/simple_separate_10m.net.xml"/>
        <route-files value="../route/{rou_xml_filename}.rou.xml"/>
    </input>

    <processing>
        <ignore-route-errors value="true"/>
    </processing>

    <routing>
        <device.rerouting.adaptation-steps value="18"/>
        <device.rerouting.adaptation-interval value="10"/>
    </routing>

    <report>
        <verbose value="true"/>
        <duration-log.statistics value="true"/>
        <no-step-log value="true"/>
    </report>

    <gui_only>
        <gui-settings-file value="simple.view.xml"/>
    </gui_only>

</configuration>""", file=sumocfg)

    return sumocfg_filename


def generate_csv_from_fcd(fcd_file: str, time_per_scene: int, split: str = 'train'):

    # pdb.set_trace()
    DOMTree = xml.dom.minidom.parse(fcd_file)
    collection = DOMTree.documentElement
    tracks = collection.getElementsByTagName('timestep')
    df = pd.DataFrame()

    for t in trange(len(tracks)):    # each time (0.1s)
        track = tracks[t]
        timestamp = float(track.getAttribute('time'))
        vehicles = track.getElementsByTagName('vehicle')
        for vehicle in vehicles:
            track_id = vehicle.getAttribute('id')
            x = float(vehicle.getAttribute('x'))
            y = float(vehicle.getAttribute('y'))
            yaw_angle = float(vehicle.getAttribute('angle'))
            speed = float(vehicle.getAttribute('speed'))
            df = df.append({'TIMESTAMP': timestamp, 'TRACK_ID': track_id, 'OBJECT_TYPE': 'tgt', 'X': x, 'Y': y,'yaw':yaw_angle,'speed':speed , 'CITY_NAME': 'SUMO'}, ignore_index=True)
        if len(df) == 0:
            continue

        ### Begin: new version (now only the tgt will be written into csv files)
        tgt_agent_ids = []
        # curr_time = df['TIMESTAMP'].max() - (PRED_LEN / SAMPLE_RATE)
        curr_time = df['TIMESTAMP'].max() - (35 / SAMPLE_RATE)
        max_time = int(df['TIMESTAMP'].max())
        min_time = max_time - time_per_scene

        for track_id, remain_df in df.groupby('TRACK_ID'):
            if len(remain_df.loc[np.isclose(remain_df['TIMESTAMP'], curr_time)]) == 0:
                continue
            if len(remain_df) < PRED_LEN + OBS_LEN:
                continue
            x, y = remain_df.loc[np.isclose(remain_df['TIMESTAMP'], curr_time)][['X', 'Y']].values.reshape(-1)
            if (-COLLECT_DATA_RADIUS < x < COLLECT_DATA_RADIUS) and (-COLLECT_DATA_RADIUS < y < COLLECT_DATA_RADIUS):
                tgt_agent_ids.append(track_id)

        if len(tgt_agent_ids) > 0:
            df = df.drop(df[df.TIMESTAMP < (df['TIMESTAMP'].max() - time_per_scene)].index) # make sure each scene is exactly time_per_scene length
            csv_df = df.loc[[id in tgt_agent_ids for id in df['TRACK_ID'].values.tolist()]]
            csv_name = f"{(min_time):0>5}-{(max_time):0>5}"
            save_csv(csv_df, csv_name, os.path.join('csv', split))
            df = df.drop(df[df.TIMESTAMP <= (df['TIMESTAMP'].max() - time_per_scene + 3.5)].index)     # sliding window of 3.5 seconds (avoid overlap between 2 csv)
            del csv_df
            tgt_agent_ids = []
        ### End: new version

    return None

# def generate_csv_from_fcd(fcd_file: str, time_per_scene: int, split: str = 'train'):

#     # pdb.set_trace()
#     DOMTree = xml.dom.minidom.parse(fcd_file)
#     collection = DOMTree.documentElement
#     tracks = collection.getElementsByTagName('timestep')
#     df = pd.DataFrame()

#     for t in trange(len(tracks)):    # each time (0.1s)
#         track = tracks[t]
#         timestamp = float(track.getAttribute('time'))
#         vehicles = track.getElementsByTagName('vehicle')
#         for vehicle in vehicles:
#             track_id = vehicle.getAttribute('id')
#             x = float(vehicle.getAttribute('x'))
#             y = float(vehicle.getAttribute('y'))
#             yaw_angle = float(vehicle.getAttribute('angle'))
#             speed = float(vehicle.getAttribute('speed'))
#             df = df.append({'TIMESTAMP': timestamp, 'TRACK_ID': track_id, 'OBJECT_TYPE': 'tgt', 'X': x, 'Y': y,'yaw':yaw_angle,'speed':speed , 'CITY_NAME': 'SUMO'}, ignore_index=True)
#         if len(df) == 0:
#             continue

#         ### Begin: new version (now only the tgt will be written into csv files)
#         tgt_agent_ids = []
#         curr_time = df['TIMESTAMP'].max() - (PRED_LEN / SAMPLE_RATE)
#         max_time = int(df['TIMESTAMP'].max())
#         min_time = max_time - time_per_scene

#         for track_id, remain_df in df.groupby('TRACK_ID'):
#             if len(remain_df.loc[np.isclose(remain_df['TIMESTAMP'], curr_time)]) == 0:
#                 continue
#             if len(remain_df) < PRED_LEN + OBS_LEN:
#                 continue
#             x, y = remain_df.loc[np.isclose(remain_df['TIMESTAMP'], curr_time)][['X', 'Y']].values.reshape(-1)
#             if x > -COLLECT_DATA_RADIUS and x < COLLECT_DATA_RADIUS and y > -COLLECT_DATA_RADIUS and y < COLLECT_DATA_RADIUS:
#                 tgt_agent_ids.append(track_id)

#         if len(tgt_agent_ids) > 0:
#             df = df.drop(df[df.TIMESTAMP < (df['TIMESTAMP'].max() - time_per_scene)].index) # make sure each scene is exactly time_per_scene length
#             csv_df = df.loc[[id in tgt_agent_ids for id in df['TRACK_ID'].values.tolist()]]
#             csv_name = f"{(min_time):0>5}-{(max_time):0>5}"
#             save_csv(csv_df, csv_name, os.path.join('csv', split))
#             df = df.drop(df[df.TIMESTAMP <= (df['TIMESTAMP'].max() - time_per_scene + 1)].index)     # sliding window of 1 second
#             del csv_df
#             tgt_agent_ids = []
#         ### End: new version

#     return None
