#!/usr/bin/env python

# Copyright (c) 2020 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
"""
Script to integrate CARLA and SUMO simulations
"""

# ==================================================================================================
# -- imports ---------------------------------------------------------------------------------------
# ==================================================================================================

import argparse
import logging
import time

# ==================================================================================================
# -- find carla module -----------------------------------------------------------------------------
# ==================================================================================================

import glob
import os
import sys

try:
    sys.path.append(
        glob.glob('../../PythonAPI/carla/dist/carla-*%d.%d-%s.egg' %
                  (sys.version_info.major, sys.version_info.minor,
                   'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==================================================================================================
# -- find traci module -----------------------------------------------------------------------------
# ==================================================================================================

# if 'SUMO_HOME' in os.environ:
#     sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
# else:
#     sys.exit("please declare environment variable 'SUMO_HOME'")

sys.path.append(os.path.join('C:/Users/ZHD1ABT/Downloads/sumo-1.13.0', 'tools'))    # "traci" folder is here, you need to change this path to your sumo path
from util_0907 import get_intention_from_vehicle_id, get_intention_vector, GNN, rotation_matrix_back, GNN_mtl, GNN_mtl_gnn
import numpy as np
import torch

# ==================================================================================================
# -- sumo integration imports ----------------------------------------------------------------------
# ==================================================================================================

from sumo_integration.bridge_helper import BridgeHelper  # pylint: disable=wrong-import-position
from sumo_integration.carla_simulation import CarlaSimulation  # pylint: disable=wrong-import-position
from sumo_integration.constants import INVALID_ACTOR_ID  # pylint: disable=wrong-import-position
from sumo_integration.sumo_simulation import SumoSimulation  # pylint: disable=wrong-import-position

# ==================================================================================================
# -- synchronization_loop --------------------------------------------------------------------------
# ==================================================================================================


class SimulationSynchronization(object):
    """
    SimulationSynchronization class is responsible for the synchronization of sumo and carla
    simulations.
    """
    def __init__(self,
                 sumo_simulation,
                 carla_simulation,
                 tls_manager='none',
                 sync_vehicle_color=False,
                 sync_vehicle_lights=False):

        self.sumo = sumo_simulation
        self.carla = carla_simulation

        self.tls_manager = tls_manager
        self.sync_vehicle_color = sync_vehicle_color
        self.sync_vehicle_lights = sync_vehicle_lights

        if tls_manager == 'carla':
            self.sumo.switch_off_traffic_lights()
        elif tls_manager == 'sumo':
            self.carla.switch_off_traffic_lights()

        # Mapped actor ids.
        self.sumo2carla_ids = {}  # Contains only actors controlled by sumo.
        self.carla2sumo_ids = {}  # Contains only actors controlled by carla.

        BridgeHelper.blueprint_library = self.carla.world.get_blueprint_library()
        BridgeHelper.offset = self.sumo.get_net_offset()

        # Configuring carla simulation in sync mode.
        settings = self.carla.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.carla.step_length
        self.carla.world.apply_settings(settings)

        self.steer_record = {}
        self.trajs = {}
        self.control_record = []
        self.to_be_destroyed_ids = []
        self.rest_of_life = {}
        self.odometers = {}
        self.odometers_cum = []
        self.i_v = 0
        self.color_box = [(0,0,0),(255,255,255),(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255),(255,0,255),(192,192,192),(128,128,128),(128,0,0),(128,128,0),(0,128,0),(128,0,128),(0,128,128),(0,0,128)]
        self.num_color = len(self.color_box)


    def tick(self):
        """
        Tick to simulation synchronization
        """
        # -----------------
        # sumo-->carla sync
        # -----------------
        self.sumo.tick()

        # Spawning new sumo actors in carla (i.e, not controlled by carla).
        sumo_spawned_actors = self.sumo.spawned_actors - set(self.carla2sumo_ids.values())
        for sumo_actor_id in sumo_spawned_actors:
            self.sumo.subscribe(sumo_actor_id)
            sumo_actor = self.sumo.get_actor(sumo_actor_id)

            carla_blueprint = BridgeHelper.get_carla_blueprint(sumo_actor, self.sync_vehicle_color)
            if carla_blueprint is not None:
                carla_transform = BridgeHelper.get_carla_transform(sumo_actor.transform,
                                                                   sumo_actor.extent)   # extent: from vehicle's corner to vehicle's center

                carla_blueprint.set_attribute('color', str(self.color_box[self.i_v % self.num_color])[1:-1])
                self.i_v += 1
                carla_actor_id = self.carla.spawn_actor(carla_blueprint, carla_transform)
                if carla_actor_id != INVALID_ACTOR_ID:
                    self.sumo2carla_ids[sumo_actor_id] = carla_actor_id
            else:
                self.sumo.unsubscribe(sumo_actor_id)

        # Destroying sumo arrived actors in carla.
        # for sumo_actor_id in self.sumo.destroyed_actors:
        #     if sumo_actor_id in self.sumo2carla_ids:
        #         self.carla.destroy_actor(self.sumo2carla_ids.pop(sumo_actor_id))
        for sumo_actor_id in self.sumo.destroyed_actors:
            if sumo_actor_id in self.sumo2carla_ids:
                self.to_be_destroyed_ids.append(self.sumo2carla_ids.pop(sumo_actor_id))

        # # 在这里构建sumo_control类似的图, 输入模型, 获取控制信号, 当车进入控制圈之后, 接管其控制任务
        # v_features = []
        # tgt_sumo_ids = []
        # graph_sumo_ids = []
        # for v_id in list(self.sumo2carla_ids.keys()):

        #     sumo_actor = self.sumo.get_actor(v_id)
        #     # carla_actor = self.carla.get_actor(self.sumo2carla_ids[v_id])
        #     # x = sumo_actor.transform.location.x
        #     # y = sumo_actor.transform.location.y
        #     x = self.carla.get_actor(self.sumo2carla_ids[v_id]).get_location().x
        #     y = self.carla.get_actor(self.sumo2carla_ids[v_id]).get_location().y
        #     z = self.carla.get_actor(self.sumo2carla_ids[v_id]).get_location().z
        #     if abs(z) > 1:
        #         continue
        #     carla_yaw = self.carla.get_actor(self.sumo2carla_ids[v_id]).get_transform().rotation.yaw
        #     # x += np.sin(np.deg2rad(90 - carla_yaw)) * 2.5
        #     # y += np.cos(np.deg2rad(90 - carla_yaw)) * 2.5
        #     x += np.cos(np.deg2rad(carla_yaw)) * 2.0
        #     y += np.sin(np.deg2rad(carla_yaw)) * 2.0
        #     # if abs(x) <= 70 or abs(y) <= 70:
        #     if np.linalg.norm(np.array([x, y])) > 70:
        #         if v_id not in self.trajs.keys():
        #             self.trajs[v_id] = []
        #         continue
        #     if v_id not in self.trajs.keys():
        #         continue
        #     if abs(x) > 70 or abs(y) > 70:
        #         continue
        #     graph_sumo_ids.append(v_id)
        #     if np.linalg.norm(np.array([x, y])) < 20  or (v_id in self.control_record and np.linalg.norm(np.array([x, y])) < 60):
        #         tgt_sumo_ids.append(v_id)
        #         self.control_record.append(v_id)
        #     intention = get_intention_vector(get_intention_from_vehicle_id(v_id))
        #     # v_feature = np.concatenate((np.array([x, -y]), intention)).reshape(1,5)
        #     v_feature = np.concatenate((np.array([x, y]), intention)).reshape(1,5)
        #     v_features.append(v_feature)
        # if len(v_features) > 0:
        #     v_features = np.concatenate(v_features, axis=0) # [v, 5]
        #     n_v = v_features.shape[0]
        #     edge_indexs = torch.tensor([[x,y] for x in range(n_v) for y in range(n_v)]).T.to(self.device)
        #     v_features = torch.tensor(v_features).float().to(self.device)
        #     with torch.no_grad():
        #         ## 1)
        #         out = self.model(v_features, edge_indexs).cpu().numpy().reshape(n_v, 30, 2).transpose(0,2,1)   # [v, 2, pred]
                
        #         ## 2)
        #         # _out = self.model(v_features, edge_indexs).cpu().numpy().reshape(n_v, 30, 4)
        #         # out = _out[:,:,:2].transpose(0,2,1)   # [v, 2, pred]
        #         # out2 = _out[:,:,2:]   # [v, pred, 2]

        #         # out2 = self.model(v_features, edge_indexs).cpu().numpy().reshape(n_v, 30, 2)

        #         ## 3)
        #         # out = self.model(v_features, edge_indexs).cpu().numpy().reshape(n_v, 30, 2)   # [v, pred, 2]

        # Updating sumo actors in carla.
        for sumo_actor_id in self.sumo2carla_ids:
            carla_actor_id = self.sumo2carla_ids[sumo_actor_id]

            sumo_actor = self.sumo.get_actor(sumo_actor_id)
            carla_actor = self.carla.get_actor(carla_actor_id)

            carla_transform = BridgeHelper.get_carla_transform(sumo_actor.transform,
                                                               sumo_actor.extent)
            if self.sync_vehicle_lights:
                carla_lights = BridgeHelper.get_carla_lights_state(carla_actor.get_light_state(),
                                                                   sumo_actor.signals)
            else:
                carla_lights = None
            x, y = carla_actor.get_transform().location.x, carla_actor.get_transform().location.y
            if abs(x) < 60 and abs(y) < 60:
                self.carla.synchronize_vehicle1(carla_actor_id, carla_transform, sumo_actor.velocity, self.steer_record, carla_lights)
                if abs(x) < 40 and abs(y) < 40:
                    if carla_actor_id not in self.odometers:
                        self.odometers[carla_actor_id] = [] # skip the first step [0,0]
                    else:
                        self.odometers[carla_actor_id].append([x, y])

            else:
                self.carla.synchronize_vehicle(carla_actor_id, carla_transform, sumo_actor.velocity, self.steer_record, carla_lights)

            # if sumo_actor_id in tgt_sumo_ids:
            #     index = graph_sumo_ids.index(sumo_actor_id) # index in graph

            #     #################################
            #     # 1)
            #     sumo_actor = self.sumo.get_actor(sumo_actor_id)
            #     yaw_carla = np.deg2rad(sumo_actor.transform.rotation.yaw - 90)
            #     yaw_carla = np.deg2rad(carla_actor.get_transform().rotation.yaw)
            #     rotation_back = rotation_matrix_back(yaw_carla) # [2,2]
            #     global_delta = (rotation_back @ out[index]).transpose(1,0)  # [pred_len, 2]
            #     x, y = v_features[index][0].item(), v_features[index][1].item()
            #     x = x - np.cos(yaw_carla) * 2.0
            #     y = y - np.sin(yaw_carla) * 2.0
            #     # x = sumo_actor.transform.location.x
            #     # y = sumo_actor.transform.location.y
            #     pos = [x+global_delta[0,0], y+global_delta[0,1]]
            #     # sumo_actor.transform.location.x = pos[0]
            #     # sumo_actor.transform.location.y = pos[1]
            #     carla_transform.location.x = pos[0]
            #     carla_transform.location.y = pos[1]
            #     # carla_transform.rotation.yaw = np.rad2deg(yaw_carla)
            #     self.carla.synchronize_vehicle(carla_actor_id, carla_transform, sumo_actor.velocity, self.steer_record, carla_lights)

            #     # 2)
            #     # acc, delta = out2[index][1]
            #     # acc, delta = out2[index][2:5,:].mean(axis=0)
            #     # self.carla.synchronize_vehicle2(carla_actor_id, acc, delta)

            #     # 3)
            #     # self.carla.synchronize_vehicle3(carla_actor_id, out[index])
            #     # self.carla.get_actor(carla_actor_id).get_velocity()
            #     #################################

            # elif sumo_actor_id in self.control_record:
            #     # self.carla.synchronize_vehicle2(carla_actor_id, 10, 0)
            #     self.carla.destroy_actor(carla_actor_id)

            # else:
            #     x, y = self.carla.get_actor(carla_actor_id).get_location().x, self.carla.get_actor(carla_actor_id).get_location().y
            #     if 80 > abs(x) > 20 or 80 > abs(y) > 20:
            #         self.carla.synchronize_vehicle(carla_actor_id, carla_transform, sumo_actor.velocity, self.steer_record, carla_lights)    
            #     else:
            #         self.carla.synchronize_vehicle(carla_actor_id, carla_transform, sumo_actor.velocity, self.steer_record, carla_lights)

        if self.sumo.sumo_time > 241:
        # if self.sumo.sumo_time > 101:
            for v in self.odometers:
                odometer = self.odometers[v]
                if len(odometer) == 0:
                    continue
                odometer = np.array(odometer)   # [step, 2]
                stepwise_move = odometer[:-1,:] - odometer[1:, :]
                stepwise_move = np.linalg.norm(stepwise_move, axis=-1)
                odometer = stepwise_move.sum()
                self.odometers_cum.append(odometer)
            self.odometers_cum = np.array(self.odometers_cum)
            self.odometers_cum = self.odometers_cum[self.odometers_cum > 0]
            self.odometers_cum = self.odometers_cum.sum()
            print("++++++++++++++++++++++++++++++++++++")
            print("Odometers_cum: ", self.odometers_cum)
            print("++++++++++++++++++++++++++++++++++++")
            exit

        for v in self.to_be_destroyed_ids:
            carla_actor_id = v
            if carla_actor_id not in self.rest_of_life:
                self.rest_of_life[carla_actor_id] = 100
                # self.rest_of_life[carla_actor_id] = 1
            carla_actor = self.carla.get_actor(carla_actor_id)
            x, y = carla_actor.get_transform().location.x, carla_actor.get_transform().location.y
            if abs(x) < 60 and abs(y) < 60 and self.rest_of_life[carla_actor_id] > 0:
                self.carla.synchronize_vehicle2(carla_actor_id)
                self.rest_of_life[carla_actor_id] -= 1
                # carla_actor.set_autopilot()
            else:
                self.carla.destroy_actor(v)
        
        # Updates traffic lights in carla based on sumo information.
        if self.tls_manager == 'sumo':
            common_landmarks = self.sumo.traffic_light_ids & self.carla.traffic_light_ids
            for landmark_id in common_landmarks:
                sumo_tl_state = self.sumo.get_traffic_light_state(landmark_id)
                carla_tl_state = BridgeHelper.get_carla_traffic_light_state(sumo_tl_state)

                self.carla.synchronize_traffic_light(landmark_id, carla_tl_state)

        # -----------------
        # carla-->sumo sync
        # -----------------
        self.carla.tick()

        # Spawning new carla actors (not controlled by sumo)
        carla_spawned_actors = self.carla.spawned_actors - set(self.sumo2carla_ids.values())
        for carla_actor_id in carla_spawned_actors:
            self.carla.destroy_actor(carla_actor_id)    
            # Dekai: if the last execution is killed when the sumo is still connected, there could be some  vehicles remaining in Carla.
            # These vehicles will be viewed as being spawned by carla and then updated to sumo, which is not we want, so we need to remove them.
            carla_actor = self.carla.get_actor(carla_actor_id)

            type_id = BridgeHelper.get_sumo_vtype(carla_actor)
            color = carla_actor.attributes.get('color', None) if self.sync_vehicle_color else None
            if type_id is not None:
                sumo_actor_id = self.sumo.spawn_actor(type_id, color)
                if sumo_actor_id != INVALID_ACTOR_ID:
                    self.carla2sumo_ids[carla_actor_id] = sumo_actor_id
                    self.sumo.subscribe(sumo_actor_id)

        # Destroying required carla actors in sumo.
        for carla_actor_id in self.carla.destroyed_actors:
            if carla_actor_id in self.carla2sumo_ids:
                self.sumo.destroy_actor(self.carla2sumo_ids.pop(carla_actor_id))

        # Updating carla actors in sumo.
        for carla_actor_id in self.carla2sumo_ids:
            sumo_actor_id = self.carla2sumo_ids[carla_actor_id]

            carla_actor = self.carla.get_actor(carla_actor_id)
            sumo_actor = self.sumo.get_actor(sumo_actor_id)

            sumo_transform = BridgeHelper.get_sumo_transform(carla_actor.get_transform(),
                                                             carla_actor.bounding_box.extent)
            if self.sync_vehicle_lights:
                carla_lights = self.carla.get_actor_light_state(carla_actor_id)
                if carla_lights is not None:
                    sumo_lights = BridgeHelper.get_sumo_lights_state(sumo_actor.signals,
                                                                     carla_lights)
                else:
                    sumo_lights = None
            else:
                sumo_lights = None

            self.sumo.synchronize_vehicle(sumo_actor_id, sumo_transform, sumo_lights)

        # Updates traffic lights in sumo based on carla information.
        if self.tls_manager == 'carla':
            common_landmarks = self.sumo.traffic_light_ids & self.carla.traffic_light_ids
            for landmark_id in common_landmarks:
                carla_tl_state = self.carla.get_traffic_light_state(landmark_id)
                sumo_tl_state = BridgeHelper.get_sumo_traffic_light_state(carla_tl_state)

                # Updates all the sumo links related to this landmark.
                self.sumo.synchronize_traffic_light(landmark_id, sumo_tl_state)

    def close(self):
        """
        Cleans synchronization.
        """
        # Configuring carla simulation in async mode.
        settings = self.carla.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self.carla.world.apply_settings(settings)

        # Destroying synchronized actors.
        for carla_actor_id in self.sumo2carla_ids.values():
            self.carla.destroy_actor(carla_actor_id)

        for sumo_actor_id in self.carla2sumo_ids.values():
            self.sumo.destroy_actor(sumo_actor_id)

        # Closing sumo and carla client.
        self.carla.close()
        self.sumo.close()


def synchronization_loop(args):
    """
    Entry point for sumo-carla co-simulation.
    """
    sumo_simulation = SumoSimulation(args.sumo_cfg_file, args.step_length, args.sumo_host,
                                     args.sumo_port, args.sumo_gui, args.client_order)
    carla_simulation = CarlaSimulation(args.carla_host, args.carla_port, args.step_length)

    synchronization = SimulationSynchronization(sumo_simulation, carla_simulation, args.tls_manager,
                                                args.sync_vehicle_color, args.sync_vehicle_lights)
    try:
        while True:
            start = time.time()

            synchronization.tick()

            end = time.time()
            elapsed = end - start
            if elapsed < args.step_length:
                time.sleep(args.step_length - elapsed)

    except KeyboardInterrupt:
        logging.info('Cancelled by user.')

    finally:
        logging.info('Cleaning synchronization')

        # Add
        while True:
            start = time.time()

            synchronization.tick()

            end = time.time()
            elapsed = end - start
            if elapsed < args.step_length:
                time.sleep(args.step_length - elapsed)

        synchronization.close()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('sumo_cfg_file', type=str, help='sumo configuration file')
    argparser.add_argument('--carla-host',
                           metavar='H',
                           default='127.0.0.1',
                           help='IP of the carla host server (default: 127.0.0.1)')
    argparser.add_argument('--carla-port',
                           metavar='P',
                           default=2000,
                           type=int,
                           help='TCP port to listen to (default: 2000)')
    argparser.add_argument('--sumo-host',
                           metavar='H',
                           default=None,
                           help='IP of the sumo host server (default: 127.0.0.1)')
    argparser.add_argument('--sumo-port',
                           metavar='P',
                           default=None,
                           type=int,
                           help='TCP port to listen to (default: 8813)')
    argparser.add_argument('--sumo-gui', action='store_true', help='run the gui version of sumo')
    argparser.add_argument('--step-length',
                           default=0.05,
                           type=float,
                           help='set fixed delta seconds (default: 0.05s)')
    argparser.add_argument('--client-order',
                           metavar='TRACI_CLIENT_ORDER',
                           default=1,
                           type=int,
                           help='client order number for the co-simulation TraCI connection (default: 1)')
    argparser.add_argument('--sync-vehicle-lights',
                           action='store_true',
                           help='synchronize vehicle lights state (default: False)')
    argparser.add_argument('--sync-vehicle-color',
                           action='store_true',
                           help='synchronize vehicle color (default: False)')
    argparser.add_argument('--sync-vehicle-all',
                           action='store_true',
                           help='synchronize all vehicle properties (default: False)')
    argparser.add_argument('--tls-manager',
                           type=str,
                           choices=['none', 'sumo', 'carla'],
                           help="select traffic light manager (default: none)",
                           default='none')
    argparser.add_argument('--debug', action='store_true', help='enable debug messages')
    arguments = argparser.parse_args()

    if arguments.sync_vehicle_all is True:
        arguments.sync_vehicle_lights = True
        arguments.sync_vehicle_color = True

    if arguments.debug:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
    else:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    synchronization_loop(arguments)
