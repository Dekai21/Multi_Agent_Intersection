#!/usr/bin/env python

# Copyright (c) 2020 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
""" This module is responsible for the management of the carla simulation. """

# ==================================================================================================
# -- imports ---------------------------------------------------------------------------------------
# ==================================================================================================

import logging

import carla  # pylint: disable=import-error
import numpy as np

from .constants import INVALID_ACTOR_ID, SPAWN_OFFSET_Z
from .MPC_XY_Frame import (K_BRAKE, K_STEER, K_THROTTLE, PATH, Node,
                           calc_ref_trajectory_in_T_step, get_destination_in_T_step, linear_mpc_control, MPC_module)
from util_0907 import get_intention_from_vehicle_id, get_intention_vector, GNN, rotation_matrix_back, GNN_mtl

# ==================================================================================================
# -- carla simulation ------------------------------------------------------------------------------
# ==================================================================================================


class CarlaSimulation(object):
    """
    CarlaSimulation is responsible for the management of the carla simulation.
    """
    def __init__(self, host, port, step_length):
        self.client = carla.Client(host, port)
        self.client.set_timeout(2.0)

        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.step_length = step_length

        # The following sets contain updated information for the current frame.
        self._active_actors = set()
        self.spawned_actors = set()
        self.destroyed_actors = set()

        # Set traffic lights.
        self._tls = {}  # {landmark_id: traffic_ligth_actor}

        # Dekai: Add Dicts for Path and Node
        self.path_dict = {}     # { [vehicle_id: int]: target traj }
        self.a_old_dict = {}    # { [vehicle_id: int]: accelaration in the last step }
        self.delta_old_dict = {}    # { [vehicle_id: int]: accelaration in the last step }

        tmp_map = self.world.get_map()
        for landmark in tmp_map.get_all_landmarks_of_type('1000001'):
            if landmark.id != '':
                traffic_ligth = self.world.get_traffic_light(landmark)
                if traffic_ligth is not None:
                    self._tls[landmark.id] = traffic_ligth
                else:
                    logging.warning('Landmark %s is not linked to any traffic light', landmark.id)

    def get_actor(self, actor_id):
        """
        Accessor for carla actor.
        """
        return self.world.get_actor(actor_id)

    # This is a workaround to fix synchronization issues when other carla clients remove an actor in
    # carla without waiting for tick (e.g., running sumo co-simulation and manual control at the
    # same time)
    def get_actor_light_state(self, actor_id):
        """
        Accessor for carla actor light state.

        If the actor is not alive, returns None.
        """
        try:
            actor = self.get_actor(actor_id)
            return actor.get_light_state()
        except RuntimeError:
            return None

    @property
    def traffic_light_ids(self):
        return set(self._tls.keys())

    def get_traffic_light_state(self, landmark_id):
        """
        Accessor for traffic light state.

        If the traffic ligth does not exist, returns None.
        """
        if landmark_id not in self._tls:
            return None
        return self._tls[landmark_id].state

    def switch_off_traffic_lights(self):
        """
        Switch off all traffic lights.
        """
        for actor in self.world.get_actors():
            if actor.type_id == 'traffic.traffic_light':
                actor.freeze(True)
                # We set the traffic light to 'green' because 'off' state sets the traffic light to
                # 'red'.
                actor.set_state(carla.TrafficLightState.Green)

    def spawn_actor(self, blueprint, transform):
        """
        Spawns a new actor.

            :param blueprint: blueprint of the actor to be spawned.
            :param transform: transform where the actor will be spawned.
            :return: actor id if the actor is successfully spawned. Otherwise, INVALID_ACTOR_ID.
        """
        transform = carla.Transform(transform.location + carla.Location(0, 0, SPAWN_OFFSET_Z),
                                    transform.rotation)

        batch = [
            carla.command.SpawnActor(blueprint, transform).then(
                # carla.command.SetSimulatePhysics(carla.command.FutureActor, False))
                carla.command.SetSimulatePhysics(carla.command.FutureActor, True))
                # https://github.com/carla-simulator/carla/issues/4043#issuecomment-817851851
                # Enable physics of vehicles to make them controllable through throttle and steering instead of simply by vehicle.set_transform(transform) 
        ]

        response = self.client.apply_batch_sync(batch, False)[0]
        if response.error:
            logging.error('Spawn carla actor failed. %s', response.error)
            return INVALID_ACTOR_ID

        return response.actor_id

    def destroy_actor(self, actor_id):
        """
        Destroys the given actor.
        """
        actor = self.world.get_actor(actor_id)
        if actor is not None:
            return actor.destroy()
        return False

    def synchronize_vehicle(self, vehicle_id, transform, velocity, steer_record: dict, lights=None):
        """
        Updates vehicle state. (teleport)

            :param vehicle_id: id of the actor to be updated.
            :param transform: new vehicle transform (i.e., position and rotation).
            :param lights: new vehicle light state.
            :return: True if successfully updated. Otherwise, False.
        """
        vehicle = self.world.get_actor(vehicle_id)
        
        if vehicle is None:
            return False

        # Original synchronization method: update the vehicles to the latest transform
        vehicle.set_transform(transform)
        # vehicle.apply_control(carla.VehicleControl(throttle=0.6, steer=0))
        
        yaw_carla = np.deg2rad(vehicle.get_transform().rotation.yaw)
        vehicle.set_target_velocity(carla.Vector3D(np.cos(yaw_carla)*velocity, np.sin(yaw_carla)*velocity, 0.0))
        
        
        # local_yaw = np.rad2deg(np.arctan2(vel[:,0], vel[:,1]))
        # delta_yaw = np.deg2rad(local_yaw[1:] - local_yaw[:-1]) * 10   # [pred-2]
        # yaw_carla = vehicle.get_transform().rotation.yaw    # degree
        # rotation_back = rotation_matrix_back(np.deg2rad(yaw_carla - local_yaw[0])) # [2,2]
        # vel_global = rotation_back @ vel[0:n_step,:].transpose(1,0) # [2, step]
        # vel_global = vel_global.mean(axis=1)    # [2]
        # vehicle.set_target_velocity(carla.Vector3D(vel_global[0], vel_global[1], 0.0))
        # vehicle.set_target_angular_velocity(carla.Vector3D(0, 0, -delta_yaw[0].item()))

        if lights is not None:
            vehicle.set_light_state(carla.VehicleLightState(lights))
        return True

    def synchronize_vehicle1(self, vehicle_id, transform, velocity, steer_record: dict, lights=None):
        """
        Updates vehicle state. (use MPC)

            :param vehicle_id: id of the actor to be updated.
            :param transform: new vehicle transform (i.e., position and rotation).
            :param lights: new vehicle light state.
            :param steer_record: just for debug.
            :return: True if successfully updated. Otherwise, False.
        """
        vehicle = self.world.get_actor(vehicle_id)


        
        if vehicle is None:
            return False

        # Original synchronization method: update the vehicles to the latest transform
        # vehicle.set_transform(transform)
        # vehicle.apply_control(carla.VehicleControl(throttle=0.6, steer=0))

        # ============== MPC ============== #
        if vehicle_id not in self.path_dict.keys():
            self.a_old_dict[vehicle_id] = None
            self.delta_old_dict[vehicle_id] = None
            self.path_dict[vehicle_id] = PATH(cx=transform.location.x, cy=transform.location.y, cyaw=np.deg2rad(transform.rotation.yaw), cv=10)
        else:
            self.path_dict[vehicle_id].update_route(cx=transform.location.x, cy=transform.location.y, cyaw=np.deg2rad(transform.rotation.yaw), cv=velocity)
        
        v_vector = self.world.get_actor(vehicle_id).get_velocity()
        v = (v_vector.x**2 + v_vector.y**2 + v_vector.z**2)**(1/2)

        node = Node(x=self.world.get_actor(vehicle_id).get_transform().location.x, 
            y=self.world.get_actor(vehicle_id).get_transform().location.y,
            yaw=np.deg2rad(self.world.get_actor(vehicle_id).get_transform().rotation.yaw),
            v=v)    # get the current state of the vehicle
        
        # === 1) trace a desired traj === #

        z_ref, target_ind = calc_ref_trajectory_in_T_step(node, self.path_dict[vehicle_id])

        if (z_ref[3, -1] - node.yaw) > np.pi:
            node = Node(x=self.world.get_actor(vehicle_id).get_transform().location.x, 
                y=self.world.get_actor(vehicle_id).get_transform().location.y,
                yaw=np.deg2rad(self.world.get_actor(vehicle_id).get_transform().rotation.yaw + 360),
                v=v)    # get the current state of the vehicle
        elif (z_ref[3, -1] - node.yaw) < -np.pi:
            z_ref[3, :] += np.pi * 2

        z0 = [node.x, node.y, node.v, node.yaw]
        a_opt, delta_opt = self.a_old_dict[vehicle_id], self.delta_old_dict[vehicle_id]
        try:
            a_opt, delta_opt, x_opt, y_opt, yaw_opt, v_opt = linear_mpc_control(z_ref, z0, a_opt, delta_opt)
        except:
            a_opt, delta_opt = self.a_old_dict[vehicle_id], self.delta_old_dict[vehicle_id]

        # # === 1) trace a desired traj === #

        # # === 2) trace a single point === #
        # z_target = get_destination_in_T_step(node, self.path_dict[vehicle_id])

        # if (z_target[3] - node.yaw) > np.pi:
        #     node = Node(x=self.world.get_actor(vehicle_id).get_transform().location.x, 
        #         y=self.world.get_actor(vehicle_id).get_transform().location.y,
        #         yaw=np.deg2rad(self.world.get_actor(vehicle_id).get_transform().rotation.yaw + 360),
        #         v=v)    # get the current state of the vehicle
        # elif (z_target[3] - node.yaw) < -np.pi:
        #     z_target[3] += np.pi * 2

        # a_opt, delta_opt = self.a_old_dict[vehicle_id], self.delta_old_dict[vehicle_id]
        # a_opt, delta_opt = MPC_module(node, z_target, a_opt, delta_opt)

        # # === 2) trace a single point === #

        self.a_old_dict[vehicle_id], self.delta_old_dict[vehicle_id] = a_opt, delta_opt
        if delta_opt is not None:
            delta_exc, a_exc = delta_opt[0], a_opt[0]
        
        steer = np.rad2deg(delta_exc) / K_STEER    # [-1, 1] ~ [-58, 58]
        # steer += np.random.uniform(-0.15, 0.15)

        if vehicle_id not in steer_record.keys():
            steer_record[vehicle_id] = [steer]
        else:
            steer_record[vehicle_id] += [steer]

        if a_exc > 0:
            vehicle.apply_control(carla.VehicleControl(throttle=min(a_exc*K_THROTTLE, 1.0), steer=steer))
        else:
            vehicle.apply_control(carla.VehicleControl(brake=min(abs(a_exc)*K_BRAKE, 1.0), steer=steer, throttle=0))
        
        # # print(vehicle_id, ' throttle: ', a_exc)
        # # print(vehicle_id, ' delta_exc: ', delta_exc)

        # ============== MPC ============== # 

        if lights is not None:
            vehicle.set_light_state(carla.VehicleLightState(lights))
        return True

    def synchronize_vehicle2(self, vehicle_id, acc=None, delta=None, lights=None):
        """
        Updates vehicle state.

            :param vehicle_id: id of the actor to be updated.
            :param transform: new vehicle transform (i.e., position and rotation).
            :param lights: new vehicle light state.
            :return: True if successfully updated. Otherwise, False.
        """
        vehicle = self.world.get_actor(vehicle_id)
        
        if vehicle is None:
            return False

        # Original synchronization method: update the vehicles to the latest transform
        # vehicle.set_transform(transform)

        # steer = np.rad2deg(delta) / K_STEER * 0.75    # [-1, 1] ~ [-58, 58]
        # if acc > -6:
        #     vehicle.apply_control(carla.VehicleControl(throttle=min((acc+6)*K_THROTTLE*4, 1.0), steer=steer))
        # else:
        #     vehicle.apply_control(carla.VehicleControl(brake=min(abs(acc)*K_BRAKE*0.0, 1.0), steer=steer, throttle=0))

        vehicle.apply_control(carla.VehicleControl(steer=0, throttle=1.0))

        # self.get_actor(114).get_velocity().x
        # self.get_actor(114).get_angular_velocity().x
        
        # # print(vehicle_id, ' throttle: ', a_exc)
        # # print(vehicle_id, ' delta_exc: ', delta_exc)

        if lights is not None:
            vehicle.set_light_state(carla.VehicleLightState(lights))
        return True

    @staticmethod
    def clamp_number(num, min_num, max_num):
        return max(min(num, max(min_num, max_num)), min(min_num, max_num))

    def synchronize_vehicle3(self, vehicle_id, out, lights=None):
        """
        Updates vehicle state.

            :param vehicle_id: id of the actor to be updated.
            :param out: [pred, 2]
            :param transform: new vehicle transform (i.e., position and rotation).
            :param lights: new vehicle light state.
            :return: True if successfully updated. Otherwise, False.
        """
        vehicle = self.world.get_actor(vehicle_id)
        
        if vehicle is None:
            return False

        # Original synchronization method: update the vehicles to the latest transform
        # vehicle.set_transform(transform)
        
        out[:,1] = np.clip(out[:,1], a_min=1e-10, a_max=1e10)
        out = np.concatenate((np.array([[0,0]]), out), axis=0)
        stepwise_movement = out[1:, :] - out[:-1, :]    # [pred, 2]
        # vel = np.linalg.norm(stepwise_movement, axis=1) * 10   # [pred]
        vel = stepwise_movement[:,1] * 10
        curr_vel = np.linalg.norm(np.array([vehicle.get_velocity().x, vehicle.get_velocity().y]))
        v_gap = vel[:1].mean() - curr_vel
        if v_gap >= 0 and out[-3:, 1].mean() > 1.0 and vel[0] > 0:
            # throttle = v_gap * 2.0
            throttle = v_gap * 0.1
            brake = 0
        else:
            throttle = 0
            brake = abs(v_gap) * 0.01
        curr_yaw = vehicle.get_transform().rotation.yaw
        target_yaw = np.arctan2(stepwise_movement[:,1], stepwise_movement[:,0]) - (np.pi/2)
        n_step = 1
        # if abs(out[-1,1])/(abs(out[-1,0])+1e-6) > 3.5:
        #     steer = 0
        if target_yaw[:n_step].mean() < 0:   # left turn
            steer = target_yaw[:n_step].mean() * 2.1    # 0.07
        else:
            steer = target_yaw[:n_step].mean() * 2.5    # 0.07
        throttle = CarlaSimulation.clamp_number(throttle, 0.0, 1.0)
        steer = CarlaSimulation.clamp_number(steer, -1.0, 1.0)
        brake = CarlaSimulation.clamp_number(brake, 0.0, 1.0)
        vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))

        # n_step = 3
        # vel = (out[1:, :] - out[:-1, :]) * 10   # [pred-1, 2]
        # local_yaw = np.rad2deg(np.arctan2(vel[:,0], vel[:,1]))
        # delta_yaw = np.deg2rad(local_yaw[1:] - local_yaw[:-1]) * 10   # [pred-2]
        # yaw_carla = self.get_actor(vehicle_id).get_transform().rotation.yaw    # degree
        # rotation_back = rotation_matrix_back(np.deg2rad(yaw_carla - local_yaw[0])) # [2,2]
        # vel_global = rotation_back @ vel[0:n_step,:].transpose(1,0) # [2, step]
        # vel_global = vel_global.mean(axis=1)    # [2]
        # vehicle.set_target_velocity(carla.Vector3D(vel_global[0], vel_global[1], 0.0))
        # vehicle.set_target_angular_velocity(carla.Vector3D(0, 0, -delta_yaw[0].item()))
        

        # steer = np.rad2deg(delta) / K_STEER * 0.75    # [-1, 1] ~ [-58, 58]
        # if acc > -6:
        #     vehicle.apply_control(carla.VehicleControl(throttle=min((acc+6)*K_THROTTLE*4, 1.0), steer=steer))
        # else:
        #     vehicle.apply_control(carla.VehicleControl(brake=min(abs(acc)*K_BRAKE*0.0, 1.0), steer=steer, throttle=0))

        # self.get_actor(114).get_velocity().x
        # self.get_actor(114).get_angular_velocity().x
        
        # # print(vehicle_id, ' throttle: ', a_exc)
        # # print(vehicle_id, ' delta_exc: ', delta_exc)

        if lights is not None:
            vehicle.set_light_state(carla.VehicleLightState(lights))
        return True

    def synchronize_traffic_light(self, landmark_id, state):
        """
        Updates traffic light state.

            :param landmark_id: id of the landmark to be updated.
            :param state: new traffic light state.
            :return: True if successfully updated. Otherwise, False.
        """
        if not landmark_id in self._tls:
            logging.warning('Landmark %s not found in carla', landmark_id)
            return False

        traffic_light = self._tls[landmark_id]
        traffic_light.set_state(state)
        return True

    def tick(self):
        """
        Tick to carla simulation.
        """
        self.world.tick()

        # Update data structures for the current frame.
        current_actors = set(
            [vehicle.id for vehicle in self.world.get_actors().filter('vehicle.*')])
        self.spawned_actors = current_actors.difference(self._active_actors)
        self.destroyed_actors = self._active_actors.difference(current_actors)
        self._active_actors = current_actors

    def close(self):
        """
        Closes carla client.
        """
        for actor in self.world.get_actors():
            if actor.type_id == 'traffic.traffic_light':
                actor.freeze(False)
