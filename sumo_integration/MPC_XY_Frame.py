"""
Linear MPC controller (X-Y frame)
Code modified from MPC_XY_Frame.py developed by Huiming Zhou et al.
link: https://github.com/zhm-real/MotionPlanning/blob/master/Control/MPC_XY_Frame.py
"""

import os
import sys
import math
import cvxpy
import numpy as np
import matplotlib.pyplot as plt

from typing import Union

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../MotionPlanning/")

# import Control.draw as draw
# import CurvesGenerator.reeds_shepp as rs
# import CurvesGenerator.cubic_spline as cs

K_STEER = 56.0
K_THROTTLE = 1.5
K_BRAKE = 0.15

class P:
    # System config
    NX = 4  # state vector: z = [x, y, v, phi]
    NU = 2  # input vector: u = [acceleration, steer]
    T = 3  # finite time horizon length
    T_aug = 30  # finite time horizon length
    # Dekai: if T is 1, the vehicle would have a larger turning radius

    # MPC config
    # Q = np.diag([12.0, 12.0, 1.0, 12.0])  # penalty for states   # Dekai: if set the third value (penalty for velocity) to 0.0, the vehicle is difficult to start. 
    # Qf = np.diag([5.0, 5.0, 1.0, 20.0])  # penalty for end state # Dekai: since now we only trace a single target point but not a desired traj, only Qf is used but not Q
    Qf = np.diag([5.0, 5.0, 0.0, 20.0])  # penalty for end state # Dekai: since now we only trace a single target point but not a desired traj, only Qf is used but not Q
    R = np.diag([0.01, 5.8])  # penalty for inputs  # Dekai: had better choose large penalty for steering to avoid zig-zag
    Rd = np.diag([0.01, 10.1])  # penalty for change of inputs

    dist_stop = 1.5  # stop permitted when dist to goal < dist_stop
    speed_stop = 0.5 / 3.6  # stop permitted when speed < speed_stop
    time_max = 500.0  # max simulation time
    iter_max = 5  # max iteration
    target_speed = 10.0 / 3.6  # target speed
    N_IND = 10  # search index number
    dt = 0.1  # time step
    d_dist = 1.0  # dist step
    du_res = 0.25  # threshold for stopping iteration

    # vehicle config
    RF = 3.3  # [m] distance from rear to vehicle front end of vehicle
    RB = 0.8  # [m] distance from rear to vehicle back end of vehicle
    W = 2.4  # [m] width of vehicle
    WD = 0.7 * W  # [m] distance between left-right wheels
    WB = 2.5  # [m] Wheel base
    TR = 0.44  # [m] Tyre radius
    TW = 0.7  # [m] Tyre width

    steer_max = np.deg2rad(60.0)  # max steering angle [rad]
    steer_change_max = np.deg2rad(30.0)  # maximum steering speed [rad/s]
    speed_max = 55.0 / 3.6  # maximum speed [m/s]
    speed_min = -20.0 / 3.6  # minimum speed [m/s]
    acceleration_max = 1.0  # maximum acceleration [m/s2]


class Node:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0, direct=1.0):   # current state
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.direct = direct

    def update(self, a, delta, direct):
        delta = self.limit_input_delta(delta)
        self.x += self.v * math.cos(self.yaw) * P.dt
        self.y += self.v * math.sin(self.yaw) * P.dt
        self.yaw += self.v / P.WB * math.tan(delta) * P.dt
        self.direct = direct
        self.v += self.direct * a * P.dt
        self.v = self.limit_speed(self.v)

    @staticmethod
    def limit_input_delta(delta):
        if delta >= P.steer_max:
            return P.steer_max

        if delta <= -P.steer_max:
            return -P.steer_max

        return delta

    @staticmethod
    def limit_speed(v):
        if v >= P.speed_max:
            return P.speed_max

        if v <= P.speed_min:
            return P.speed_min

        return v


class PATH:
    def __init__(self, cx, cy, cyaw, cv, ck=None):
        self.cx = [cx]
        self.cy = [cy]
        self.cyaw = [cyaw]
        # self.ck = [ck]
        # self.length = len(cx)
        self.ind_old = 0
        self.cv = [cv]

    def update_route(self, cx, cy, cyaw, cv):
        self.cx += [cx]
        self.cy += [cy]
        self.cyaw += [cyaw]
        self.cv += [cv]

        # self.cx = [cx]
        # self.cy = [cy]
        # self.cyaw = [cyaw]

    def nearest_index(self, node):
        """
        calc index of the nearest node in N steps
        :param node: current information
        :return: nearest index, lateral distance to ref point
        """

        dx = [node.x - x for x in self.cx[self.ind_old: (self.ind_old + P.N_IND)]]
        dy = [node.y - y for y in self.cy[self.ind_old: (self.ind_old + P.N_IND)]]
        dist = np.hypot(dx, dy)

        ind_in_N = int(np.argmin(dist))
        ind = self.ind_old + ind_in_N
        self.ind_old = ind

        rear_axle_vec_rot_90 = np.array([[math.cos(node.yaw + math.pi / 2.0)],
                                         [math.sin(node.yaw + math.pi / 2.0)]])

        vec_target_2_rear = np.array([[dx[ind_in_N]],
                                      [dy[ind_in_N]]])

        er = np.dot(vec_target_2_rear.T, rear_axle_vec_rot_90)
        er = er[0][0]

        return ind, er


def calc_ref_trajectory_in_T_step(node, ref_path, sp=None)->np.ndarray:
    """
    calc referent trajectory in T steps: [x, y, v, yaw]
    using the current velocity, calc the T points along the reference path
    :param node: current information
    :param ref_path: reference path: [x, y, v, yaw]
    :param sp: speed profile (designed speed strategy)
    :return: reference trajectory [4, T+1]
    """

    # if sp is None:
        # sp = np.ones(200, dtype=np.float) * 40 / 3.6    # max speed in sumo routefile
        
    z_ref = np.zeros((P.NX, P.T + 1))
    length = len(ref_path.cx)

    # ============== get the clost step and look further for N steps ============== #
    
    # ind, _ = ref_path.nearest_index(node)

    # z_ref[0, 0] = ref_path.cx[ind]
    # z_ref[1, 0] = ref_path.cy[ind]
    # # z_ref[2, 0] = sp[ind]
    # z_ref[2, 0] = ref_path.cv[ind]
    # z_ref[3, 0] = ref_path.cyaw[ind]

    # dist_move = 0.0

    # for i in range(1, P.T + 1):
    #     dist_move += abs(node.v) * P.dt
    #     ind_move = int(round(dist_move / P.d_dist))
    #     index = min(ind + ind_move, length - 1)

    #     z_ref[0, i] = ref_path.cx[index]
    #     z_ref[1, i] = ref_path.cy[index]
    #     # z_ref[2, i] = sp[index]
    #     z_ref[2, i] = ref_path.cv[index]
    #     z_ref[3, i] = ref_path.cyaw[index]

    # ============== get the clost step and look further for N steps ============== #

    # ============== get the last N steps ============== #

    z_ref[0, -1] = ref_path.cx[-1]
    z_ref[1, -1] = ref_path.cy[-1]
    z_ref[2, -1] = ref_path.cv[-1]
    z_ref[3, -1] = ref_path.cyaw[-1]    
    dist_move = 0.0
    for i in range(P.T - 1, -1, -1):
        dist_move += abs(node.v) * P.dt
        ind_move = int(round(dist_move / P.d_dist))
        index = max(length -1 - ind_move, 0)

        z_ref[0, i] = ref_path.cx[index]
        z_ref[1, i] = ref_path.cy[index]
        z_ref[2, i] = ref_path.cv[index]
        z_ref[3, i] = ref_path.cyaw[index]

    # ============== get the last N steps ============== #

    return z_ref, 0


def get_destination_in_T_step(node, ref_path)->np.ndarray:
    """
    calc desired destination in T steps: [x, y, v, yaw]
    :param node: current information
    :param ref_path: reference path: [x, y, v, yaw]
    :return: destination [4]
    """

    z_target = np.zeros(4)
    z_target[0] = ref_path.cx[-1]
    z_target[1] = ref_path.cy[-1]
    z_target[2] = ref_path.cv[-1]
    z_target[3] = ref_path.cyaw[-1]    

    return z_target


def linear_mpc_control(z_ref, z0, a_old, delta_old):
    """
    linear mpc controller
    :param z_ref: reference trajectory in T steps
    :param z0: initial state vector
    :param a_old: acceleration of T steps of last time
    :param delta_old: delta of T steps of last time
    :return: acceleration and delta strategy based on current information
    """

    if a_old is None or delta_old is None:
        a_old = [0.0] * P.T
        delta_old = [0.0] * P.T

    x, y, yaw, v = None, None, None, None

    for k in range(P.iter_max):
        z_bar = predict_states_in_T_step(z0, a_old, delta_old, z_ref)
        a_rec, delta_rec = a_old[:], delta_old[:]
        a_old, delta_old, x, y, yaw, v = solve_linear_mpc(z_ref, z_bar, z0, delta_old)

        du_a_max = max([abs(ia - iao) for ia, iao in zip(a_old, a_rec)])
        du_d_max = max([abs(ide - ido) for ide, ido in zip(delta_old, delta_rec)])

        if max(du_a_max, du_d_max) < P.du_res:
            break

    return a_old, delta_old, x, y, yaw, v


def linear_mpc_control_data_aug(z_ref, z0, a_old, delta_old):
    """
    linear mpc controller
    :param z_ref: reference trajectory in T steps
    :param z0: initial state vector
    :param a_old: acceleration of T steps of last time
    :param delta_old: delta of T steps of last time
    :return: acceleration and delta strategy based on current information
    """

    if a_old is None or delta_old is None:
        a_old = [0.0] * P.T_aug
        delta_old = [0.0] * P.T_aug

    x, y, yaw, v = None, None, None, None

    for k in range(P.iter_max):
        z_bar = predict_states_in_T_step(z0, a_old, delta_old, z_ref, pred_len=P.T_aug)
        a_rec, delta_rec = a_old[:], delta_old[:]
        a_old, delta_old, x, y, yaw, v = solve_linear_mpc(z_ref, z_bar, z0, delta_old, pred_len=P.T_aug)

        du_a_max = max([abs(ia - iao) for ia, iao in zip(a_old, a_rec)])
        du_d_max = max([abs(ide - ido) for ide, ido in zip(delta_old, delta_rec)])

        if max(du_a_max, du_d_max) < P.du_res:
            break

    return a_old, delta_old, x, y, yaw, v


def predict_states_in_T_step(z0: list, a: np.ndarray, delta: np.ndarray, z_ref: np.ndarray, pred_len: int=P.T):
    """
    given the current state, using the acceleration and delta strategy of last time,
    predict the states of vehicle in T steps.
    :param z0: [4], initial state
    :param a: [T], acceleration strategy of last time
    :param delta: [T], delta strategy of last time
    :param z_ref: [4, T+1], reference trajectory
    :return: predict states in T steps (z_bar, used for calc linear motion model)
    """

    z_bar = z_ref * 0.0

    for i in range(P.NX):
        z_bar[i, 0] = z0[i]

    node = Node(x=z0[0], y=z0[1], v=z0[2], yaw=z0[3])

    for ai, di, i in zip(a, delta, range(1, pred_len + 1)):
        node.update(ai, di, 1.0)    # 1.0 is forward direction
        z_bar[0, i] = node.x
        z_bar[1, i] = node.y
        z_bar[2, i] = node.v
        z_bar[3, i] = node.yaw

    return z_bar


def predict_states_in_T_step_2(curr_state: Node, a: np.ndarray, delta: np.ndarray, T: int = P.T)-> np.ndarray:
    """
    given the current state, using the acceleration and delta strategy of last time,
    predict the states of vehicle in T steps.
    :param curr_state: [x, y, v, yaw], initial state
    :param a: [T], acceleration strategy of last time
    :param delta: [T], delta strategy of last time
    :param T: num of future steps
    :return: [4, T+1] predict states in T steps (including curr state)
    """

    z_bar = np.zeros((4, T+1))
    z_bar[:, 0] = curr_state.x, curr_state.y, curr_state.v, curr_state.yaw

    for ai, di, i in zip(a, delta, range(1, P.T + 1)):
        curr_state.update(ai, di, 1.0)    # 1.0 is forward direction
        z_bar[0, i] = curr_state.x
        z_bar[1, i] = curr_state.y
        z_bar[2, i] = curr_state.v
        z_bar[3, i] = curr_state.yaw

    return z_bar


def calc_linear_discrete_model(v, phi, delta):
    """
    calc linear and discrete time dynamic model.
    :param v: speed: v_bar
    :param phi: angle of vehicle: phi_bar
    :param delta: steering angle: delta_bar
    :return: A, B, C
    """

    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)
    P_dt_v = P.dt * v

    # A = np.array([[1.0, 0.0, P.dt * cos_phi, - P.dt * v * sin_phi],
    #               [0.0, 1.0, P.dt * sin_phi, P.dt * v * cos_phi],
    #               [0.0, 0.0, 1.0, 0.0],
    #               [0.0, 0.0, P.dt * math.tan(delta) / P.WB, 1.0]])

    # B = np.array([[0.0, 0.0],
    #               [0.0, 0.0],
    #               [P.dt, 0.0],
    #               [0.0, P.dt * v / (P.WB * math.cos(delta) ** 2)]])

    # C = np.array([P.dt * v * sin_phi * phi,
    #               -P.dt * v * cos_phi * phi,
    #               0.0,
    #               -P.dt * v * delta / (P.WB * math.cos(delta) ** 2)])

    A = np.array([[1.0, 0.0, P.dt * cos_phi, - P_dt_v * sin_phi],
                  [0.0, 1.0, P.dt * sin_phi, P_dt_v * cos_phi],
                  [0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, P.dt * math.tan(delta) / P.WB, 1.0]])

    B = np.array([[0.0, 0.0],
                  [0.0, 0.0],
                  [P.dt, 0.0],
                  [0.0, P_dt_v / (P.WB * math.cos(delta) ** 2)]])

    C = np.array([P_dt_v * sin_phi * phi,
                  -P_dt_v * cos_phi * phi,
                  0.0,
                  -P_dt_v * delta / (P.WB * math.cos(delta) ** 2)])

    return A, B, C


def solve_linear_mpc(z_ref: np.ndarray, z_bar: np.ndarray, z0: list, d_bar: np.ndarray, pred_len: int=P.T):
    """
    solve the quadratic optimization problem using cvxpy, solver: OSQP
    :param z_ref: [4, 7], reference trajectory (desired trajectory: [x, y, v, yaw])
    :param z_bar: [4, 7], predicted states in T steps
    :param z0: [4], initial state
    :param d_bar: [6], delta_bar
    :return: optimal acceleration and steering strategy
    """

    z = cvxpy.Variable((P.NX, pred_len + 1))
    u = cvxpy.Variable((P.NU, pred_len))

    cost = 0.0
    constrains = []

    for t in range(pred_len):
        cost += cvxpy.quad_form(u[:, t], P.R)
        # cost += cvxpy.quad_form(z_ref[:, t] - z[:, t], P.Q)

        A, B, C = calc_linear_discrete_model(z_bar[2, t], z_bar[3, t], d_bar[t])

        constrains += [z[:, t + 1] == A @ z[:, t] + B @ u[:, t] + C]

        if t < pred_len - 1:
            cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], P.Rd)
            # constrains += [cvxpy.abs(u[1, t + 1] - u[1, t]) <= P.steer_change_max * P.dt]

    cost += cvxpy.quad_form(z_ref[:, pred_len] - z[:, pred_len], P.Qf)

    constrains += [z[:, 0] == z0]
    # constrains += [z[2, :] <= P.speed_max]
    # constrains += [z[2, :] >= P.speed_min]
    # constrains += [cvxpy.abs(u[0, :]) <= P.acceleration_max]
    constrains += [cvxpy.abs(u[1, :]) <= P.steer_max]

    prob = cvxpy.Problem(cvxpy.Minimize(cost), constrains)
    prob.solve(solver=cvxpy.OSQP)

    a, delta, x, y, yaw, v = None, None, None, None, None, None

    if prob.status == cvxpy.OPTIMAL or \
            prob.status == cvxpy.OPTIMAL_INACCURATE:
        x = z.value[0, :]
        y = z.value[1, :]
        v = z.value[2, :]
        yaw = z.value[3, :]
        a = u.value[0, :]
        delta = u.value[1, :]
    else:
        print("Cannot solve linear mpc!")

    return a, delta, x, y, yaw, v


def solve_linear_mpc_2(z_target: np.ndarray, z_bar: np.ndarray, z0: list, d_bar: np.ndarray):
    """
    solve the quadratic optimization problem using cvxpy, solver: OSQP
    :param z_target: [4], target destination (desired: [x, y, v, yaw])
    :param z_bar: [4, T+1], predicted states in T steps
    :param z0: [4], initial state
    :param d_bar: [T], delta_bar
    :return: optimal acceleration and steering strategy
    """

    z = cvxpy.Variable((P.NX, P.T + 1))
    u = cvxpy.Variable((P.NU, P.T))

    cost = 0.0
    constrains = []

    for t in range(P.T):
        cost += cvxpy.quad_form(u[:, t], P.R)
        # cost += cvxpy.quad_form(z_ref[:, t] - z[:, t], P.Q)

        A, B, C = calc_linear_discrete_model(z_bar[2, t], z_bar[3, t], d_bar[t])

        constrains += [z[:, t + 1] == A @ z[:, t] + B @ u[:, t] + C]

        if t < P.T - 1:
            cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], P.Rd)
            # constrains += [cvxpy.abs(u[1, t + 1] - u[1, t]) <= P.steer_change_max * P.dt]

    cost += cvxpy.quad_form(z_target - z[:, P.T], P.Qf)

    constrains += [z[:, 0] == z0]
    # constrains += [z[2, :] <= P.speed_max]
    # constrains += [z[2, :] >= P.speed_min]
    # constrains += [cvxpy.abs(u[0, :]) <= P.acceleration_max]
    constrains += [cvxpy.abs(u[1, :]) <= P.steer_max]

    prob = cvxpy.Problem(cvxpy.Minimize(cost), constrains)
    prob.solve(solver=cvxpy.OSQP)

    a, delta, x, y, yaw, v = None, None, None, None, None, None

    if prob.status == cvxpy.OPTIMAL or \
            prob.status == cvxpy.OPTIMAL_INACCURATE:
        x = z.value[0, :]
        y = z.value[1, :]
        v = z.value[2, :]
        yaw = z.value[3, :]
        a = u.value[0, :]
        delta = u.value[1, :]
    else:
        print("Cannot solve linear mpc!")

    return a, delta, x, y, yaw, v


def calc_speed_profile(cx, cy, cyaw, target_speed)-> list:
    """
    design appropriate speed strategy
    :param cx: x of reference path [m]
    :param cy: y of reference path [m]
    :param cyaw: yaw of reference path [m]
    :param target_speed: target speed [m/s]
    :return: speed profile
    """

    speed_profile = [target_speed] * len(cx)
    direction = 1.0  # forward

    # Set stop point
    for i in range(len(cx) - 1):
        dx = cx[i + 1] - cx[i]
        dy = cy[i + 1] - cy[i]

        move_direction = math.atan2(dy, dx)

        if dx != 0.0 and dy != 0.0:
            dangle = abs(pi_2_pi(move_direction - cyaw[i]))
            if dangle >= math.pi / 4.0:
                direction = -1.0
            else:
                direction = 1.0

        if direction != 1.0:
            speed_profile[i] = - target_speed
        else:
            speed_profile[i] = target_speed

    speed_profile[-1] = 0.0

    return speed_profile


def pi_2_pi(angle):
    if angle > math.pi:
        return angle - 2.0 * math.pi

    if angle < -math.pi:
        return angle + 2.0 * math.pi

    return angle


def MPC_module(curr_state: Node, target_state: np.ndarray, a_old: list, delta_old: list, T: int = P.T)-> Union[list, list]:
    """
    :param curr_state: Node[x, y, v, yaw]
    :param target_state: [4], [x, y, v, yaw]
    :param a_old: [T], if init, input None
    :param delta_old: [T], if init, input None
    :param T: num of steps to arrive the destination
    
    :return a: [T]
    :return delta: [T]
    """

    if a_old is None or delta_old is None:
        a_old = [0.0] * P.T
        delta_old = [0.0] * P.T
    else:
        assert len(a_old) == T and len(delta_old) == T

    x, y, yaw, v = None, None, None, None
    z0 = [curr_state.x, curr_state.y, curr_state.v, curr_state.yaw]

    for k in range(P.iter_max):
        z_bar = predict_states_in_T_step_2(curr_state, a_old, delta_old, T)
        a_rec, delta_rec = a_old[:], delta_old[:]
        a_old, delta_old, x, y, yaw, v = solve_linear_mpc_2(target_state, z_bar, z0, delta_old)

        du_a_max = max([abs(ia - iao) for ia, iao in zip(a_old, a_rec)])
        du_d_max = max([abs(ide - ido) for ide, ido in zip(delta_old, delta_rec)])

        if max(du_a_max, du_d_max) < P.du_res:
            break

    return a_old, delta_old

# def main():
#     ax = [0.0, 15.0, 30.0, 50.0, 60.0]
#     ay = [0.0, 40.0, 15.0, 30.0, 0.0]
#     cx, cy, cyaw, ck, s = cs.calc_spline_course(
#         ax, ay, ds=P.d_dist)    # target route

#     sp = calc_speed_profile(cx, cy, cyaw, P.target_speed)

#     ref_path = PATH(cx, cy, cyaw, ck)
#     node = Node(x=cx[0], y=cy[0], yaw=cyaw[0], v=0.0)

#     time = 0.0
#     x = [node.x]
#     y = [node.y]
#     yaw = [node.yaw]
#     v = [node.v]
#     t = [0.0]
#     d = [0.0]
#     a = [0.0]

#     delta_opt, a_opt = None, None
#     a_exc, delta_exc = 0.0, 0.0

#     while time < P.time_max:
#         z_ref, target_ind = \
#             calc_ref_trajectory_in_T_step(node, ref_path, sp)

#         z0 = [node.x, node.y, node.v, node.yaw]

#         a_opt, delta_opt, x_opt, y_opt, yaw_opt, v_opt = \
#             linear_mpc_control(z_ref, z0, a_opt, delta_opt)

#         if delta_opt is not None:
#             delta_exc, a_exc = delta_opt[0], a_opt[0]

#         node.update(a_exc, delta_exc, 1.0)
#         time += P.dt

#         x.append(node.x)
#         y.append(node.y)
#         yaw.append(node.yaw)
#         v.append(node.v)
#         t.append(time)
#         d.append(delta_exc)
#         a.append(a_exc)

#         dist = math.hypot(node.x - cx[-1], node.y - cy[-1])

#         if dist < P.dist_stop and \
#                 abs(node.v) < P.speed_stop:
#             break

#         dy = (node.yaw - yaw[-2]) / (node.v * P.dt)
#         steer = rs.pi_2_pi(-math.atan(P.WB * dy))

#         plt.cla()
#         draw.draw_car(node.x, node.y, node.yaw, steer, P)
#         plt.gcf().canvas.mpl_connect('key_release_event',
#                                      lambda event:
#                                      [exit(0) if event.key == 'escape' else None])

#         if x_opt is not None:
#             plt.plot(x_opt, y_opt, color='darkviolet', marker='*')

#         plt.plot(cx, cy, color='gray')
#         plt.plot(x, y, '-b')
#         plt.plot(cx[target_ind], cy[target_ind])
#         plt.axis("equal")
#         plt.title("Linear MPC, " + "v = " + str(round(node.v * 3.6, 2)))
#         plt.pause(0.001)

#     plt.show()


# if __name__ == '__main__':
#     main()
