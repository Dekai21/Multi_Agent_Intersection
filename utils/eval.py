#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020-06-18 22:00
# @Author  : Xiaoke Huang
# @Email   : xiaokehuang@foxmail.com

import math
import os
from pprint import pprint
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
# from argoverse.evaluation.eval_forecasting import \
#     get_displacement_errors_and_miss_rate
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Circle

from utils.config import OBS_LEN, STEPWISE_LOSS


LOW_PROB_THRESHOLD_FOR_METRICS = 0.05


def get_tgt_agents_pos(data)-> List[np.ndarray]:
    assert data.num_graphs == 1, f"The batch size should be 1."
    tgt_agents_pos = []
    num_tgt_agent = data.y.shape[0] # TODO: check the shape of y, if the 0 dim is the num of tgt
    for i_tgt_agent in range(1, 1+num_tgt_agent):
        tgt_agents_pos.append(data.x[i_tgt_agent*(OBS_LEN-1)-1, 2:4].cpu().numpy().reshape(1,2))
    
    return tgt_agents_pos


def plot_gts_and_fcsts(global_forecasted_trajectories: List[np.ndarray], 
                       global_gt_trajectories: List[np.ndarray], 
                       tgt_agent_pos: List[np.ndarray],
                       epoch: int,
                       scene_name: str,
                       plt_show: bool,
                       plt_write: bool,
                       folder_name: str = 'images'):
    
    if not plt_show and not plt_write:
        return None
    
    assert len(global_forecasted_trajectories) == len(global_gt_trajectories)
    _global_forecasted_trajectory, _global_gt_trajectory = [], []
    circles = []
    for global_forecasted_trajectory, global_gt_trajectory in zip(global_forecasted_trajectories, global_gt_trajectories):
        _global_forecasted_trajectory.append([[pt[0], pt[1]] for pt in global_forecasted_trajectory])
        _global_gt_trajectory.append([[pt[0], pt[1]] for pt in global_gt_trajectory])
    for pos in tgt_agent_pos:
        circle = Circle((pos[0, 0], pos[0, 1]), radius=1.5, fill=False)   # pos.shape = [1, 2]
        circles.append(circle)
    fig, ax = plt.subplots()
    global_forecasted_trajectory_collection = LineCollection(_global_forecasted_trajectory, edgecolors='red')
    global_gt_trajectory_collection = LineCollection(_global_gt_trajectory, edgecolors='green')
    circle_collection = PatchCollection(circles, edgecolors='grey', linewidth=1, linestyles='dotted', match_original=True)
    ax.add_collection(global_forecasted_trajectory_collection)
    ax.add_collection(global_gt_trajectory_collection)
    ax.add_collection(circle_collection)
    ax.set_xlim([-55, 55])
    ax.set_ylim([-55, 55])
    # ax.autoscale_view()
    ax.set_aspect('equal')
    if plt_show:
        plt.show(block=False)
        plt.pause(1)
    if plt_write:
        folder_name = os.path.join('images', folder_name)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        # file_name = "scenario%04d.epoch%03d.png" % (plt_idx, epoch)
        plt.savefig(os.path.join(folder_name, scene_name))
    fig.clf()
    plt.close()
    return None


def get_fde(forecasted_trajectory: np.ndarray, gt_trajectory: np.ndarray) -> float:
    """Compute Final Displacement Error.

    Args:
        forecasted_trajectory: Predicted trajectory with shape (pred_len x 2)
        gt_trajectory: Ground truth trajectory with shape (pred_len x 2)

    Returns:
        fde: Final Displacement Error

    """
    fde = math.sqrt(
        (forecasted_trajectory[-1, 0] - gt_trajectory[-1, 0]) ** 2
        + (forecasted_trajectory[-1, 1] - gt_trajectory[-1, 1]) ** 2
    )
    return fde


def get_ade(forecasted_trajectory: np.ndarray, gt_trajectory: np.ndarray) -> float:
    """Compute Average Displacement Error.

    Args:
        forecasted_trajectory: Predicted trajectory with shape (pred_len x 2)
        gt_trajectory: Ground truth trajectory with shape (pred_len x 2)

    Returns:
        ade: Average Displacement Error

    """
    pred_len = forecasted_trajectory.shape[0]
    ade = float(
        sum(
            math.sqrt(
                (forecasted_trajectory[i, 0] - gt_trajectory[i, 0]) ** 2
                + (forecasted_trajectory[i, 1] - gt_trajectory[i, 1]) ** 2
            )
            for i in range(pred_len)
        )
        / pred_len
    )
    return ade


def get_displacement_errors_and_miss_rate(
    forecasted_trajectories: Dict[int, List[np.ndarray]],
    gt_trajectories: Dict[int, np.ndarray],
    max_guesses: int,
    horizon: int,
    miss_threshold: float,
    forecasted_probabilities: Optional[Dict[int, List[float]]] = None,
) -> Dict[str, float]:
    """Compute min fde and ade for each sample.

    Note: Both min_fde and min_ade values correspond to the trajectory which has minimum fde.
    The Brier Score is defined here:
        Brier, G. W. Verification of forecasts expressed in terms of probability. Monthly weather review, 1950.
        https://journals.ametsoc.org/view/journals/mwre/78/1/1520-0493_1950_078_0001_vofeit_2_0_co_2.xml

    Args:
        forecasted_trajectories: Predicted top-k trajectory dict with key as seq_id and value as list of trajectories.
                Each element of the list is of shape (pred_len x 2).
        gt_trajectories: Ground Truth Trajectory dict with key as seq_id and values as trajectory of
                shape (pred_len x 2)
        max_guesses: Number of guesses allowed
        horizon: Prediction horizon
        miss_threshold: Distance threshold for the last predicted coordinate
        forecasted_probabilities: Probabilites associated with forecasted trajectories.

    Returns:
        metric_results: Metric values for minADE, minFDE, MR, p-minADE, p-minFDE, p-MR, brier-minADE, brier-minFDE
    """
    metric_results: Dict[str, float] = {}
    min_ade, prob_min_ade, brier_min_ade = [], [], []
    min_fde, prob_min_fde, brier_min_fde = [], [], []
    n_misses, prob_n_misses = [], []
    for k, v in gt_trajectories.items():
        curr_min_ade = float("inf")
        curr_min_fde = float("inf")
        min_idx = 0
        max_num_traj = min(max_guesses, len(forecasted_trajectories[k]))

        # If probabilities available, use the most likely trajectories, else use the first few
        if forecasted_probabilities is not None:
            sorted_idx = np.argsort([-x for x in forecasted_probabilities[k]], kind="stable")
            # sorted_idx = np.argsort(forecasted_probabilities[k])[::-1]
            pruned_probabilities = [forecasted_probabilities[k][t] for t in sorted_idx[:max_num_traj]]
            # Normalize
            prob_sum = sum(pruned_probabilities)
            pruned_probabilities = [p / prob_sum for p in pruned_probabilities]
        else:
            sorted_idx = np.arange(len(forecasted_trajectories[k]))
        pruned_trajectories = [forecasted_trajectories[k][t] for t in sorted_idx[:max_num_traj]]

        for j in range(len(pruned_trajectories)):
            fde = get_fde(pruned_trajectories[j][:horizon], v[:horizon])
            if fde < curr_min_fde:
                min_idx = j
                curr_min_fde = fde
        curr_min_ade = get_ade(pruned_trajectories[min_idx][:horizon], v[:horizon])
        min_ade.append(curr_min_ade)
        min_fde.append(curr_min_fde)
        n_misses.append(curr_min_fde > miss_threshold)

        if forecasted_probabilities is not None:
            prob_n_misses.append(1.0 if curr_min_fde > miss_threshold else (1.0 - pruned_probabilities[min_idx]))
            prob_min_ade.append(
                min(
                    -np.log(pruned_probabilities[min_idx]),
                    -np.log(LOW_PROB_THRESHOLD_FOR_METRICS),
                )
                + curr_min_ade
            )
            brier_min_ade.append((1 - pruned_probabilities[min_idx]) ** 2 + curr_min_ade)
            prob_min_fde.append(
                min(
                    -np.log(pruned_probabilities[min_idx]),
                    -np.log(LOW_PROB_THRESHOLD_FOR_METRICS),
                )
                + curr_min_fde
            )
            brier_min_fde.append((1 - pruned_probabilities[min_idx]) ** 2 + curr_min_fde)

    metric_results["minADE"] = sum(min_ade) / len(min_ade)
    metric_results["minFDE"] = sum(min_fde) / len(min_fde)
    metric_results["MR"] = sum(n_misses) / len(n_misses)
    if forecasted_probabilities is not None:
        metric_results["p-minADE"] = sum(prob_min_ade) / len(prob_min_ade)
        metric_results["p-minFDE"] = sum(prob_min_fde) / len(prob_min_fde)
        metric_results["p-MR"] = sum(prob_n_misses) / len(prob_n_misses)
        metric_results["brier-minADE"] = sum(brier_min_ade) / len(brier_min_ade)
        metric_results["brier-minFDE"] = sum(brier_min_fde) / len(brier_min_fde)
    return metric_results


def get_eval_metric_results(model, 
                            data_loader, 
                            device, 
                            out_channels, 
                            max_n_guesses, 
                            horizon, 
                            miss_threshold,
                            epoch: int = 0,
                            plt_ids: List[int] = [0],
                            plt_show: bool = False,
                            plt_write: bool = False):
    """
    ADE, FDE, and Miss Rate
    """
    forecasted_trajectories, gt_trajectories = {}, {}
    
    seq_id = 0
    model.eval()
    with torch.no_grad():
        for data_id, data in enumerate(data_loader):
            global_forecasted_trajectories, global_gt_trajectories = [], []
            gt = None
            scene_name = data.scene_name[0]
            # mutil gpu testing
            if isinstance(data, List):
                gt = torch.cat([i.y for i in data], 0).view(-1, out_channels).to(device)
            # single gpu testing
            else:
                data = data.to(device)
                gt = data.y.view(-1, out_channels).to(device)
            
            tgt_agent_pos = get_tgt_agents_pos(data)

            out = model(data)
            for i in range(gt.size(0)):
                if STEPWISE_LOSS:
                    pred_y = out[i].view((-1, 2)).cumsum(axis=0).cpu().numpy()
                    y = gt[i].view((-1, 2)).cumsum(axis=0).cpu().numpy()
                else:
                    pred_y = out[i].view((-1, 2)).cpu().numpy()
                    y = gt[i].view((-1, 2)).cpu().numpy()
                forecasted_trajectories[seq_id] = [pred_y]
                gt_trajectories[seq_id] = y
                global_forecasted_trajectories.append(pred_y + tgt_agent_pos[i])
                global_gt_trajectories.append(y + tgt_agent_pos[i])
                seq_id += 1
            if data_id in plt_ids:
                scene_name = f'{scene_name}-epoch-{epoch:0>5}'
                plot_gts_and_fcsts(global_forecasted_trajectories, global_gt_trajectories, tgt_agent_pos, epoch, scene_name, plt_show, plt_write)
            None
        
        metric_results = get_displacement_errors_and_miss_rate(
            forecasted_trajectories, gt_trajectories, max_n_guesses, horizon, miss_threshold
        )



        return metric_results

def eval_loss():
    raise NotImplementedError("not finished yet")
    model.eval()
    from utils.viz_utils import show_pred_and_gt
    with torch.no_grad():
        accum_loss = .0
        for sample_id, data in enumerate(train_loader):
            data = data.to(device)
            gt = data.y.view(-1, out_channels).to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = F.mse_loss(out, gt)
            accum_loss += batch_size * loss.item()
            print(f"loss for sample {sample_id}: {loss.item():.3f}")

            for i in range(gt.size(0)):
                pred_y = out[i].numpy().reshape((-1, 2)).cumsum(axis=0)
                y = gt[i].numpy().reshape((-1, 2)).cumsum(axis=0)
                show_pred_and_gt(pred_y, y)
                plt.show()
        print(f"eval overall loss: {accum_loss / len(ds):.3f}")
