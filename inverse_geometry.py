#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau
"""
import time
from functools import cache
from typing import TypeAlias

import numpy as np
import pinocchio as pin
from numpy.typing import NDArray
from numpy.linalg import norm
from scipy.optimize import least_squares, minimize

from config import LEFT_HOOK, RIGHT_HOOK, LEFT_HAND, RIGHT_HAND
from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
from tools import getcubeplacement, setcubeplacement, collision

Q: TypeAlias = NDArray[np.float64]

MAX_TRIES = 10  # maximum number of least squares solver attempts before the SLSQP fallback
OBSTACLE_MARGIN = 1e-2
W_POS = 1e-4
TASK_TOL = 1e-3

def get_oMframe_by_id(robot, frame: str) -> pin.SE3:
    frameid = robot.model.getFrameId(frame)
    return robot.data.oMf[frameid]


def get_coll_distance(robot, q: Q) -> float:
    pin.updateGeometryPlacements(robot.model, robot.data, robot.collision_model, robot.collision_data, q)
    idx = pin.computeDistances(robot.collision_model, robot.collision_data)
    return float(robot.collision_data.distanceResults[idx].min_distance)


@cache
def _obstacle_pair_ids(collision_model) -> tuple[int, ...]:
    obstacle_id = collision_model.getGeometryId("obstaclebase_0")
    return tuple(i for i, pair in enumerate(collision_model.collisionPairs) if pair.first == obstacle_id or pair.second == obstacle_id)


def get_obstacle_distance(robot, q):
    """Return the shortest distance from the robot or attached cube to the obstacle."""
    pairs = _obstacle_pair_ids(robot.collision_model)
    pin.framesForwardKinematics(robot.model, robot.data, q)
    pin.updateGeometryPlacements(robot.model, robot.data, robot.collision_model, robot.collision_data, q)
    return min(pin.computeDistance(robot.collision_model, robot.collision_data, i).min_distance for i in pairs)


def is_valid(robot, q: Q) -> bool:
    """Collision free and at least OBSTACLE_MARGIN away from the obstacle."""
    return (not collision(robot, q)) and get_obstacle_distance(robot, q) > OBSTACLE_MARGIN


# for least squares
def residual(q, robot, oMcubeL, oMcubeR, q_ref=None, w_pos: float = W_POS) -> NDArray:
    pin.framesForwardKinematics(robot.model, robot.data, q)
    oMeffL = get_oMframe_by_id(robot, LEFT_HAND)
    oMeffR = get_oMframe_by_id(robot, RIGHT_HAND)
    task = np.concatenate(
        [pin.log(oMeffL.inverse() * oMcubeL).vector,
         pin.log(oMeffR.inverse() * oMcubeR).vector]
    )
    if q_ref is None or w_pos == 0.0:
        return task
    return np.concatenate([task, np.sqrt(w_pos) * (q - q_ref)])


# for slsqp
def cost(q, robot, oMcubeL: pin.SE3, oMcubeR: pin.SE3, q_ref: Q = None, w_pos: float = W_POS) -> float:
    pin.framesForwardKinematics(robot.model, robot.data, q)
    oMeffL = get_oMframe_by_id(robot, LEFT_HAND)
    oMeffR = get_oMframe_by_id(robot, RIGHT_HAND)
    pos_bias = float(np.sum(w_pos * (q - q_ref) ** 2)) if q_ref is not None else 0
    return (
        norm(pin.log(oMeffL.inverse() * oMcubeL).vector) ** 2
        + norm(pin.log(oMeffR.inverse() * oMcubeR).vector) ** 2
        + pos_bias
    )


def callback(q: Q, robot, cube: pin.RobotWrapper, viz=None):
    if viz:
        from setup_meshcat import updatevisuals

        updatevisuals(viz, robot, cube, q)
        time.sleep(1e-1)


def solve_ls(robot, q_seed: Q, q_ref: Q, oMcubeL: pin.SE3, oMcubeR: pin.SE3, w_pos: float = W_POS) -> tuple[Q, bool]:
    lo, hi = robot.model.lowerPositionLimit, robot.model.upperPositionLimit

    result = least_squares(
        fun=lambda x: residual(x, robot, oMcubeL, oMcubeR, q_ref, w_pos),
        x0=np.clip(q_seed, lo, hi),
        bounds=(lo, hi),
    )
    q = result.x
    task_err = residual(q, robot, oMcubeL, oMcubeR) # residual without posture bias
    return q, float(task_err @ task_err) <= TASK_TOL


def solve_slsqp(robot, q_curr: Q, oMcubeL: pin.SE3, oMcubeR: pin.SE3, cube, viz=None, w_pos: float = W_POS) -> tuple[Q, bool]:
    lo, hi = robot.model.lowerPositionLimit, robot.model.upperPositionLimit

    constraints = [
        {"type": "ineq", "fun": lambda x: get_coll_distance(robot, x) - 1e-3},
        {"type": "ineq", "fun": lambda x: get_obstacle_distance(robot, x) - OBSTACLE_MARGIN},
    ]

    result = minimize(
        fun=lambda x: cost(x, robot, oMcubeL, oMcubeR, q_ref=q_curr, w_pos=w_pos),
        x0=q_curr,
        method="SLSQP",
        bounds=list(zip(lo, hi)),
        constraints=constraints,
        options={"maxiter": 300},
        callback=lambda x: callback(x, robot, cube, viz),
    )
    return result.x, bool(result.success and cost(result.x, robot, oMcubeL, oMcubeR) <= TASK_TOL)


def computeqgrasppose_ls(robot, qcurrent: Q, cube: pin.RobotWrapper, cubetarget: pin.SE3, viz=None) -> tuple[Q, bool]:
    """ Compute the grasp pose using least squares solver (faster but less robust). To be called during RRT"""
    setcubeplacement(robot, cube, cubetarget)

    oMcubeL = getcubeplacement(cube, LEFT_HOOK)  # placement of the left hand hook
    oMcubeR = getcubeplacement(cube, RIGHT_HOOK)  # placement of the right hand hook

    rng = np.random.default_rng()
    lo, hi = robot.model.lowerPositionLimit, robot.model.upperPositionLimit
    for attempt in range(MAX_TRIES):
        seed = np.clip(qcurrent + rng.normal(0.0, attempt * 0.03, size=qcurrent.shape), lo, hi)  # nearby posture
        q, ok = solve_ls(robot, seed, qcurrent, oMcubeL, oMcubeR, w_pos=W_POS)
        if ok and is_valid(robot, q):
            callback(q, robot, cube, viz)
            return q, True
    return q, False


def computeqgrasppose(robot, qcurrent: Q, cube: pin.RobotWrapper, cubetarget: pin.SE3, viz=None) -> tuple[Q, bool]:
    """Return a collision free configuration grasping a cube at a specific location and a success flag"""
    setcubeplacement(robot, cube, cubetarget)

    oMcubeL = getcubeplacement(cube, LEFT_HOOK)  # placement of the left hand hook
    oMcubeR = getcubeplacement(cube, RIGHT_HOOK)  # placement of the right hand hook

    q, ls_valid = computeqgrasppose_ls(robot, qcurrent, cube, cubetarget, viz)
    if ls_valid:
        return q, True

    q, ok = solve_slsqp(robot, qcurrent, oMcubeL, oMcubeR, cube, viz)
    return q, bool(ok and is_valid(robot, q))


if __name__ == "__main__":
    from tools import setupwithmeshcat
    from setup_meshcat import updatevisuals

    robot, cube, viz = setupwithmeshcat(url="tcp://127.0.0.1:6013")
    q = robot.q0.copy()

    q0, successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    updatevisuals(viz, robot, cube, q0)
    qe, successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET, viz)
    updatevisuals(viz, robot, cube, qe)
    print("done", successinit, successend)
