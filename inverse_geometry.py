#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau
"""
import time
from functools import partial
from typing import TypeAlias

import numpy as np
import pinocchio as pin
from numpy._typing import NDArray
from numpy.linalg import norm
from scipy.optimize import minimize

from tools import getcubeplacement
from config import LEFT_HOOK, RIGHT_HOOK, LEFT_HAND, RIGHT_HAND
from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET

from tools import setcubeplacement

Q: TypeAlias = NDArray[np.float64]


def get_oMframe_by_id(robot, frame: str) -> pin.SE3:
    frameid = robot.model.getFrameId(frame)
    return robot.data.oMf[frameid]


def get_coll_distance(robot, q: Q) -> float:
    pin.updateGeometryPlacements(robot.model, robot.data, robot.collision_model, robot.collision_data, q)
    if pin.computeCollisions(robot.collision_model, robot.collision_data, False):
        return 0.0
    idx = pin.computeDistances(robot.collision_model, robot.collision_data)
    return robot.collision_data.distanceResults[idx].min_distance


def cost(q, robot, oMcubeL: pin.SE3, oMcubeR: pin.SE3) -> float:
    pin.framesForwardKinematics(robot.model, robot.data, q)
    oMeffL = get_oMframe_by_id(robot, LEFT_HAND)
    oMeffR = get_oMframe_by_id(robot, RIGHT_HAND)
    return (
        norm(pin.log(oMeffL.inverse() * oMcubeL).vector) ** 2
        + norm(pin.log(oMeffR.inverse() * oMcubeR).vector) ** 2
    )


def callback(q: Q, robot, cube: pin.SE3, viz=None):
    if viz:
        from setup_meshcat import updatevisuals

        updatevisuals(viz, robot, cube, q)
        time.sleep(1e-1)


def computeqgrasppose(robot, qcurrent: Q, cube: pin.SE3, cubetarget: pin.SE3, viz=None) -> tuple[Q, bool]:
    """Return a collision free configuration grasping a cube at a specific location and a success flag"""
    setcubeplacement(robot, cube, cubetarget)

    joint_limits = [(q_min, q_max) for q_min, q_max in zip(robot.model.lowerPositionLimit, robot.model.upperPositionLimit)]
    oMcubeL = getcubeplacement(cube, LEFT_HOOK)  # placement of the left hand hook
    oMcubeR = getcubeplacement(cube, RIGHT_HOOK)  # placement of the right hand hook

    pin.framesForwardKinematics(robot.model, robot.data, qcurrent)
    result = minimize(
        fun=partial(cost, robot=robot, oMcubeL=oMcubeL, oMcubeR=oMcubeR),
        x0=qcurrent,
        method="SLSQP",
        bounds=joint_limits,
        constraints={"type": "ineq", "fun": lambda q: get_coll_distance(robot, q) - 1e-3},
        callback=partial(callback, robot=robot, cube=cube, viz=viz),
    )
    success = result.success and result.fun <= 1e-3
    return result.x, success


if __name__ == "__main__":
    from tools import setupwithmeshcat
    from setup_meshcat import updatevisuals

    robot, cube, viz = setupwithmeshcat(url="tcp://127.0.0.1:6013")
    q = robot.q0.copy()

    q0, successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    qe, successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET, viz)
    updatevisuals(viz, robot, cube, qe)
    print("done")
