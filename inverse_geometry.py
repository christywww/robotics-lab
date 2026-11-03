#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau
"""
import time
from functools import partial

import pinocchio as pin
import numpy as np
from numpy.linalg import pinv, inv, norm, svd, eig
from scipy.optimize import fmin_bfgs, minimize

from tools import collision, getcubeplacement, setcubeplacement, projecttojointlimits
from config import LEFT_HOOK, RIGHT_HOOK, LEFT_HAND, RIGHT_HAND, EPSILON
from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET

from tools import setcubeplacement


def get_oMframe_by_id(robot, frame: str):
    frameid = robot.model.getFrameId(frame)
    return robot.data.oMf[frameid]


def get_coll_distance(robot, q):
    pin.updateGeometryPlacements(robot.model, robot.data, robot.collision_model, robot.collision_data, q)
    if pin.computeCollisions(robot.collision_model, robot.collision_data, False): return 0
    idx = pin.computeDistances(robot.collision_model, robot.collision_data)
    return robot.collision_data.distanceResults[idx].min_distance


def cost(q, oMcubeL, oMcubeR):
    pin.framesForwardKinematics(robot.model, robot.data, q)
    oMeffL = get_oMframe_by_id(robot, LEFT_HAND)
    oMeffR = get_oMframe_by_id(robot, RIGHT_HAND)
    return (
        norm(pin.log(oMeffL.inverse() * oMcubeL).vector)
        + norm(pin.log(oMeffR.inverse() * oMcubeR).vector)
    )


def callback(q, viz=None):
    # setcubeplacement(robot, cube, CUBE_PLACEMENT)
    updatevisuals(viz, robot, cube, q)
    time.sleep(1e-1)


def computeqgrasppose(robot, qcurrent, cube, cubetarget, viz=None):
    '''Return a collision free configuration grasping a cube at a specific location and a success flag'''
    setcubeplacement(robot, cube, cubetarget)

    joint_limits = [(q_min, q_max) for q_min, q_max in zip(robot.model.lowerPositionLimit, robot.model.upperPositionLimit)]
    oMcubeL = getcubeplacement(cube, LEFT_HOOK)  # placement of the left hand hook
    oMcubeR = getcubeplacement(cube, RIGHT_HOOK)  # placement of the right hand hook

    pin.framesForwardKinematics(robot.model, robot.data, qcurrent)
    result = minimize(
        fun=partial(cost, oMcubeL=oMcubeL, oMcubeR=oMcubeR),
        x0=qcurrent,
        method="SLSQP",
        bounds=joint_limits,
        constraints={"type": "ineq", "fun": lambda q: get_coll_distance(robot, q) - 1e-3},
        callback=partial(callback, viz=viz),
    )
    return result.x, result.success


if __name__ == "__main__":
    from tools import setupwithmeshcat
    from setup_meshcat import updatevisuals

    robot, cube, viz = setupwithmeshcat(url="tcp://127.0.0.1:6013")
    q = robot.q0.copy()

    q0, successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    qe, successend = computeqgrasppose(robot, q0, cube, CUBE_PLACEMENT_TARGET, viz)
    updatevisuals(viz, robot, cube, qe)
    print("done")
