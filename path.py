#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 11:44:32 2023

@author: stonneau
"""
from collections import namedtuple
from typing import TypeAlias

import pinocchio as pin
import numpy as np
from numpy._typing import NDArray
from pinocchio.utils import rotate
import time

Q: TypeAlias = NDArray[np.float64]
GraphNode = namedtuple('GraphNode', ['parent', 'q', 'cube_placement'])


def rand_conf() -> tuple[Q, pin.SE3]:
    success = False
    while not success:
        cube_pos = pin.SE3(rotate("z", 0.0),np.random.uniform(low=[0.25, -0.5, 0.93], high=[0.5, 0.3, 1.93], size=3))
        q, success = computeqgrasppose(robot, robot.q0, cube, cube_pos)
    return q, cube_pos

def lerp(q1: Q, q2: Q, t: float) -> Q:
    return pin.interpolate(robot.model, q1, q2, t)

def distance(q1: Q, q2: Q) -> float:
    '''Return the euclidian distance between two configurations'''
    return sum((i-j)**2 for i, j in zip(q1, q2))**0.5

def new_conf(q_near: Q, q_rand: Q, cube_placement_qnear: pin.SE3, cube_placement_qrand: pin.SE3, discretisation_steps: int, delta_q: float = None) -> Q:
    '''Return the closest configuration q_new such that the path q_near => q_new is the longest
    along the linear interpolation (q_near,q_rand) that is collision free and of length <  delta_q'''
    q_end = q_rand.copy()
    dist = distance(q_near, q_rand)
    if delta_q is not None and dist > delta_q:
        # compute the configuration that corresponds to a path of length delta_q
        q_end = lerp(q_near,q_rand,delta_q/dist)
        cube_placement = lerp(cube_placement_qnear, cube_placement_qrand,delta_q/dist)
        # find closest grasp pose to q_end
        q_end = computeqgrasppose(robot, q_end, cube, cube_placement)
        # now dist == delta_q
    dt = 1 / discretisation_steps
    for i in range(1,discretisation_steps):
        q = lerp(q_near,q_end,dt*i)
        if not collision(robot, q):
            return lerp(q_near,q_end,dt*(i-1))
    return q_end

def valid_edge(q_new: Q, q_goal: Q, discretisation_steps: int) -> bool:
    return np.linalg.norm(q_goal - new_conf(q_new, q_goal, None, None, discretisation_steps)) < 1e-3

def nearest_vertex(G: list[GraphNode], q_rand: Q) -> int:
    """returns the index of the Node of G with the configuration closest to q_rand"""
    min_distance, min_node = 999999, None
    for idx, node in enumerate(G):
        if distance(q_rand, node.q) < min_distance:
            min_distance = distance(q_rand, node.q)
            min_node = idx
    return min_node


def add_edge_and_vertex(G: list[GraphNode], parent: int, q: Q, cube_placement: pin.SE3):
    G += [GraphNode(parent, q, cube_placement)]

def getpath(G: list[GraphNode]) -> list[Q]:
    path = []
    node = G[-1]
    while node.parent is not None:
        path = [node.q] + path
        node = G[node.parent]
    path = [G[0].q] + path
    return path

# returns a collision free path from qinit to qgoal under grasping constraints
# the path is expressed as a list of configurations
def computepath(q_init: Q, q_goal: Q, cube_placement_q0: pin.SE3, cube_placement_qgoal: pin.SE3, k: int=1000, discretisation_steps: int=200) -> list[Q]:
    '''Return a collision free path from qinit to qgoal under grasping constraints'''
    G = [GraphNode(None, q_init, cube_placement_q0)]
    success = False
    for i in range(k):
        q_rand, cube_placement_qrand = rand_conf()
        q_near_index = nearest_vertex(G, q_rand)
        _, q_near, cube_placement_qnear = G[q_near_index]
        q_new = new_conf(q_near, q_rand, cube_placement_qnear, cube_placement_qrand, discretisation_steps)
        add_edge_and_vertex(G, q_near_index, q_new, cube_placement_qrand)
        if valid_edge(q_new, q_goal, discretisation_steps):
            print(f"Path found after {i} iterations!")
            add_edge_and_vertex(G, len(G) - 1, q_goal, cube_placement_qgoal)
            success = True
            break
    return getpath(G) if success else []


def displaypath(robot, path, dt, viz):
    for q in path:
        viz.display(q)
        time.sleep(dt)


if __name__ == "__main__":
    from tools import setupwithmeshcat, collision
    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
    from inverse_geometry import computeqgrasppose

    robot, cube, viz = setupwithmeshcat(url="tcp://127.0.0.1:6013")

    q = robot.q0.copy()
    q0, successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    qe, successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET,  viz)
    
    if not(successinit and successend):
        print ("error: invalid initial or end configuration")
    
    path = computepath(q0, qe, CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)
    displaypath(robot,path,dt=0.5,viz=viz) #you'll probably want to lower dt
    
