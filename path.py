#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 11:44:32 2023

@author: stonneau
"""
from typing import NamedTuple, Optional, TypeAlias

import pinocchio as pin
import numpy as np
from numpy._typing import NDArray
from pinocchio.utils import rotate
import time
from tools import collision
from inverse_geometry import computeqgrasppose

Q: TypeAlias = NDArray[np.float64]

class GraphNode(NamedTuple):
    parent: Optional[int]
    q: Q
    cube_placement: pin.SE3
    transition: list[Q]  # grasping configurations from the parent's q (excluded) to q (included)


def rand_conf(robot, cube: pin.RobotWrapper) -> tuple[Q, pin.SE3]:
    success = False
    while not success:
        cube_pos = pin.SE3(rotate("z", 0.0), np.random.uniform(low=[0.3, -0.3, 0.93], high=[0.5, 0.15, 1.4], size=3))
        q, success = computeqgrasppose(robot, robot.q0, cube, cube_pos)
    return q, cube_pos

def lerp(q1: Q, q2: Q, t: float) -> Q:
    return q1 * (1 - t) + q2 * t

def lerp_se3(p1: pin.SE3, p2: pin.SE3, t: float) -> pin.SE3:
    rot = p1.rotation @ pin.exp3(t * pin.log3(p1.rotation.T @ p2.rotation))
    return pin.SE3(rot, lerp(p1.translation, p2.translation, t))

def distance(q1: Q, q2: Q) -> float:
    '''Return the euclidian distance between two configurations'''
    return sum((i-j)**2 for i, j in zip(q1, q2))**0.5

def new_conf(robot, cube, q_near: Q, q_rand: Q, cube_placement_qnear: pin.SE3, cube_placement_qrand: pin.SE3, discretisation_steps: int) -> tuple[list[Q], pin.SE3]:
    '''Return the list of configurations such that the path q_near => q_new is the longest
    along the linear interpolation (q_near,q_rand) that is collision free'''
    transition = []
    q_prev, cube_placement_last = q_near, cube_placement_qnear
    dt = 1 / discretisation_steps
    for i in range(1,discretisation_steps + 1):
        cube_pos = lerp_se3(cube_placement_qnear, cube_placement_qrand, dt*i)
        q_curr, success = computeqgrasppose(robot, q_prev, cube, cube_pos)
        if success and not collision(robot, q_curr):
            transition.append(q_curr)
        else:
            break
        q_prev, cube_placement_last = q_curr, cube_pos
    return transition, cube_placement_last

def valid_edge(robot, cube, q_new: Q, q_goal: Q, cube_placement_qnear: pin.SE3, cube_placement_qgoal: pin.SE3, discretisation_steps: int) -> tuple[bool, list[Q]]:
    transition, cube_placement_last = new_conf(robot, cube, q_new, q_goal, cube_placement_qnear, cube_placement_qgoal, discretisation_steps)
    reached = np.linalg.norm(pin.log6(cube_placement_last.inverse() * cube_placement_qgoal).vector) < 1e-6
    return reached, transition

def nearest_vertex(G: list[GraphNode], q_rand: Q) -> int:
    """returns the index of the Node of G with the configuration closest to q_rand"""
    min_distance, min_node = 999999, None
    for idx, node in enumerate(G):
        if distance(q_rand, node.q) < min_distance:
            min_distance = distance(q_rand, node.q)
            min_node = idx
    return min_node


def add_edge_and_vertex(G: list[GraphNode], parent: int, transition: list[Q], cube_placement: pin.SE3):
    G += [GraphNode(parent, q=transition[-1], cube_placement=cube_placement, transition=transition)]

def getpath(G: list[GraphNode], idx: int = -1) -> list[Q]:
    path = []
    node = G[idx]
    while node.parent is not None:
        path = node.transition + path
        node = G[node.parent]
    return [node.q] + path

def extend(robot, cube, G: list[GraphNode], q_target: Q, cube_target: pin.SE3, steps: int) -> tuple[Optional[int], bool]:
    '''Grow G from its nearest node toward the target.
    Returns (index of new node or None, whether the target was reached).'''
    i_near = nearest_vertex(G, q_target)
    near = G[i_near]
    transition, cube_last = new_conf(robot, cube, near.q, q_target, near.cube_placement, cube_target, steps)
    if not transition:
        return None, False
    add_edge_and_vertex(G, i_near, transition, cube_last)
    reached = np.linalg.norm(pin.log6(cube_last.inverse() * cube_target).vector) < 1e-3
    return len(G) - 1, reached

# returns a collision free path from qinit to qgoal under grasping constraints
# the path is expressed as a list of configurations
def computepath(robot, cube, q_init: Q, q_goal: Q, cube_placement_q0: pin.SE3, cube_placement_qgoal: pin.SE3, k: int = 1000, discretisation_steps: int = 10) -> list[Q]:
    G_a = [GraphNode(None, q_init, cube_placement_q0, [q_init])]      # starts at the start
    G_b = [GraphNode(None, q_goal, cube_placement_qgoal, [q_goal])]   # starts at the goal
    a_is_start = True

    for i in range(k):
        if i % 10 == 0:
            print(f"running rrt-connect iteration {i}...")

        q_rand, cube_rand = rand_conf(robot, cube)
        idx_a, _ = extend(robot, cube, G_a, q_rand, cube_rand, discretisation_steps)

        if idx_a is not None:
            na = G_a[idx_a]
            # try to connect the other tree to the node we just added
            idx_b, reached = extend(robot, cube, G_b, na.q, na.cube_placement, discretisation_steps)
            if reached:
                print(f"Path found after {i} iterations!")
                if a_is_start:
                    start_path, goal_path = getpath(G_a, idx_a), getpath(G_b, idx_b)
                else:
                    start_path, goal_path = getpath(G_b, idx_b), getpath(G_a, idx_a)
                return start_path + goal_path[::-1]

        G_a, G_b = G_b, G_a
        a_is_start = not a_is_start

    return []


def displaypath(robot, path, dt, viz):
    for q in path:
        viz.display(q)
        time.sleep(dt)


if __name__ == "__main__":
    from tools import setupwithmeshcat
    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET

    robot, cube, viz = setupwithmeshcat(url="tcp://127.0.0.1:6013")

    q = robot.q0.copy()
    q0, successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    qe, successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET,  viz)

    if not(successinit and successend):
        print ("error: invalid initial or end configuration")

    path = computepath(robot, cube, q0, qe, CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)
    displaypath(robot, path,dt=0.05,viz=viz)
