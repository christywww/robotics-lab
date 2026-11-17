#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 11:44:32 2023

@author: stonneau
"""
import random
import time
from typing import NamedTuple, Optional, TypeAlias
import math

import numpy as np
import pinocchio as pin
from numpy._typing import NDArray
from pinocchio.utils import rotate

from inverse_geometry import computeqgrasppose
from tools import collision, distanceToObstacle

STEP_SIZE = 0.02 # min distance for discretization step in new_conf
Q: TypeAlias = NDArray[np.float64]

class GraphNode(NamedTuple):
    parent: Optional[int]
    q: Q
    cube_placement: pin.SE3
    transition: list[Q]  # grasping configurations from the parent's q (excluded) to q (included)
    transition_placements: list[pin.SE3]  # cube placement for each q in transition


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
    return sum((i - j) ** 2 for i, j in zip(q1, q2)) ** 0.5


def new_conf(robot, cube, q_near, q_rand, cube_placement_qnear, cube_placement_qrand, discretisation_steps: int):
    '''discretisation_steps is now the MAXIMUM number of steps.'''
    dist = np.linalg.norm(cube_placement_qrand.translation - cube_placement_qnear.translation)
    n_steps = max(1, min(discretisation_steps, math.ceil(dist / STEP_SIZE)))

    transition, placements = [], []
    q_prev = q_near
    for i in range(1, n_steps + 1):
        cube_pos = lerp_se3(cube_placement_qnear, cube_placement_qrand, i / n_steps)
        q_curr, success = computeqgrasppose(robot, q_prev, cube, cube_pos)
        if success and not collision(robot, q_curr):
            transition.append(q_curr)
            placements.append(cube_pos)
        else:
            break
        q_prev = q_curr
    return transition, placements


def valid_edge(robot, cube, q_a: Q, q_b: Q, p_a: pin.SE3, p_b: pin.SE3, discretisation_steps: int) -> tuple[bool, list[Q], list[pin.SE3]]:
    transition, placements = new_conf(robot, cube, q_a, q_b, p_a, p_b, discretisation_steps)
    if not transition:
        return False, [], []
    reached = np.linalg.norm(pin.log6(placements[-1].inverse() * p_b).vector) < 1e-3
    return reached, transition, placements


def nearest_vertex(G: list[GraphNode], q_rand: Q) -> int:
    """returns the index of the Node of G with the configuration closest to q_rand"""
    min_distance, min_node = float("inf"), None
    for idx, node in enumerate(G):
        d = distance(q_rand, node.q)
        if d < min_distance:
            min_distance, min_node = d, idx
    return min_node


def add_edge_and_vertex(G: list[GraphNode], parent: int, transition: list[Q], placements: list[pin.SE3]):
    G.append(GraphNode(parent, transition[-1], placements[-1], transition, placements))


def getpath(G: list[GraphNode], idx: int = -1) -> tuple[list[Q], list[pin.SE3]]:
    qs, placements = [], []
    node = G[idx]
    while node.parent is not None:
        qs = node.transition + qs
        placements = node.transition_placements + placements
        node = G[node.parent]
    return [node.q] + qs, [node.cube_placement] + placements


def extend(robot, cube, G: list[GraphNode], q_target: Q, cube_target: pin.SE3, steps: int) -> tuple[Optional[int], bool]:
    '''Grow G from its nearest node toward the target.
    Returns (index of new node or None, whether the target was reached).'''
    i_near = nearest_vertex(G, q_target)
    near = G[i_near]
    transition, placements = new_conf(robot, cube, near.q, q_target, near.cube_placement, cube_target, steps)
    if not transition:
        return None, False
    add_edge_and_vertex(G, i_near, transition, placements)
    reached = np.linalg.norm(pin.log6(placements[-1].inverse() * cube_target).vector) < 1e-3
    return len(G) - 1, reached


def shortcut(robot, cube, path: list[Q], placements: list[pin.SE3], iterations: int = 50, steps: int = 10) -> list[Q]:
    path, placements = list(path), list(placements)
    for _ in range(iterations):
        if len(path) < 3:
            break
        i = random.randint(0, len(path) - 3)
        j = random.randint(i + 2, len(path) - 1)
        reached, transition, new_pl = valid_edge(robot, cube, path[i], path[j], placements[i], placements[j], steps)
        if reached:
            # keep path[j] itself so the path stays continuous with what follows
            path = path[:i + 1] + transition[:-1] + path[j:]
            placements = placements[:i + 1] + new_pl[:-1] + placements[j:]
    return path


# returns a collision free path from qinit to qgoal under grasping constraints
# the path is expressed as a list of configurations
def computepath(robot, cube, q_init: Q, q_goal: Q, cube_placement_q0: pin.SE3, cube_placement_qgoal: pin.SE3, k: int = 1000, discretisation_steps: int = 10) -> list[Q]:
    G_a = [GraphNode(None, q_init, cube_placement_q0, [q_init], [cube_placement_q0])]  # starts at the start
    G_b = [GraphNode(None, q_goal, cube_placement_qgoal, [q_goal], [cube_placement_qgoal])]  # starts at the goal
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
                    (sq, sp), (gq, gp) = getpath(G_a, idx_a), getpath(G_b, idx_b)
                else:
                    (sq, sp), (gq, gp) = getpath(G_b, idx_b), getpath(G_a, idx_a)
                return shortcut(robot, cube, sq + gq[::-1], sp + gp[::-1], steps=discretisation_steps)

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
    qe, successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET, viz)

    if not (successinit and successend):
        print("error: invalid initial or end configuration")

    path = computepath(robot, cube, q0, qe, CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)
    displaypath(robot, path, dt=0.05, viz=viz)
