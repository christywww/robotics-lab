#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau
"""

import numpy as np

from bezier import Bezier

import pinocchio as pin

from config import LEFT_HAND, RIGHT_HAND

from scipy.special import comb


from tools import collision

# in my solution these gains were good enough for all joints but you might want to tune this.
Kp = 300  # 300.               # proportional gain (P of PD)
Kv = 2 * np.sqrt(Kp)  # 2 * np.sqrt(Kp)   # derivative gain (D of PD)


def cumulative_distances(path):
    dists = [0.]
    for i in range(1, len(path)):
        dists.append(dists[-1] + np.linalg.norm(path[i] - path[i - 1]))
    return np.array(dists) / dists[-1]

def bezier_basis(degree, u_values):
    """Bernstein matrix: row k = weights of the control points at u_values[k] (u in [0, 1])."""
    u = np.asarray(u_values, dtype=float)[:, None]      # (N, 1)
    i = np.arange(degree + 1)[None, :]                  # (1, n+1)
    return comb(degree, i) * u**i * (1 - u)**(degree - i)

import quadprog

def quadprog_solve_qp(H, q, G=None, h=None, C=None, d=None):
    """min 0.5 x'Hx + q'x   s.t.  G x <= h,  C x = d
    (from the lab's convex optimisation tutorial, notebook 6)"""
    qp_G = 0.5 * (H + H.T)
    qp_a = -q
    qp_C = qp_b = None
    meq = 0
    if C is not None:
        if G is not None:
            qp_C, qp_b = -np.vstack([C, G]).T, -np.hstack([d, h])
        else:
            qp_C, qp_b = -C.T, -d
        meq = C.shape[0]
    elif G is not None:
        qp_C, qp_b = -G.T, -h
    return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]

def optimise_control_points(path, u, n, T, qmin, qmax, vmax, w_acc=1e-3):
    """Fit a degree-n Bezier to the path, one QP per joint (matrices shared).
    Cost: follow the path + small penalty on acceleration.
    Equalities: exact start/end, zero velocity at both ends.
    Inequalities: |velocity| <= vmax, qmin <= free control points <= qmax
    (these hold along the whole curve by the convex hull property)."""
    path = np.array(path)
    dim = path.shape[1]
    B = bezier_basis(n, u)                                 # (len(path), n+1)
    I = np.eye(n + 1)
    D1 = np.diff(I, 1, axis=0) * n / T                     # velocity control points
    D2 = np.diff(I, 2, axis=0) * n * (n - 1) / T**2        # acceleration control points

    H = 2 * (B.T @ B + w_acc * D2.T @ D2) + 1e-8 * I
    C = I[[0, 1, n - 1, n]]                                # P0, P1, Pn-1, Pn fixed
    free = I[2:n - 1]                                      # control points 2 .. n-2
    G = np.vstack([D1, -D1, free, -free])

    P = np.zeros((n + 1, dim))
    for j in range(dim):
        ref = path[:, j]
        q = -2 * B.T @ ref
        d = np.array([ref[0], ref[0], ref[-1], ref[-1]])
        h = np.r_[np.full(2 * n, vmax[j]),
                  np.full(n - 3, qmax[j]), np.full(n - 3, -qmin[j])]
        P[:, j] = quadprog_solve_qp(H, q, G, h, C, d)
    return P

def squeeze_torques(robot, q):
    pin.computeJointJacobians(robot.model, robot.data, q)
    pin.framesForwardKinematics(robot.model, robot.data, q)

    idL = robot.model.getFrameId(LEFT_HAND)
    idR = robot.model.getFrameId(RIGHT_HAND)
    pL = robot.data.oMf[idL].translation
    pR = robot.data.oMf[idR].translation
    axis = (pR - pL) / np.linalg.norm(pR - pL)  # unit vector, left hand -> right hand

    tau = np.zeros(robot.model.nv)
    for fid, sign in ((idL, +1), (idR, -1)):
        J = pin.getFrameJacobian(robot.model, robot.data, fid, pin.LOCAL_WORLD_ALIGNED)
        f_lin = sign * FORCE * axis
        tau += J.T @ np.r_[f_lin, 0., 0., 0.]
    return tau

def controllaw(sim, robot, trajs, tcurrent, cube):
    q, vq = sim.getpybulletstate()

    q_ref = trajs[0](tcurrent)
    vq_ref = trajs[1](tcurrent)
    aq_ref = trajs[2](tcurrent)

    aq = (aq_ref + Kp * (q_ref - q) + Kv * (vq_ref - vq))

    torques = pin.rnea(robot.model, robot.data, q, vq, aq) + squeeze_torques(robot, q)
    sim.step(torques)





if __name__ == "__main__":

    from tools import setupwithpybullet, setupwithpybulletandmeshcat, rununtil
    from config import DT
    import pybullet as pyb

    robot, sim, cube = setupwithpybullet()
    for l in (11, 18):
        pyb.changeDynamics(sim.robot, l, mass=1e-3, localInertiaDiagonal=[1e-7] * 3)
    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
    from inverse_geometry import computeqgrasppose
    from path import computepath


    q0, successinit = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT, None)
    qe, successend = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT_TARGET, None)
    path = computepath(robot, cube, q0,qe,CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)

    FORCE = 300.

    # setting initial configuration
    sim.setqsim(path[0])
    def maketraj(path, T):
        q_of_t = Bezier(path, t_max=T)


        vq_of_t = q_of_t.derivative(1)
        vvq_of_t = vq_of_t.derivative(1)

        return q_of_t, vq_of_t, vvq_of_t


    total_time = 20.
    u = cumulative_distances(path)
    n = 12
    control_points = optimise_control_points(
        path, u, n, total_time,
        robot.model.lowerPositionLimit, robot.model.upperPositionLimit,
        vmax=np.full(15, 1.0))

    print(control_points)

    trajs = maketraj(control_points, total_time)
    tcur = 0.



    while tcur < total_time:
        rununtil(controllaw, DT, sim, robot, trajs, tcur, cube)
        tcur += DT
