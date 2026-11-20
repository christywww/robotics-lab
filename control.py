#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau

References
----------
[1] B. Siciliano, L. Sciavicco, L. Villani, G. Oriolo, "Robotics: Modelling, Planning and Control",
    Springer, 2009.
[2] L. Biagiotti, C. Melchiorri, "Trajectory Planning for Automatic Machines and Robots",
    Springer, 2008.
[3] D. Goldfarb, A. Idnani, "A numerically stable dual method for solving strictly convex quadratic
    programs", Mathematical Programming 27, 1983.
[4] R. Featherstone, "Rigid Body Dynamics Algorithms", Springer, 2008.
[5] J. Carpentier et al., "The Pinocchio C++ library", IEEE/SICE SII, 2019.
[6] E. Coumans, Y. Bai, "PyBullet Quickstart Guide".
[7] Course Tutorial 6.
"""

import numpy as np
import pinocchio as pin
import quadprog
from scipy.special import comb

from bezier import Bezier
from config import LEFT_HAND, RIGHT_HAND, DT

# joint PID gains (Kv = 2*sqrt(Kp): critically damped) [1]
Kp = 300.
Kv = 2 * np.sqrt(Kp)
Ki = 600.

# grip force feedback
FORCE = 20        # initial squeeze command (N)
F_TARGET = 100.   # target contact force per hand (N)
K_F = 20.         # force integral gain (1/s)
grip = [FORCE]    # current squeeze command

integ = np.zeros(15)   # integral of the joint tracking error


def path_divisions(path, samples):
    """Chunks of `samples` points; consecutive chunks share an endpoint."""
    divisions = []
    last = len(path) - 1
    start = 0
    while start < last:
        end = min(start + samples - 1, last)
        if last - end < 2:          # don't leave a 1-interval tail
            end = last
        divisions.append(path[start:end + 1])
        start = end
    return divisions


def densify(path, max_step=0.1):
    """Insert points along each straight edge so no step exceeds max_step."""
    out = [np.asarray(path[0])]
    for a, b in zip(path[:-1], path[1:]):
        a, b = np.asarray(a), np.asarray(b)
        m = max(1, int(np.ceil(np.linalg.norm(b - a) / max_step)))
        for s in range(1, m + 1):
            out.append(a + (b - a) * s / m)
    return out


def smooth_path(path, window=3):
    """Moving average over the path samples (endpoints stay exact)."""
    p = np.array(path)
    padded = np.pad(p, ((window // 2, window // 2), (0, 0)), mode="edge")
    kernel = np.ones(window) / window
    out = np.stack([np.convolve(padded[:, j], kernel, mode="valid") for j in range(p.shape[1])]).T
    out[0], out[-1] = p[0], p[-1]
    return list(out)


def cumulative_distances(path):
    """Normalised arc length (0 to 1) at each sample (chord-length parameterisation [2])."""
    dists = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    if dists[-1] == 0:
        return np.linspace(0, 1, len(path))
    return dists / dists[-1]


def bezier_basis(degree, u_values):
    """Bernstein basis matrix [2]."""
    u = np.asarray(u_values, dtype=float)[:, None]      # (N, 1)
    i = np.arange(degree + 1)[None, :]                  # (1, n+1)
    return comb(degree, i) * u**i * (1 - u)**(degree - i)


def quadprog_solve_qp(H, q, G=None, h=None, C=None, d=None):
    """From Tutorial 6 [7], using quadprog [3]."""
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


def optimise_control_points(path_divided, n, T_base, qmin, qmax, vmax, w_acc=1e-6, slow=1.7):
    """Piecewise Bezier fit of degree n (n >= 6), one QP per joint and piece [2].

    Cost: least-squares fit to the path + acceleration penalty. Joint limits on the free control points.
    Each piece is C2-continuous with the previous one; the trajectory starts and ends at rest.
    Piece duration comes from vmax and is increased if the QP is infeasible."""
    control_points_list, T_list = [], []
    I = np.eye(n + 1)
    prev_P, prev_T = None, None
    last = len(path_divided) - 1

    for k, path in enumerate(path_divided):
        is_last = (k == last)
        path = np.array(path)
        B = bezier_basis(n, cumulative_distances(path))

        fixed = [0, 1, 2, n - 1, n] if is_last else [0, 1, 2, n]
        free = I[3:n - 1] if is_last else I[3:n]
        n_free = free.shape[0]
        C = I[fixed]
        G = np.vstack([free, -free])

        def solve_piece(T):
            D2 = np.diff(I, 2, axis=0) * n * (n - 1) / T**2
            H = 2 * (B.T @ B + w_acc * D2.T @ D2) + 1e-8 * I
            P = np.zeros((n + 1, path.shape[1]))
            for j in range(path.shape[1]):
                ref = path[:, j]
                if prev_P is None:                       # first piece: start at rest
                    p0 = p1 = p2 = ref[0]
                else:                                    # inherit position, velocity, acceleration
                    p0 = prev_P[n, j]
                    p1 = p0 + (prev_P[n, j] - prev_P[n - 1, j]) * T / prev_T
                    acc_prev = prev_P[n, j] - 2 * prev_P[n - 1, j] + prev_P[n - 2, j]
                    p2 = 2 * p1 - p0 + (T / prev_T) ** 2 * acc_prev
                d = np.array([p0, p1, p2] + [ref[-1]] * (2 if is_last else 1))
                h = np.r_[np.full(n_free, qmax[j]), np.full(n_free, -qmin[j])]
                P[:, j] = quadprog_solve_qp(H, -2 * B.T @ ref, G, h, C, d)
            return P

        seg = np.abs(np.diff(path, axis=0)).sum(axis=0)
        T = max(T_base, slow * np.max(seg / vmax))
        for _ in range(12):
            try:
                P = solve_piece(T)
                break
            except ValueError:
                T *= 1.15                                # QP infeasible: slow down and retry
        else:
            raise RuntimeError(f"piece {k} infeasible even at T={T:.2f}")

        T_list.append(T)
        control_points_list.append(P)
        prev_P, prev_T = P, T
    return control_points_list, T_list


def maketraj(control_points_list, T_list):
    trajs = []
    for P, T in zip(control_points_list, T_list):
        q_of_t = Bezier(P, t_max=T)
        vq_of_t = q_of_t.derivative(1)
        trajs.append([q_of_t, vq_of_t, vq_of_t.derivative(1)])
    return trajs


def squeeze_torques(robot, q, CUBE_MASS, force):
    """Grip + lift forces at the hands mapped to joint torques with tau = J^T f [1, 5]."""
    pin.computeJointJacobians(robot.model, robot.data, q)
    pin.framesForwardKinematics(robot.model, robot.data, q)

    idL = robot.model.getFrameId(LEFT_HAND)
    idR = robot.model.getFrameId(RIGHT_HAND)
    pL = robot.data.oMf[idL].translation
    pR = robot.data.oMf[idR].translation
    axis = (pR - pL) / np.linalg.norm(pR - pL)  # unit vector, left hand -> right hand
    lift = np.array([0., 0., CUBE_MASS * 9.81 / 2])

    tau = np.zeros(robot.model.nv)
    for fid, sign in ((idL, +1), (idR, -1)):
        J = pin.getFrameJacobian(robot.model, robot.data, fid, pin.LOCAL_WORLD_ALIGNED)
        tau += J[:3].T @ (sign * force * axis + lift)
    return tau


def controllaw(sim, robot, trajs, end_times, tcurrent, CUBE_MASS):
    q, vq = sim.getpybulletstate()

    # active piece and local time (holds the final pose after the last piece)
    idx = min(int(np.searchsorted(end_times, tcurrent, side="right")), len(trajs) - 1)
    start = 0. if idx == 0 else end_times[idx - 1]
    q_of_t, vq_of_t, aq_of_t = trajs[idx]
    t_local = min(tcurrent - start, q_of_t.T_max_)

    q_ref, vq_ref, aq_ref = q_of_t(t_local), vq_of_t(t_local), aq_of_t(t_local)

    # computed-torque control: PID + feedforward acceleration [1]
    integ[:] = np.clip(integ + (q_ref - q) * DT, -0.5, 0.5)   # clip prevents windup
    aq = aq_ref + Kp * (q_ref - q) + Kv * (vq_ref - vq) + Ki * integ

    # force feedback: measured contact force on the cube [6], integral control on the squeeze [1]
    robot_id = max(range(pyb.getNumBodies()), key=pyb.getNumJoints)
    measured = sum(c[9] for c in pyb.getContactPoints(bodyA=sim.cubeId, bodyB=robot_id)) / 2
    grip[0] = np.clip(grip[0] + K_F * (F_TARGET - measured) * DT, 0., 400.)

    # inverse dynamics (RNEA) [4] + squeeze torques
    torques = pin.rnea(robot.model, robot.data, q, vq, aq) + squeeze_torques(robot, q, CUBE_MASS, grip[0])
    sim.step(torques)


if __name__ == "__main__":

    from tools import setupwithpybullet, rununtil
    import pybullet as pyb
    robot, sim, cube = setupwithpybullet()

    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
    from inverse_geometry import computeqgrasppose
    from path import computepath

    N = 6               # Bezier degree (C2 needs at least 6)
    SAMPLES = 6         # path samples per piece
    T_BASE = 0.4        # minimum duration of a piece (s)
    VMAX = 2.0          # speed limit (rad/s) used for timing
    HOLD = 2.0          # extra time holding the final pose (s)

    q0, _ = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT, None)
    qe, _ = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT_TARGET, None)

    path = computepath(robot, cube, q0, qe, CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)
    path = smooth_path(densify(path, max_step=0.1))

    sim.setqsim(path[0])
    CUBE_MASS = pyb.getDynamicsInfo(sim.cubeId, -1)[0]

    control_points, T_list = optimise_control_points(
        path_divisions(path, SAMPLES), N, T_BASE,
        robot.model.lowerPositionLimit, robot.model.upperPositionLimit,
        vmax=np.full(15, VMAX))

    trajs = maketraj(control_points, T_list)
    end_times = np.cumsum(T_list)

    input("Control Trajectory Ready!! Press Enter to start the simulation...")
    tcur = 0.
    while tcur < end_times[-1] + HOLD:
        rununtil(controllaw, DT, sim, robot, trajs, end_times, tcur, CUBE_MASS)
        tcur += DT
