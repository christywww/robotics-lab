#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau

Control part of the lab: takes the collision-free path from path.py, turns it into a smooth
time-parameterised trajectory (piecewise Bezier curves fitted with a QP), and then tracks it in
PyBullet with an inverse-dynamics (computed torque) controller. On top of that the hands squeeze
the cube, and the squeeze force is adjusted using force feedback read from the simulator.

References
----------
[1] B. Siciliano, L. Sciavicco, L. Villani, G. Oriolo, "Robotics: Modelling, Planning and Control",
    Springer, 2009. (statics / Jacobian transpose, motion control, force control chapters)
[2] L. Biagiotti, C. Melchiorri, "Trajectory Planning for Automatic Machines and Robots",
    Springer, 2008. (Bezier / spline trajectories)
[3] D. Goldfarb, A. Idnani, "A numerically stable dual method for solving strictly convex quadratic
    programs", Mathematical Programming 27, 1983. (the algorithm used by the quadprog package)
[4] R. Featherstone, "Rigid Body Dynamics Algorithms", Springer, 2008. (recursive Newton-Euler)
[5] J. Carpentier et al., "The Pinocchio C++ library", IEEE/SICE SII, 2019.
[6] E. Coumans, Y. Bai, "PyBullet Quickstart Guide". (getContactPoints)
[7] Course Tutorial 6 (QP solver wrapper).
"""

import numpy as np
import pinocchio as pin
import quadprog
from scipy.special import comb

from bezier import Bezier
from config import LEFT_HAND, RIGHT_HAND, DT


Kp = 300.
Kv = 2 * np.sqrt(Kp)
Ki = 600.

# Grip force feedback.
# FORCE is only the *starting* squeeze command; the loop below changes it every step so that the
# contact force PyBullet actually measures ends up at F_TARGET.
FORCE = 100 #Force we want each hand to command (N). The actual force the cube feels is measured and used to adjust this command.
F_TARGET = 100.   # contact force we want each hand to actually press with (N)
K_F = 20.         # integral gain on the force error (1/s); 0 switches the feedback off
grip = [FORCE]    # current squeeze command; a list so controllaw can update it without `global`

integ = np.zeros(15)   # running integral of the joint tracking error (for the Ki term)


def path_divisions(path, samples):
    """Split the path into chunks of `samples` points; neighbouring chunks share an endpoint.
    Each chunk becomes one Bezier piece. Fitting short pieces works much better than trying to fit
    one curve to the whole path, which would cut corners near the obstacle."""
    divisions = []
    last = len(path) - 1
    start = 0
    while start < last:
        end = min(start + samples - 1, last)
        if last - end < 2:          # don't leave a tiny 1-interval piece at the end, merge it in
            end = last
        divisions.append(path[start:end + 1])
        start = end
    return divisions


def densify(path, max_step=0.1):
    """The planner only returns the corners of the path. This adds points along each straight edge
    (linear interpolation in joint space) so that no step is bigger than max_step, giving the curve
    fit enough samples to follow the edges closely."""
    out = [np.asarray(path[0])]
    for a, b in zip(path[:-1], path[1:]):
        a, b = np.asarray(a), np.asarray(b)
        m = max(1, int(np.ceil(np.linalg.norm(b - a) / max_step)))
        for s in range(1, m + 1):
            out.append(a + (b - a) * s / m)
    return out


def smooth_path(path, window=3):
    """Filter to round off the sharp corners a bit.
    Kept small on purpose, a bigger window cut the corners so much that the cube hit the obstacle.
    The first and last points are put back so the start and goal grasps stay exact."""
    p = np.array(path)
    padded = np.pad(p, ((window // 2, window // 2), (0, 0)), mode="edge")
    kernel = np.ones(window) / window
    out = np.stack([np.convolve(padded[:, j], kernel, mode="valid") for j in range(p.shape[1])]).T
    out[0], out[-1] = p[0], p[-1]
    return list(out)


def cumulative_distances(path):
    """Arc length along the path, scaled to go from 0 to 1. This tells the fit which value of the
    Bezier parameter u each sample should correspond to, so samples that are far apart along the
    path are also far apart along the curve (chord-length parameterisation [2])."""
    dists = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    if dists[-1] == 0:                       # degenerate case: all points identical
        return np.linspace(0, 1, len(path))
    return dists / dists[-1]


def bezier_basis(degree, u_values):
    """Matrix of Bernstein polynomials: row k holds B_i,n(u_k) = C(n,i) u^i (1-u)^(n-i) for every
    control point i. A Bézier curve evaluated at the samples is then just B @ P [2]."""
    u = np.asarray(u_values, dtype=float)[:, None]      # (N, 1)
    i = np.arange(degree + 1)[None, :]                  # (1, n+1)
    return comb(degree, i) * u**i * (1 - u)**(degree - i)


def quadprog_solve_qp(H, q, G=None, h=None, C=None, d=None):
    """From Tutorial 6 [7].
    Solves  min 1/2 x'Hx + q'x  subject to  Gx <= h  and  Cx = d.
    quadprog [3] uses the opposite sign convention (C'x >= b, equalities first), hence the minus signs."""
    qp_G = 0.5 * (H + H.T)      # make sure H is symmetric
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
    """Fits one Bézier curve of degree n to each chunk of the path, by solving a small QP per joint.

    Cost: least squares between the curve and the path samples, plus a small penalty on the
    acceleration so the curve doesn't wiggle between samples.
    Constraints:
      - joint limits on the free control points (a Bézier curve stays inside the convex hull of its
        control points, so bounding them bounds the whole curve [2]);
      - the first three control points are fixed so the piece starts with the same position,
        velocity and acceleration as the previous piece ended with (C2 continuity). That's why n >= 6;
      - the last control point is the end of the chunk; for the final piece the second-to-last is
        fixed as well so the robot comes to rest at the goal.
    The duration T of each piece comes from the busiest joint's distance divided by vmax (with a
    safety factor `slow`). If the QP is infeasible, T is increased and it tries again."""
    control_points_list, T_list = [], []
    I = np.eye(n + 1)
    prev_P, prev_T = None, None
    last = len(path_divided) - 1

    for k, path in enumerate(path_divided):
        is_last = (k == last)
        path = np.array(path)
        B = bezier_basis(n, cumulative_distances(path))

        # which control points are fixed (equality constraints) and which the QP can choose
        fixed = [0, 1, 2, n - 1, n] if is_last else [0, 1, 2, n]
        free = I[3:n - 1] if is_last else I[3:n]
        n_free = free.shape[0]
        C = I[fixed]
        G = np.vstack([free, -free])          # upper and lower joint limits as G x <= h

        def solve_piece(T):
            # second differences of the control points, scaled by n(n-1)/T^2, are the control
            # points of the acceleration curve -> this term penalises acceleration
            D2 = np.diff(I, 2, axis=0) * n * (n - 1) / T**2
            H = 2 * (B.T @ B + w_acc * D2.T @ D2) + 1e-8 * I   # tiny regulariser keeps H positive definite
            P = np.zeros((n + 1, path.shape[1]))
            for j in range(path.shape[1]):     # joints are independent here, so one QP per joint
                ref = path[:, j]
                if prev_P is None:                       # first piece: start at rest
                    p0 = p1 = p2 = ref[0]
                else:
                    # match the previous piece's end position, velocity and acceleration.
                    # Derivatives of a Bézier curve come from differences of its control points;
                    # the T / prev_T factors rescale them because the pieces last different times.
                    p0 = prev_P[n, j]
                    p1 = p0 + (prev_P[n, j] - prev_P[n - 1, j]) * T / prev_T
                    acc_prev = prev_P[n, j] - 2 * prev_P[n - 1, j] + prev_P[n - 2, j]
                    p2 = 2 * p1 - p0 + (T / prev_T) ** 2 * acc_prev
                d = np.array([p0, p1, p2] + [ref[-1]] * (2 if is_last else 1))
                h = np.r_[np.full(n_free, qmax[j]), np.full(n_free, -qmin[j])]
                P[:, j] = quadprog_solve_qp(H, -2 * B.T @ ref, G, h, C, d)
            return P

        # time for this piece: long enough that the joint that moves furthest stays under vmax
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
    """Turns each set of control points into position, velocity and acceleration functions of time.
    The derivative of a Bézier curve is another Bézier curve, so the Bézier class can give these directly."""
    trajs = []
    for P, T in zip(control_points_list, T_list):
        q_of_t = Bezier(P, t_max=T)
        vq_of_t = q_of_t.derivative(1)
        trajs.append([q_of_t, vq_of_t, vq_of_t.derivative(1)])
    return trajs


def squeeze_torques(robot, q, CUBE_MASS, force):
    """Joint torques that make each hand push on the cube.

    Each hand pushes towards the other one with `force` newtons, and also pushes up
    with half the cube's weight so the arms carry the cube between them. A force f at a hand is turned
    into joint torques with tau = J^T f, from the duality between velocities and forces [1].
    Only the linear part of the Jacobian (first 3 rows) is used since there's no torque applied."""
    pin.computeJointJacobians(robot.model, robot.data, q)
    pin.framesForwardKinematics(robot.model, robot.data, q)

    idL = robot.model.getFrameId(LEFT_HAND)
    idR = robot.model.getFrameId(RIGHT_HAND)
    pL = robot.data.oMf[idL].translation
    pR = robot.data.oMf[idR].translation
    axis = (pR - pL) / np.linalg.norm(pR - pL)  # unit vector, left hand -> right hand
    lift = np.array([0., 0., CUBE_MASS * 9.81 / 2])   # each hand holds up half the cube

    tau = np.zeros(robot.model.nv)
    for fid, sign in ((idL, +1), (idR, -1)):    # left pushes along +axis, right along -axis
        # LOCAL_WORLD_ALIGNED: Jacobian expressed at the hand but with world axes, to match `axis` [5]
        J = pin.getFrameJacobian(robot.model, robot.data, fid, pin.LOCAL_WORLD_ALIGNED)
        tau += J[:3].T @ (sign * force * axis + lift)
    return tau


def controllaw(sim, robot, trajs, end_times, tcurrent, CUBE_MASS):
    """Called once per simulation step.
    1. look up where the robot should be at this time (reference trajectory),
    2. PID on the joint error gives a desired acceleration,
    3. inverse dynamics (RNEA) turns that acceleration into joint torques,
    4. force feedback adjusts the grip, and the squeeze torques are added on top."""
    q, vq = sim.getpybulletstate()

    # find which Bezier piece we're in and the local time within it.
    # After the last piece, t_local is capped so the reference just holds the final pose.
    idx = min(int(np.searchsorted(end_times, tcurrent, side="right")), len(trajs) - 1)
    start = 0. if idx == 0 else end_times[idx - 1]
    q_of_t, vq_of_t, aq_of_t = trajs[idx]
    t_local = min(tcurrent - start, q_of_t.T_max_)

    q_ref, vq_ref, aq_ref = q_of_t(t_local), vq_of_t(t_local), aq_of_t(t_local)

    # Computed torque control [1]: desired acceleration = feedforward from the trajectory
    # + PID correction on the tracking error. The integral is clipped so it can't wind up.
    integ[:] = np.clip(integ + (q_ref - q) * DT, -0.5, 0.5)
    aq = aq_ref + Kp * (q_ref - q) + Kv * (vq_ref - vq) + Ki * integ

    # Force feedback on the grip.
    # Commanding X newtons doesn't mean the cube actually feels X newtons (model errors, the PID
    # pulling the hands around, soft contacts), so measure the real contact force and correct for it.
    # The robot is the body with the most joints. getContactPoints returns every contact between the
    # cube and the robot; c[9] is the normal force of each one [6]. Halving the total gives roughly
    # the force per hand.
    # (pyb is imported in __main__; that works because this file is run as a script.)
    robot_id = max(range(pyb.getNumBodies()), key=pyb.getNumJoints)
    measured = sum(c[9] for c in pyb.getContactPoints(bodyA=sim.cubeId, bodyB=robot_id)) / 2

    # Integral force control [1]: nudge the command up if the cube feels too little force, down if
    # too much. Clipped between 0 (never pull the hands apart) and 400 N (don't crush it).
    grip[0] = np.clip(grip[0] + K_F * (F_TARGET - measured) * DT, 0., 400.)
    force = grip[0]

    # RNEA [4] gives the torques for the desired motion (including gravity and Coriolis terms);
    # the grip / lift torques are added on top.
    torques = pin.rnea(robot.model, robot.data, q, vq, aq) + squeeze_torques(robot, q, CUBE_MASS, force)
    sim.step(torques)


if __name__ == "__main__":

    from tools import setupwithpybullet, rununtil
    import pybullet as pyb
    robot, sim, cube = setupwithpybullet()

    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
    from inverse_geometry import computeqgrasppose
    from path import computepath

    # trajectory settings
    N = 6               # Bezier degree (C2 needs at least 6: 3 fixed points at the start + end point + free ones)
    SAMPLES = 6         # path samples per piece
    T_BASE = 0.4        # minimum duration of a piece (s)
    VMAX = 2.0          # speed limit (rad/s) used to work out how long each piece takes
    HOLD = 2.0          # extra seconds at the end holding the final pose so the robot can catch up

    # grasp configurations at the start and goal cube placements
    q0, _ = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT, None)
    qe, _ = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT_TARGET, None)

    # plan a collision-free path, then add samples along the edges and round the corners slightly
    path = computepath(robot, cube, q0, qe, CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)
    path = smooth_path(densify(path, max_step=0.1))

    sim.setqsim(path[0])
    CUBE_MASS = pyb.getDynamicsInfo(sim.cubeId, -1)[0]   # mass of the cube's base link, for the lift force

    # fit the piecewise Bezier trajectory
    control_points, T_list = optimise_control_points(
        path_divisions(path, SAMPLES), N, T_BASE,
        robot.model.lowerPositionLimit, robot.model.upperPositionLimit,
        vmax=np.full(15, VMAX))

    trajs = maketraj(control_points, T_list)
    end_times = np.cumsum(T_list)   # time at which each piece ends

    # run the controller for the length of the trajectory plus the hold time
    input("Control Trajectory Ready!! Press Enter to run the simulation...")
    tcur = 0.
    while tcur < end_times[-1] + HOLD:
        rununtil(controllaw, DT, sim, robot, trajs, end_times, tcur, CUBE_MASS)
        tcur += DT
