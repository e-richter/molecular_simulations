import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def velocity_verlet(r1_0, r2_0, p1_0, p2_0, t_max, dt, f):
    t = np.arange(0, t_max, dt)

    r1 = np.zeros((len(t), r1_0.shape[0]))
    r2 = np.zeros((len(t), r2_0.shape[0]))
    p1 = np.zeros((len(t), p1_0.shape[0]))
    p2 = np.zeros((len(t), p2_0.shape[0]))

    r1[0] = r1_0
    r2[0] = r2_0
    p1[0] = p1_0
    p2[0] = p2_0

    r1_i = r1_0
    r2_i = r2_0
    p1_i = p1_0
    p2_i = p2_0

    for i in range(len(t) - 1):
        a11, a12 = f(r1_i, r2_i)

        p1_i += a11 * dt / 2.
        p2_i += a12 * dt / 2.

        r1_i += p1_i * dt
        r2_i += p2_i * dt

        a21, a22 = f(r1_i, r2_i)

        p1_i += a21 * dt / 2.
        p2_i += a22 * dt / 2.

        r1[i + 1] = r1_i
        r2[i + 1] = r2_i

        p1[i + 1] = p1_i
        p2[i + 1] = p2_i

    return r1, r2, p1, p2, t


def BABAB(r1_0, r2_0, p1_0, p2_0, t_max, dt, f, l):
    t = np.arange(0, t_max, dt)

    r1 = np.zeros((len(t), r1_0.shape[0]))
    r2 = np.zeros((len(t), r2_0.shape[0]))
    p1 = np.zeros((len(t), p1_0.shape[0]))
    p2 = np.zeros((len(t), p2_0.shape[0]))

    r1[0] = r1_0
    r2[0] = r2_0
    p1[0] = p1_0
    p2[0] = p2_0

    r1_i = r1_0
    r2_i = r2_0
    p1_i = p1_0
    p2_i = p2_0

    for i in range(len(t) - 1):
        a11, a12 = f(r1_i, r2_i)

        p1_i += a11 * dt * l
        p2_i += a12 * dt * l

        r1_i += p1_i * dt / 2.
        r2_i += p2_i * dt / 2.

        a21, a22 = f(r1_i, r2_i)

        p1_i += a21 * dt * (1 - 2 * l)
        p2_i += a22 * dt * (1 - 2 * l)

        r1_i += p1_i * dt / 2.
        r2_i += p2_i * dt / 2.

        a31, a32 = f(r1_i, r2_i)

        p1_i += a31 * dt * l
        p2_i += a32 * dt * l

        r1[i + 1] = r1_i
        r2[i + 1] = r2_i

        p1[i + 1] = p1_i
        p2[i + 1] = p2_i

    return r1, r2, p1, p2, t


def rungekutta4(y0, t, f, v):
    n = len(t)
    y = np.zeros((n, y0.shape[0], y0.shape[1]))
    y[0] = y0

    for i in range(n - 1):
        r1 = y[i][0]
        r2 = y[i][1]
        p1 = y[i][2]
        p2 = y[i][3]

        h = t[i + 1] - t[i]

        k11, k12 = v(p1, p2)
        k13, k14 = f(r1, r2)

        k21, k22 = v(p1 + k13 * h / 2., p2 + k14 * h / 2.)
        k23, k24 = f(r1 + k11 * h / 2., r2 + k12 * h / 2.)

        k31, k32 = v(p1 + k23 * h / 2., p2 + k24 * h / 2.)
        k33, k34 = f(r1 + k21 * h / 2., r2 + k22 * h / 2.)

        k41, k42 = v(p1 + k33 * h / 2., p2 + k34 * h / 2.)
        k43, k44 = f(r1 + k31 * h / 2., r2 + k32 * h / 2.)

        y[i + 1][0] = r1 + (h / 6.) * (k11 + 2 * k21 + 2 * k31 + k41)

        y[i + 1][1] = r2 + (h / 6.) * (k12 + 2 * k22 + 2 * k32 + k42)

        y[i + 1][2] = p1 + (h / 6.) * (k13 + 2 * k23 + 2 * k33 + k43)

        y[i + 1][3] = p2 + (h / 6.) * (k14 + 2 * k24 + 2 * k34 + k44)

    return y


def BABAB_Ndim(r0, p0, t_max, dt, f, lam, thermal_noise: bool, periodic=None):
    if periodic is None:
        periodic = {'PBC': False,
                    'box_size': 0,
                    'closed': False}

    t = np.arange(0, t_max, dt)

    r = np.zeros([len(t), r0.shape[0], r0.shape[1]])
    p = np.zeros([len(t), p0.shape[0], p0.shape[1]])

    r[0] = r0
    p[0] = p0

    r_i = r0
    p_i = p0

    if thermal_noise:
        tn = int(1 / dt)
    else:
        tn = np.nan

    for i in tqdm(range(len(t) - 1)):

        if i % tn == 0:
            p_i = np.random.normal(loc=0.0, scale=1.0, size=r0.shape)
        else:
            a1 = f(r_i, periodic={'PBC': periodic['PBC'], 'box_size': periodic['box_size']})
            p_i += a1 * dt * lam

        r_i += p_i * dt / 2.
        if periodic['PBC']:
            r_i[np.where(r_i > periodic['box_size'] / 2.)] -= periodic['box_size']
            r_i[np.where(r_i < -periodic['box_size'] / 2.)] += periodic['box_size']

        a2 = f(r_i, periodic={'PBC': periodic['PBC'], 'box_size': periodic['box_size']})

        p_i += a2 * dt * (1 - 2 * lam)

        r_i += p_i * dt / 2.
        if periodic['PBC']:
            r_i[np.where(r_i > periodic['box_size'] / 2.)] -= periodic['box_size']
            r_i[np.where(r_i < -periodic['box_size'] / 2.)] += periodic['box_size']

        a3 = f(r_i, periodic={'PBC': periodic['PBC'], 'box_size': periodic['box_size']})

        p_i += a3 * dt * lam

        r[i + 1] = r_i
        p[i + 1] = p_i

    return r, p, t


def velocity_verlet_Ndim(r0, p0, t_max, dt, f, periodic=None):
    if periodic is None:
        periodic = {'PBC': False,
                    'box_size': 0,
                    'closed': False}

    t = np.arange(0, t_max, dt)

    r = np.zeros([len(t), r0.shape[0], r0.shape[1]])
    p = np.zeros([len(t), p0.shape[0], p0.shape[1]])

    r[0] = r0
    p[0] = p0

    r_i = r0
    p_i = p0

    for i in tqdm(range(len(t) - 1)):
        a1 = f(r_i, periodic={'PBC': periodic['PBC'], 'box_size': periodic['box_size']}, closed=periodic['closed'])

        r_i += p_i * dt + a1 * dt * dt / 2.

        if periodic['PBC']:
            r_i[np.where(r_i > periodic['box_size'] / 2.)] -= periodic['box_size']
            r_i[np.where(r_i < -periodic['box_size'] / 2.)] += periodic['box_size']

        a2 = f(r_i, periodic={'PBC': periodic['PBC'], 'box_size': periodic['box_size']}, closed=periodic['closed'])

        p_i += (a1 + a2) * dt / 2.

        r[i + 1] = r_i
        p[i + 1] = p_i

    return r, p, t


def marsaglia_method(N):

    z1 = np.zeros(N)
    z2 = np.zeros(N)

    for i in range(N):

        while True:
            u = (np.random.rand() * 2) - 1
            v = (np.random.rand() * 2) - 1
            q = u**2 + v**2

            if q < 1:
                p = np.sqrt(-2 * np.log(q) / q)

                z1[i] = u*p
                z2[i] = v*p

                break

    return z1, z2

