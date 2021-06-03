import numpy as np


def rtb_force(r1, r2, m0=1, r0=0):
    G = 4 * np.pi ** 2
    m_E = 3e-6
    m_L = 3.69e-8

    F1 = - G * m0 * m_E * (r1 - r0) / np.linalg.norm(r1 - r0) ** 3 \
         - G * m_E * m_L * (r1 - r2) / np.linalg.norm(r1 - r2) ** 3

    F2 = - G * m0 * m_L * (r2 - r0) / np.linalg.norm(r2 - r0) ** 3 \
         - G * m_L * m_E * (r2 - r1) / np.linalg.norm(r2 - r1) ** 3

    return F1, F2


def rtb_velocity(p1, p2):
    m_E = 3e-6
    m_L = 3.69e-8
    v1 = p1 / m_E
    v2 = p2 / m_L

    return v1, v2


def dumbbell_force(r1, r2, k=1):
    F1 = -k * (r1-r2)
    F2 = -k * (r2-r1)

    return F1, F2


def FENE_force(r1, r2, eps=1, a=1):
    F1 = -eps / a ** 2 * (1 / (1 - (np.linalg.norm(r1 - r2) ** 2 / a ** 2))) * (r1 - r2)
    F2 = -eps / a ** 2 * (1 / (1 - (np.linalg.norm(r2 - r1) ** 2 / a ** 2))) * (r2 - r1)

    return F1, F2


def chain_force(r, k=1):
    N = len(r)
    bond_force = np.zeros(r.shape)
    bond_force[0] = k * (r[1] - r[0])
    bond_force[N - 1] = -k * (r[N - 1] - r[N - 2])

    for i in range(1, N - 1):
        f = -(r[i] - r[i - 1]) + (r[i + 1] - r[i])
        f *= k
        bond_force[i] = f

    return bond_force / np.linalg.norm(bond_force)

def chain_force_PBC(r, L, k=1):
    N = len(r)
    bond_force = np.zeros(r.shape)
    
    r01 = r[1] - r[0]
    for dim in range(r.shape[1]):
        if r01[dim] > .5 * L:
                    r01[dim] -= L
        elif r01[dim] < -0.5 * L:
                    r01[dim] += L
    
    rN = r[N - 1] - r[N - 2]
    for dim in range(r.shape[1]):
        if rN[dim] > .5 * L:
                    r01[dim] -= L
        elif rN[dim] < -0.5 * L:
                    r01[dim] += L
    bond_force[0] = k * r01
    bond_force[N - 1] = -k * rN

    for i in range(1, N - 1):
        r1 = r[i] - r[i - 1]
        r2 = r[i+1] - r[i]
        for dim in range(r.shape[1]):
            if r1[dim] > .5 * L:
                r1[dim] -= L
            elif r1[dim] < -0.5 * L:
                r1[dim] += L
            if r2[dim] > .5 * L:
                r2[dim] -= L
            elif r2[dim] < -0.5 * L:
                r2[dim] += L
            f = -r1 + r2
            f *= k
            bond_force[i] = f
                
    return bond_force / np.linalg.norm(bond_force)

def chain_force_closed(r, k=1):
    N = len(r)
    bond_force = np.zeros(r.shape)
    bond_force[0] = k * (r[1] - r[0]) - k * (r[0] - r[N - 1])
    bond_force[N - 1] = -k * (r[N - 1] - r[N - 2]) + k * (r[0] - r[N - 1])

    for i in range(1, N - 1):
        f = -(r[i] - r[i - 1]) + (r[i + 1] - r[i])
        f *= k
        bond_force[i] = f

    return bond_force / np.linalg.norm(bond_force)
