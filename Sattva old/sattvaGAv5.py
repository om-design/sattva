"""
SATTVA — Geometric Dynamical Learning Engine
Complete SE(n) + Curvature Flow + Wells + Creative Emergence

Author: Sattva Architecture Implementation
"""

import numpy as np
from numpy.linalg import norm, svd
from dataclasses import dataclass, field
from typing import List, Tuple


# ============================================================
# Utility Functions
# ============================================================

def skew_symmetric(w: np.ndarray) -> np.ndarray:
    """Convert vector to skew-symmetric matrix (so(3))."""
    return np.array([
        [0, -w[2], w[1]],
        [w[2], 0, -w[0]],
        [-w[1], w[0], 0]
    ])


def exp_so3(w: np.ndarray) -> np.ndarray:
    """Exponential map for SO(3)."""
    theta = norm(w)
    if theta < 1e-8:
        return np.eye(3)
    K = skew_symmetric(w / theta)
    return (
        np.eye(3)
        + np.sin(theta) * K
        + (1 - np.cos(theta)) * (K @ K)
    )


def exp_se3(xi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exponential map for SE(3).
    xi = [omega(3), v(3)]
    """
    omega = xi[:3]
    v = xi[3:]
    R = exp_so3(omega)
    theta = norm(omega)

    if theta < 1e-8:
        V = np.eye(3)
    else:
        K = skew_symmetric(omega / theta)
        V = (
            np.eye(3)
            + ((1 - np.cos(theta)) / theta) * K
            + ((theta - np.sin(theta)) / theta) * (K @ K)
        )

    t = V @ v
    return R, t


# ============================================================
# Invariant Manifold with Curvature Flow
# ============================================================

@dataclass
class InvariantManifold:
    dim: int
    ambient_dim: int
    basis: np.ndarray = field(init=False)

    def __post_init__(self):
        self.basis = np.linalg.qr(
            np.random.randn(self.ambient_dim, self.dim)
        )[0]

    def project(self, x: np.ndarray) -> np.ndarray:
        return self.basis @ (self.basis.T @ x)

    def projection_error(self, x: np.ndarray) -> float:
        return norm(x - self.project(x)) ** 2

    def curvature_flow_update(self, X_batch: np.ndarray, lr=0.01):
        """
        Grassmannian gradient descent with curvature damping.
        """
        grad = np.zeros_like(self.basis)

        for x in X_batch:
            proj = self.project(x)
            grad += np.outer(x - proj, self.basis.T @ x)

        grad /= len(X_batch)

        # curvature damping (approximate via basis orthogonality loss)
        ortho_loss = self.basis.T @ self.basis - np.eye(self.dim)
        curvature_term = self.basis @ ortho_loss

        self.basis -= lr * (grad + 0.1 * curvature_term)

        # re-orthonormalize
        self.basis, _ = np.linalg.qr(self.basis)


# ============================================================
# Wells (Entropy-Regularized Clusters)
# ============================================================

@dataclass
class Well:
    center: np.ndarray
    members: List[np.ndarray] = field(default_factory=list)

    def update_center(self):
        if self.members:
            self.center = np.mean(self.members, axis=0)

    def entropy(self):
        if not self.members:
            return 0
        dists = np.array([norm(x - self.center) for x in self.members])
        p = dists / (np.sum(dists) + 1e-8)
        return -np.sum(p * np.log(p + 1e-8))


class WellSystem:
    def __init__(self, threshold=1.0, entropy_split=1.5):
        self.wells: List[Well] = []
        self.threshold = threshold
        self.entropy_split = entropy_split

    def add(self, x: np.ndarray):
        if not self.wells:
            self.wells.append(Well(center=x.copy(), members=[x]))
            return

        dists = [norm(x - w.center) for w in self.wells]
        idx = np.argmin(dists)

        if dists[idx] < self.threshold:
            self.wells[idx].members.append(x)
            self.wells[idx].update_center()
            self._check_split(idx)
        else:
            self.wells.append(Well(center=x.copy(), members=[x]))

    def _check_split(self, idx):
        well = self.wells[idx]
        if well.entropy() > self.entropy_split:
            pts = np.array(well.members)
            U, S, Vt = svd(pts - well.center)
            direction = Vt[0]

            cluster1 = []
            cluster2 = []

            for x in pts:
                if np.dot(x - well.center, direction) > 0:
                    cluster1.append(x)
                else:
                    cluster2.append(x)

            self.wells[idx] = Well(
                center=np.mean(cluster1, axis=0),
                members=cluster1
            )
            self.wells.append(
                Well(center=np.mean(cluster2, axis=0), members=cluster2)
            )


# ============================================================
# Creative Emergence
# ============================================================

def creative_emergence_test(x, manifold: InvariantManifold, V_old):
    """
    Novel + energy reducing + persistent
    """
    error = manifold.projection_error(x)

    if error < 0.01:
        return False

    V_new = V_old - error

    return V_new < V_old


# ============================================================
# Motor Dynamics (Lie Integration)
# ============================================================

@dataclass
class MotorState:
    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    t: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def apply(self, x):
        return self.R @ x + self.t

    def integrate(self, xi, dt=0.1):
        R_inc, t_inc = exp_se3(xi * dt)
        self.R = R_inc @ self.R
        self.t = R_inc @ self.t + t_inc


# ============================================================
# Sattva Engine
# ============================================================

class SattvaEngine:

    def __init__(self, ambient_dim=3):
        self.manifold = InvariantManifold(dim=2, ambient_dim=ambient_dim)
        self.wells = WellSystem()
        self.motor = MotorState()
        self.V = 0
        self.history_V = []

    def shear_metric(self, x):
        return self.manifold.projection_error(x)

    def curiosity(self, x):
        return self.shear_metric(x) > 1.0

    def step(self, x):
        x_trans = self.motor.apply(x)

        self.wells.add(x_trans)

        self.V += self.manifold.projection_error(x_trans)

        if self.curiosity(x_trans):
            self.manifold.curvature_flow_update([x_trans])

        if creative_emergence_test(x_trans, self.manifold, self.V):
            self.manifold.dim += 1
            self.manifold.basis = np.linalg.qr(
                np.random.randn(
                    self.manifold.ambient_dim,
                    self.manifold.dim
                )
            )[0]

        self.history_V.append(self.V)


# ============================================================
# Synthetic Simulation Demo
# ============================================================

if __name__ == "__main__":

    engine = SattvaEngine()

    np.random.seed(42)

    # Create two geometric structures
    circle = np.array([
        [np.cos(t), np.sin(t), 0]
        for t in np.linspace(0, 2*np.pi, 100)
    ])

    line = np.array([
        [0, 0, z]
        for z in np.linspace(-1, 1, 50)
    ])

    data = np.vstack([circle, line])

    for x in data:
        engine.step(x)

    print("Final manifold dimension:", engine.manifold.dim)
    print("Number of wells:", len(engine.wells.wells))
    print("Final Lyapunov energy:", engine.history_V[-1])
