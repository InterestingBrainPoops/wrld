"""Spring-mass-damper environment.

Physics: m*x'' + c*x' + k*x = F
Observation: [position, velocity] (2D)
Action: force F (1D)
"""
import numpy as np


class SpringMassDamperEnv:
    def __init__(self, m: float = 1.0, k: float = 1.0, c: float = 0.3,
                 dt: float = 0.05, substeps: int = 10):
        self.m = m
        self.k = k
        self.c = c
        self.dt = dt
        self.substeps = substeps
        self.dt_inner = dt / substeps

        self.x = 0.0
        self.v = 0.0

    def reset(self, x0: float = None, v0: float = None) -> np.ndarray:
        if x0 is None:
            x0 = np.random.uniform(-2.0, 2.0)
        if v0 is None:
            v0 = np.random.uniform(-2.0, 2.0)
        self.x = float(x0)
        self.v = float(v0)
        return np.array([self.x, self.v], dtype=np.float32)

    def step(self, force: float) -> np.ndarray:
        force = float(force)
        for _ in range(self.substeps):
            a = (force - self.c * self.v - self.k * self.x) / self.m
            self.v += a * self.dt_inner        # velocity first (semi-implicit Euler)
            self.x += self.v * self.dt_inner
        return np.array([self.x, self.v], dtype=np.float32)

    def get_state(self) -> tuple:
        return (self.x, self.v)
