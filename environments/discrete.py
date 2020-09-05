"""

"""
import numpy as np
from math import tanh


class MountainCar:
    """
    s = <x, v> where x represents position and v represents velocity
    """

    def __init__(self, nonlinear=True):
        self.x = None
        self.v = None
        self.goal = 1.0

        self.nonlinear = nonlinear

    def reset(self):
        """"""
        self.x = -0.5
        self.v = 0

    def step(self, action):
        """"""
        force = self.calc_force(action=action)
        self.v += force
        self.x += self.v

    def calc_force(self, action):
        """"""
        # (1) Calculate the downhill force
        if self.x < 0:
            downhill = 0.05 * (-2 * self.x - 1)
        else:
            downhill = 0.05 * ((-(1 + 5 * self.x**2)**(-1/2)) - self.x**2
                               * (1 + 5 * self.x**2)**(-3/2) - (self.x**4 / 16))

        # (2) Calculate the engine force
        engine = 0.03 * tanh(action)

        # (3) Calculate the friction
        friction = -0.25 * self.v

        return downhill + engine + friction

    def gen_state(self):
        """To be called after force is applied"""
        # Noisy sense of of it's position, x
        o_t = self.x + 0.01 * np.random.normal()
        if self.nonlinear:
            o_ht = np.exp((self.x - 1.0)**2 / 0.18)



print(2 * 0.3**2)