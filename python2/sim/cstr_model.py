import math
import numpy as np

from math import exp
from dataclasses import dataclass
from scipy.integrate import odeint


@dataclass
class CSTRModel:

    # fmt: off
    T: float  # Kelvin
    Tc: float  # Kelvin
    Ca: float  # kmol/m3
    dTc: float
    dT: float = 0  # Kelvin
    dCa: float = 0  # kmol/m3

    # Constants
    F: float = 1  # Volumetric flow rate (m3/h)
    V: float = 1  # Reactor volume (m3)
    k0: float = 34930800  # Pre-exponential nonthermal factor (1/h)
    E: float = 11843  # Activation energy per mole (kcal/kmol)
    R: float = 1.98588  # 1.985875 #Boltzmann's ideal gas constant (kcal/(kmol·K))
    dH: float = -5960  # Heat of reaction per mole kcal/kmol
    phoCp: float = 500  # Density multiplied by heat capacity (kcal/(m3·K))
    UA: float = 150  # Overall heat transfer coefficient multiplied by tank area (kcal/(K·h))
    Cafin: float = 10  # kmol/m3
    Tf: float = 298.2  # K
    # fmt: on

    def model(z, t, u):
        x = z[0]
        y = z[1]

        # fmt: off
        dxdt = (self.F / self.V * (self.Cafin - x)) - (self.k0 * exp(-self.E / (self.R * y)) * x)
        dydt = (
            (self.F / self.V * (self.Tf - y))
            - ((self.dH / self.phoCp) * (self.k0 * exp(-self.E / (self.R * y)) * x))
            - ((self.UA / (self.phoCp * self.V)) * (y - self.Tc + u))
        )
        # fmt: on

        dzdt = [dxdt, dydt]
        return dzdt

    def __post_init__(self):
        z0 = [self.Ca, self.T]  # Initial condition
        n = 3  # End time?
        t = np.linspace(0, 1, n)

        u = np.zeros(n)  # Input
        u[0] = self.dTc  # -1, 0, ...., 0

        # solution
        x = np.empty_like(t)
        y = np.empty_like(t)

        x[0] = z0[0]
        y[0] = z0[1]

        # solve ODE
        for i in range(1, n):
            tspan = [t[i - 1], t[i]]
            z = odeint(self.model, z0, tspan, args=(u[i],))
            x[i] = z[1][0]
            y[i] = z[1][1]
            z0 = z[1]

        self.Ca = x[1]
        self.T = y[1]
        self.Tc += self.dTc

    def run_sim(self):

        # initial cond
        z0 = [self.Ca, self.T]
        n = 3  # 3 4
        t = np.linspace(0, 1, n)
        # input
        # u = [0,self.dTc]
        u = np.zeros(n)
        u[0] = self.dTc  # -1 0

        # solution
        x = np.empty_like(t)
        y = np.empty_like(t)

        x[0] = z0[0]
        y[0] = z0[1]

        # solve ODE
        for i in range(1, n):
            tspan = [t[i - 1], t[i]]
            z = odeint(self.model, z0, tspan, args=(u[i],))
            x[i] = z[1][0]
            y[i] = z[1][1]
            z0 = z[1]

        self.Ca = x[1]
        self.T = y[1]
        self.Tc += self.dTc
