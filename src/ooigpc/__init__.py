"""
ooigpc — Output-only identification of thermoacoustic systems
          using polynomial chaos expansions.

Paper
-----
Silva, C.F. et al. (2026). *Output-only identification of thermoacoustic
systems using polynomial chaos expansions.* Combust. Flame.

Quick start
-----------
>>> from ooigpc.solvers.langevin import simulate_langevin
>>> from ooigpc.igpc import identify_from_signal
>>> import numpy as np
>>>
>>> # Simulate the Langevin SDE
>>> omega = 2 * np.pi * 120          # 120 Hz mode
>>> x = simulate_langevin(nu=16, kappa=3.4, Gamma=1e7, omega=omega,
...                       N=2_000_000, dt=5e-5, seed=0)
>>>
>>> # Identify parameters
>>> tau = np.arange(2.5e-4, 0.201, 2.5e-4)
>>> out = identify_from_signal(x, tau, dt_sim=5e-5, omega=omega)
>>> print(out['identification'])
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("ooigpc")
except PackageNotFoundError:
    __version__ = "unknown"

__author__ = "Camilo F. Silva"
__email__ = "camilo.silva@tum.de"
