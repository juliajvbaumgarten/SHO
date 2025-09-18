"""
Spring oscillator
1) Explicit Euler
2) Symplectic (Semi-Implicit) Euler

Hooke's law --> F = -k x
Newton's 2nd law --> m x'' = F = -k x
Let v = x'

Then the ODE system is:
dx/dt = v
dv/dt = -(k/m) x
"""

# ---------- Import modules ---------- 
import math
import numpy as np
import matplotlib.pyplot as plt

# ----------- Parameters -------------
m = 1.0 # mass [kg]
k = 4.0 # Spring [N/m] (omega = sqrt(k/m) = 2 rad/s)
omega = math.sqrt(k/m)

# ------------------------------------
dt = 0.01 # timestep [s]
T = 20.0 # total time [s]
n_steps = int(T/dt)

# ------ Initial Conditions ---------
x0 = 1.0
v0 = 0.5

def explicit_euler(x0, v0, dt, n):
  """Explicit Euler: x_{n+1} = x_n + dt * v_n
      v_{n+1} = v_n + dt * ( - (k/m) * x_n)
  """
  x = np.empty(n+1)
  v = np.empty(n+1)
  x[0], v[0] = x0, v0
  
  








