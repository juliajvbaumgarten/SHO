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
  for i in range(n):
    x[i+1] = x[i] + dt * v[i]
    a = -(k/m) * x[i]
    v[i+1] = v[i] + dt * a
  return x,v

def symplectic_euler(x0, v0, dt, n):
  """Symplectic (semi-implicit) Euler: 
      v_{n+1} = v_n + dt * ( - (k/m) * x_n)
      x_{n+1} = x_n + dt * v_{n+1}
  """
  x = np.empty(n+1)
  v = np.empty(n+1)
  x[0], v[0] = x0, v0
  for i in range(n):
    a = -(k/m) * x[i]
    v[i+1] = v[i] + dt * a
    x[i+1] = x[i] + dt * v[i+1]
  return x,v

def energy(x, v):
  """Total energy E = T + V = 1/2 m v^2 + 1/2 k x^2"""
  return 0.5 * m * v * v + 0.5 * k * x * x

# Time vector
t = np.linspace(0.0, T, n_steps+1)

# Run Methods
x_e, v_e = explicit_euler(x0, v0, dt, n_steps)
x_s, v_s = symplectic_euler(x0, v0, dt, n_steps)

# Energies
E_e = energy(x_e, v_e)
E_s = energy(x_s, v_s)

# -------------- Plots --------------------------------

# Displacement vs time
plt.figure()
plt.plot(t, x_e, label="Explicit Euler")
plt.plot(t, x_s, label="Symplectic Euler")
plt.xlabel("t [s]")
plt.ylabel("x(t) [m]")
plt.title("Spring Oscillator Displacement")
plt.legend()



# Energy vs time
plt.figure()
plt.plot(t, E_e, label="Explicit Euler Energy")
plt.plot(t, E_s, label="Symplectic Euler Energy")
plt.xlabel("t [s]")
plt.ylabel("Energy [J]")
plt.title("Energy as a function of time")
plt.legend()

# Numeric Summary
print("Parameters: m=%.3f kg, k=%.3f N/m, omega=%.3f rad/s, dt=%.4f s, steps=%d" % (m, k, omega, dt, n_steps))
print("Initial energy: %.6f J" % energy(x0, v0))
print("Final energy (Explicit Euler):  %.6f J" % E_e[-1])
print("Final energy (Symplectic Euler): %.6f J" % E_s[-1])

if __name__ == "__main__":
  plt.show()







