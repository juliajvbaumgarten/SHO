"""
Spring oscillator
1) Explicit Euler
2) Symplectic (Semi-Implicit) Euler
3) Runge-Kutta 2 (Midpoint method)

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

def acceleration(x):
  return -(k/m) * x

def explicit_euler(x0, v0, dt, n):
  """Explicit Euler: x_{n+1} = x_n + dt * v_n
      v_{n+1} = v_n + dt * ( - (k/m) * x_n)
  """
  x = np.empty(n+1)
  v = np.empty(n+1)
  x[0], v[0] = x0, v0
  for i in range(n):
    x[i+1] = x[i] + dt * v[i]
    v[i+1] = v[i] + dt * acceleration(x[i])
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
    v[i+1] = v[i] + dt * acceleration(x[i])
    x[i+1] = x[i] + dt * v[i+1]
  return x,v

def rk2_midpoint(x0, v0, dt, n):
  x = np.empty(n+1)
  v = np.empty(n+1)
  x[0], v[0] = x0, v0
  for i in range(n):
    # k1
    k1x = v[i]
    k1v = acceleration(x[i])
    # midpoint
    xmid = x[i] + 0.5 * dt * k1x
    vmid = v[i] + 0.5 * dt * k1v
    # k2 at midpoint
    k2x = vmid
    k2v = acceleration(x[i])
    # advance
    x[i+1] = x[i] + dt * k2x
    v[i+1] = v[i] + dt * k2v
  return x, v

def exact_solution(t, x0, v0):
  """Analytic SHO solution"""
  x = x0 * np.cos(omega * t) + (v0/omega) * np.sin(omega * t)
  v = -x0 * omega * np.sin(omega * t) + v0 * np.cos(omega * t)
  return x, v

def energy(x, v):
  """Total energy E = T + V = 1/2 m v^2 + 1/2 k x^2"""
  return 0.5 * m * v * v + 0.5 * k * x * x

def compare_integrators(x0,v0, dt, T):
  """Compares all methods against analytic solution"""
  # Time vector
  t = np.linspace(0.0, T, n_steps+1)

  # Analytic
  x_ex, v_ex = exact_solution(t, x0, v0)
  E_ex = energy(x_ex, v_ex)
  
  # Run Methods
  x_e, v_e = explicit_euler(x0, v0, dt, n_steps)
  x_s, v_s = symplectic_euler(x0, v0, dt, n_steps)
  x_r, v_r = rk2_midpoint(x0, v0, dt, n_steps)
  
  # Energies
  E_e = energy(x_e, v_e)
  E_s = energy(x_s, v_s)
  E_r = energy(x_r, v_r)

  # Errors
  err_e_final = abs(x_e[-1] - x_ex[-1])
  err_s_final = abs(x_s[-1] - x_ex[-1])
  err_r_final = abs(x_r[-1] - x_ex[-1])

  dE_e = np.max(np.abs(E_e - E_ex))
  dE_s = np.max(np.abs(E_s - E_ex))
  dE_r = np.max(np.abs(E_r - E_ex))

  print(f"\n=== Accuracy @ T={T:.3f}s, dt={dt} ===")
  print(f"Final |x - x_exact|:")
  print(f"  Explicit Euler   : {err_e_final:.6g}")
  print(f"  Symplectic Euler : {err_s_final:.6g}")
  print(f"  RK2 (midpoint)   : {err_r_final:.6g}")

  print("\nMax |Energy - Exact Energy| over [0,T]:")
  print(f"  Explicit Euler   : {dE_e:.6g}")
  print(f"  Symplectic Euler : {dE_s:.6g}")
  print(f"  RK2 (midpoint)   : {dE_r:.6g}")

  plt.figure()
  plt.plot(t, x_e, label="Explicit Euler")
  plt.plot(t, x_s, label="Symplectic Euler")
  plt.plot(t, x_r, label="RK2 (midpoint)")
  plt.plot(t, x_ex, linestyle="--", label="Analytic")
  plt.xlabel("t [s]")
  plt.ylabel("x(t) [m]")
  plt.title("SHO: displacement vs time")
  plt.legend()

  plt.figure()
  plt.plot(t, E_e, label="Explicit Euler")
  plt.plot(t, E_s, label="Symplectic Euler")
  plt.plot(t, E_r, label="RK2 (midpoint)")
  plt.plot(t, E_ex, linestyle="--", label="Analytic")
  plt.xlabel("t [s]")
  plt.ylabel("Energy [J]")
  plt.title("Energy vs time")
  plt.legend()
  
  return {
      "t": t,
      "explicit": (x_e, v_e, E_e),
      "symplectic": (x_s, v_s, E_s),
      "rk2": (x_r, v_r, E_r),
      "analytic": (x_ex, v_ex, E_ex),
  }

# -------------- Plots --------------------------------

# Time vector
t = np.linspace(0.0, T, n_steps+1)

# Run Methods
x_e, v_e = explicit_euler(x0, v0, dt, n_steps)
x_s, v_s = symplectic_euler(x0, v0, dt, n_steps)
x_r, v_r = rk2_midpoint(x0, v0, dt, n_steps)
  
# Energies
E_e = energy(x_e, v_e)
E_s = energy(x_s, v_s)
E_r = energy(x_r, v_r)

# Displacement vs time
plt.figure()
plt.plot(t, x_e, label="Explicit Euler")
plt.plot(t, x_s, label="Symplectic Euler")
plt.plot(t, x_r, label="RK2 (midpoint)")
plt.xlabel("t [s]")
plt.ylabel("x(t) [m]")
plt.title("Spring Oscillator Displacement")
plt.legend()



# Energy vs time
plt.figure()
plt.plot(t, E_e, label="Explicit Euler Energy")
plt.plot(t, E_s, label="Symplectic Euler Energy")
plt.plot(t, E_r, label="RK2 (midpoint)")
plt.xlabel("t [s]")
plt.ylabel("Energy [J]")
plt.title("Energy as a function of time")
plt.legend()

# Numeric Summary
print("Parameters: m=%.3f kg, k=%.3f N/m, omega=%.3f rad/s, dt=%.4f s, steps=%d" % (m, k, omega, dt, n_steps))
print("Initial energy: %.6f J" % energy(x0, v0))
print("Final energy (Explicit Euler):  %.6f J" % E_e[-1])
print("Final energy (Symplectic Euler): %.6f J" % E_s[-1])
print("Final energy (RK2 (midpoint)): %.6f J" % E_r[-1])

if __name__ == "__main__":
  results = compare_integrators(x0, v0, dt, T)
  plt.show()



