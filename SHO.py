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
