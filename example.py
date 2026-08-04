import numpy as np
from simulation import solve_heat_cube


# Top: radial Gaussian hot spot at the origin
f_top = lambda x, y: 200 + 800 * np.exp(-(x**2 + y**2) / 20)

# Bottom: cold ring around the origin (low in center, hot at edges)
f_bottom = lambda x, y: 50 + 400 * (np.sqrt(x**2 + y**2) / 14.14)

# Left: linear gradient through the origin, increasing with x
f_left = lambda x, y: 500 + 40 * x

# Right: linear gradient through the origin, increasing with y
f_right = lambda x, y: 500 + 40 * y

# Front: saddle shape centered at origin
f_front = lambda x, y: 750

# Back: sinusoidal ripple centered at origin
f_back = lambda x, y: 500 + 300 * np.sin(np.pi * x / 10) * np.cos(np.pi * y / 10)


solve_heat_cube(f_top, f_bottom, f_left, f_right, f_front, f_back, n=15, slice_index=4, slice_axis='x')





