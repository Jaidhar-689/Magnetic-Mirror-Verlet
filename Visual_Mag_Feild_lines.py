import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


B0 = 3.12e-5  # Tesla
Re = 6.371e6  # Earth radius in meters

# Dipole magnetic field in Cartesian coords
def B_field(pos):
    x, y, z = pos
    r = np.linalg.norm(pos)
    if r == 0: return np.zeros(3)
    theta = np.arccos(z/r)
    phi = np.arctan2(y, x)

    Br = -2 * B0 * (Re / r)**3 * np.cos(theta)
    Btheta = -B0 * (Re / r)**3 * np.sin(theta)

    Bx = Br * np.sin(theta) * np.cos(phi) + Btheta * np.cos(theta) * np.cos(phi)
    By = Br * np.sin(theta) * np.sin(phi) + Btheta * np.cos(theta) * np.sin(phi)
    Bz = Br * np.cos(theta) - Btheta * np.sin(theta)
    return np.array([Bx, By, Bz])

# Integrate one field line
def trace_field_line(start, step=0.01, n_steps=10000):
    path = [start]
    pos = start.copy()
    for _ in range(n_steps):
        B = B_field(pos)
        B_norm = np.linalg.norm(B)
        if B_norm == 0: break
        pos = pos + step * B / B_norm
        path.append(pos.copy())
        if np.linalg.norm(pos) > 5: break
    return np.array(path)

# Seed points around a sphere
phi_vals = np.linspace(0, 2*np.pi, 20)
theta_vals = np.linspace(0.1, np.pi-0.1, 10)
R = 2  # above Earth's surface
seeds = []

for theta in theta_vals:
    for phi in phi_vals:
        x = R * np.sin(theta) * np.cos(phi)
        y = R * np.sin(theta) * np.sin(phi)
        z = R * np.cos(theta)
        seeds.append(np.array([x, y, z]))

# Trace field lines from each seed
field_lines = [trace_field_line(seed) for seed in seeds]

# Plot it
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot Earth
u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
xe = np.sin(v) * np.cos(u)
ye = np.sin(v) * np.sin(u)
ze = np.cos(v)
ax.plot_surface(xe, ye, ze, color='lightskyblue', alpha=0.6, linewidth=0)

# Plot field lines
for line in field_lines:
    ax.plot(line[:, 0], line[:, 1], line[:, 2], color='orange', linewidth=1)

ax.set_title("Dipole Magnetic Field Lines (3D)")
ax.set_axis_off()
plt.tight_layout()
plt.show()

