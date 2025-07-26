# Magnetic Mirror Effect using Velocity Verlet Integration

This repository simulates the **magnetic mirror effect** experienced by an electron in Earth's dipole magnetic field. The numerical integration is performed using the **magnetic velocity Verlet method**, which conserves energy well over long trajectories and is a improved verlet method.

## 🌍 Project Description

An electron with 30 keV energy is launched from the magnetic equatorial plane (at an altitude of 1 Earth radius) in a direction 45° between north and east. The Earth's magnetic field is modeled using a dipole approximation in spherical coordinates and then converted to Cartesian for simulation. The simulation shows:

- 3D trajectory of the electron
- Magnetic field strength variation
- The classic *magnetic mirror* bounce behavior
- Kinetic energy conservation over time

## 🧠 Physics & Numerical Method

The magnetic field is given in Polar Cordinates.

**Velocity Verlet integration** is used for updating position and velocity of the particle, ensuring time-symmetric evolution and better energy conservation.

## 📁 Repository Structure

| File / Folder                  | Description |
|-------------------------------|-------------|
| `Magnetic_mirror.py`          | Main simulation script using velocity Verlet method |
| `Visual_Mag_Feild_lines.py`   | Visualization of magnetic field lines (static) |
| `animated_Magnetic_mirror.py` | Script to generate animated `.mp4` of the magnetic mirror effect |
| `start_of_trajectory.py`      | Simple visualization of the electron's initial motion before mirroring |
| `Images & Videos/`            | Plots, 3D trajectories and Animation files |

## 🧰 Libraries Used

- `numpy` — numerical calculations
- `matplotlib` — 2D and 3D plotting
- `mpl_toolkits.mplot3d` — 3D trajectory visualization
- `matplotlib.animation` — generating and saving animations (in `animated_Magnetic_mirror.py`)

## 📊 Output Examples

- 3D trajectory plots in Earth's magnetic field
- z vs time graphs showing bouncing/mirroring
- Kinetic energy conservation over time
- Animated visualization (`.mp4`) of particle motion
