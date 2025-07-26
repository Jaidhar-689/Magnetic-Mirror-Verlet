import numpy as np
import matplotlib.pyplot as plt

e = 1.602e-19  # Charge of electron (C)
m_e = 9.109e-31  # Mass of electron (kg)
B0 = 3.12e-5  # Tex`sla
Re = 6.371e6  # Earth radius in meters
KE = 30e3 * e  # Convert keV to Joules
# eB/m
def B(position):
    x,y,z = position
    r = np.sqrt(x*x + y*y + z*z)
    theta = np.arccos(z/r)  
    phi = np.arctan2(y,x)   

    Br = -2*B0*(Re/r)**3*np.cos(theta)
    Btheta = -B0*(Re/r)**3*np.sin(theta)
    Bphi = 0

    # Convert spherical to Cartesian components
    Bx = Br * np.sin(theta) * np.cos(phi) + Btheta * np.cos(theta) * np.cos(phi)
    By = Br * np.sin(theta) * np.sin(phi) + Btheta * np.cos(theta) * np.sin(phi)
    Bz = Br * np.cos(theta) - Btheta * np.sin(theta)

    return np.array([Bx,By,Bz])

v_mag = np.sqrt(2*KE/m_e)
initial_position = np.array([0,2*Re,0])
initial_velocity = v_mag*np.array([-1,0,1])/np.linalg.norm([-1,0,1])

position = [initial_position]
velocity = [initial_velocity]
dt = 1e-9
endt = 1e-3
t = []
N = int(endt/dt)

for i in range(N):
    x0 = position[-1]
    d = velocity[i] + (-e * dt / (2 * m_e))*np.cross(velocity[i],B(x0))
    r = x0 + d*dt
    C = B(r)
    c = (-e*dt/(2*m_e))*C
    v = (d + np.cross(d,c) + c*np.dot(d,c))/(1+np.dot(c,c))

    position.append(r)
    velocity.append(v)
    t.append(i*dt)

    if (i*100/N)%1 == 0:
        print(f'{i*100//N}%')

t.append(endt)
position = np.array(position)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot3D(position[:, 0], position[:, 1], position[:, 2])
plt.xlabel('x')
plt.ylabel('y')
ax.set_zlabel('z')
plt.show()

ke = m_e*np.array([np.dot(i,i) for i in velocity])/2/e
# print(ke)
ke[0] = 0
plt.plot(t,ke)
plt.xlabel("Time")
plt.ylabel("KE")
plt.grid()
plt.show()
