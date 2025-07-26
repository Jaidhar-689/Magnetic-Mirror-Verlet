import numpy as np
import matplotlib.pyplot as plt

e = 1.602e-19  # Charge of electron (C)
m_e = 9.109e-31  # Mass of electron (kg)
B0 = 3.12e-5  # Tex`sla
Re = 6.371e6  # Earth radius in meters
KE = 30e3 * e  # Convert keV to Joules

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
Bmag = [np.linalg.norm(B(initial_position))]

dt = 2*np.pi*m_e/(e*Bmag[0])
N = int(1e5)
endt = dt*N
print(f"End Time = {endt}, dt = {dt}, N = {N}")
t = np.linspace(0,endt,N+1)

for i in range(N):
    x0 = position[-1]
    d = velocity[i] + (-e * dt / (2 * m_e))*np.cross(velocity[i],B(x0))
    r = x0 + d*dt
    C = B(r)
    c = (-e*dt/(2*m_e))*C
    v = (d + np.cross(d,c) + c*np.dot(d,c))/(1+np.dot(c,c))
    Bmag.append(np.linalg.norm(B(r)))

    position.append(r)
    velocity.append(v)

    if (i*100/N)%1 == 0:
        print(f'{i*100//N}%')
    if(r[2]*x0[2] <= 0):
        print(f"z = 0 at time t = {i*dt}")

Bmag = np.array(Bmag)
position = np.array(position)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x,y,z = position.T
ax.plot3D(x,y,z)
plt.xlabel('x')
plt.ylabel('y')
ax.set_zlabel('z')
plt.show()

# Plot z vs time graph (magnetic mirror )
plt.plot(t,z/Re)
plt.xlabel("Time")
plt.ylabel("Z/Re")
plt.grid()
plt.show()

# plt.plot(t,Bmag/Bmag[0])
# plt.xlabel("Time")
# plt.ylabel("Bmag/Bmag[0]")
# plt.grid()
# plt.show()

ke = m_e*np.array([np.dot(i,i) for i in velocity])/2/KE
# print(ke)
plt.plot(t,ke)
plt.xlabel("Time")
plt.ylabel("KE/KE[0]")
plt.grid()
plt.show()


from matplotlib import animation

FPS = 10
FRAMES = 100
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def init():
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
# animation function
def animate(i):
    current_index = int(position.shape[0]/FRAMES*i)
    ax.cla()
    ax.plot3D(position[:current_index, 0], 
              position[:current_index, 1], 
              position[:current_index, 2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
# call the animator.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=FRAMES, interval=100)

# Save animation as MP4

# from matplotlib.animation import FFMpegWriter

# writer = FFMpegWriter(fps=FPS, codec='libx264', extra_args=['-pix_fmt', 'yuv420p'])
# anim.save(f"Magnetic_mirror.mp4", writer=writer)

# print(f"Animation saved as magnetic mirror.mp4")

