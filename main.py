from quadrotor import Quadrotor
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":
    quad = Quadrotor()
    path = []

    for i in range(20):
        thrust = np.random.rand(4)
        quad.update(thrust, dt=0.1)
        path.append(quad.pos)

    path = np.array(path)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(path[:,0], path[:,1], path[:,2])
    # ax.axis('equal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    # ax.legend()
    plt.show()