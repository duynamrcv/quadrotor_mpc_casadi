from quadrotor import Quadrotor
from controller import Controller
from parameters import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":
    quad = Quadrotor()
    controller = Controller(quad, n_nodes=N, dt=DT)
    path = []
    ref = []

    cur_time = 0
    total_time = 20
    iter = 0
    while(total_time > cur_time):
        x_ref = get_reference(cur_time, N, DT)
        ref.append(x_ref[0,:])
        thrust = controller.compute_control_signal(x_ref)
        # print(thrust)
        quad.update(thrust, dt=DT)
        path.append(quad.pos)
        cur_time += DT

    path = np.array(path)
    ref = np.array(ref)
    
    fig = plt.figure()
    ax = fig.add_subplot() # projection='3d'q
    ax.plot(ref[:,1], ref[:,2])
    ax.plot(path[:,0], path[:,1])
    # ax.axis('equal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    # ax.set_zlabel('z [m]')
    # ax.legend()
    plt.show()