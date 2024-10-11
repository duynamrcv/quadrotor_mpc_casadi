from quadrotor import Quadrotor
from controller import Controller
from parameters import *
import numpy as np
import pickle
import logging, coloredlogs

coloredlogs.install()

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
        ref.append(x_ref[:,0])
        thrust = controller.compute_control_signal(x_ref)
        # logging.info("Thrust value [0,1]: {}\t{}\t{}\t{}".format(thrust[0], thrust[1], thrust[2], thrust[3]))
        quad.update(thrust, dt=DT)
        # print(quad.pos)
        path.append(quad.pos)
        cur_time += DT


    with open(FILE_NAME, 'wb') as file:
        path = np.array(path)
        ref = np.array(ref)
        data = dict()
        data['path'] = path
        data['ref'] = ref
        pickle.dump(data, file)