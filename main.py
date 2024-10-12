from quadrotor import Quadrotor
from controller import Controller
from parameters import *
import numpy as np
import pickle
import time

if __name__ == "__main__":
    quad = Quadrotor()
    controller = Controller(quad, n_nodes=N, dt=DT)
    path = []
    ref = []
    times = []

    cur_time = 0
    total_time = 60
    iter = 0
    while(total_time > cur_time):
        x0 = np.concatenate(quad.get_state())
        x_ref = get_reference(cur_time, x0, N, DT)
        ref.append(x_ref[:,1])
        start = time.time()
        thrust = controller.compute_control_signal(x_ref)
        times.append(time.time() - start)
        
        # logging.info("Thrust value [0,1]: {}\t{}\t{}\t{}".format(thrust[0], thrust[1], thrust[2], thrust[3]))
        quad.update(thrust, dt=DT)
        # print(quad.pos)
        path.append(quad.pos)
        cur_time += DT


    with open(FILE_NAME, 'wb') as file:
        path = np.array(path)
        ref = np.array(ref)
        times = np.array(times)
        print("Max processing time: {:.4f}s".format(times.max()))
        print("Min processing time: {:.4f}s".format(times.min()))
        print("Mean processing time: {:.4f}s".format(times.mean()))
        data = dict()
        data['path'] = path
        data['ref'] = ref
        data['times'] = times
        pickle.dump(data, file)