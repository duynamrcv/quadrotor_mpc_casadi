from parameters import *
import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

with open("results/data_False.txt", 'rb') as file:
    data = pickle.load(file)
path0 = data['path']

with open("results/data_True.txt", 'rb') as file:
    data = pickle.load(file)
path1 = data['path']
ref = data['ref']
# print(ref[:,0])
# print(ref.shape)
# print(path.shape)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(ref[:,0], ref[:,1], ref[:,2], label="reference")
ax.plot(path0[:,0], path0[:,1], path0[:,2], label="quadrotor_False")
ax.plot(path1[:,0], path1[:,1], path1[:,2], label="quadrotor_True")
# ax.axis('equal')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')
ax.legend()
plt.show()