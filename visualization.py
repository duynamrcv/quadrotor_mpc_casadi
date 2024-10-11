from parameters import *
import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

with open(FILE_NAME, 'rb') as file:
    data = pickle.load(file)

path = data['path']
ref = data['ref']
# print(ref[:,0])
# print(ref.shape)
# print(path.shape)

fig = plt.figure()
ax = fig.add_subplot() #projection='3d')
ax.plot(ref[:,0], ref[:,1])
ax.plot(path[:,0], path[:,1])
# ax.axis('equal')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
# ax.set_zlabel('z [m]')
# ax.legend()
plt.show()