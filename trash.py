# coding: utf8
import numpy as np
import backprop as biM
import multiprocessing as mp
import time
import uuid
from numba import jit
import checkSolution as cs
import os
import matplotlib.pyplot as plt
import math as mt
np.set_printoptions(precision=2, suppress=True)


#

r = 100000
x = [i*100 for i in range(r)]
y1 = [1/2*x[i]**3+x[i]**2 for i in range(r)]
y2 = [x[i]**3 for i in range(r)]
y3 = [x[i]**(mt.log(7)/mt.log(2)) for i in range(r)]
y4 = [x[i]**2.375477 for i in range(r)]

plt.plot(x, y1, color='blue')
plt.plot(x, y2, color='red')
plt.plot(x, y3, color='green')
plt.plot(x, y4, color='black')
