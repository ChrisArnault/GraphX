import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from sklearn.datasets.samples_generator import make_blobs
import math


T = 50.0
N = 100

sh = math.sqrt(T/N)

X = np.zeros(N)
Y = np.zeros(N)

x = 0
y = 0

for i in range(N):
    x += sh * np.random.normal(0, sh, 1)
    y += sh * np.random.normal(0, sh, 1)
    X[i] = x
    Y[i] = y
    # print(x, y)


plt.plot(X, Y)

plt.show()
