

"""
Generate points on a sphere

https://en.wikipedia.org/wiki/Kent_distribution
https://github.com/edfraenkel/kent_distribution
"""


import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d

def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec

def s1(npoints):
    vec = np.random.randn(3, npoints)
    r = 1.0
    for i in range(npoints):
        theta = np.random.random() * 2.0 * np.pi
        phi = (1 - np.sqrt(np.random.random())) * 0.5 * np.pi
        if np.random.random() > 0.5:
            phi = -phi

        x = r * np.cos(phi) * np.cos(theta)
        y = r * np.cos(phi) * np.sin(theta)
        z = r * np.sin(phi)

        vec[0][i] = x
        vec[1][i] = y
        vec[2][i] = z

    return vec

def s2(npoints):
    vec = np.random.randn(3, npoints)
    r = 1.0
    for i in range(npoints):

        theta = np.random.random() * 2 * np.pi
        phi = np.arcsin(np.random.random() * 2 - 1)

        x = r * np.cos(phi) * np.cos(theta)
        y = r * np.cos(phi) * np.sin(theta)
        z = r * np.sin(phi)

        vec[0][i] = x
        vec[1][i] = y
        vec[2][i] = z

    return vec

def s3(npoints):
    vec = np.random.randn(3, npoints)
    r = 1.0
    for i in range(npoints):
        theta = np.random.random() * 2 * np.pi
        rad = np.sqrt(np.random.random())

        x = r * rad * np.cos(theta)
        y = r * rad * np.sin(theta)
        z = 0

        vec[0][i] = x
        vec[1][i] = y
        vec[2][i] = z

    return vec

def s4(npoints):
    vec = np.random.randn(3, npoints)
    r = 1.0
    for i in range(npoints):
        theta = np.random.random() * 2 * np.pi
        rad = np.arcsin(np.random.random()) / (np.pi / 2)

        x = r * rad * np.cos(theta)
        y = r * rad * np.sin(theta)
        z = 0

        vec[0][i] = x
        vec[1][i] = y
        vec[2][i] = z

    return vec


phi = np.linspace(0, np.pi, 20)
theta = np.linspace(0, 2 * np.pi, 40)
x = np.outer(np.sin(theta), np.cos(phi))
y = np.outer(np.sin(theta), np.sin(phi))
z = np.outer(np.cos(theta), np.ones_like(phi))

# xi, yi, zi = sample_spherical(100)
xi, yi, zi = s2(1000)

fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d', 'aspect':'equal'})
#ax.plot_wireframe(x, y, z, color='k', rstride=1, cstride=1)
ax.scatter(xi, yi, zi, s=1, c='r', zorder=1)

plt.show()
