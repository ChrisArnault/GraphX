

"""
Generate mountains

"""


import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d


def gaussian_model(x, maxvalue, meanvalue, sigma):
    return maxvalue * np.exp(-(x - meanvalue)**2 / (2 * sigma**2))


def build_mountain(x0, y0, height, width, grid):
    sigma = width / 4.0 / np.sqrt(2.0)

    """
    to make it circular we extend the model to all neighbour regions around until the max value is lower
    than epsilon 
    """

    do_x = lambda column: (np.arange(0, grid, 1, float) / grid) + column
    do_y = lambda row: (np.arange(0, grid, 1, float) / grid) + row

    def line(row, column):
        x = do_x(column)
        y = do_y(row)
        y = y[:, np.newaxis]  # transpose y

        gx = gaussian_model(x, height, x0, sigma)
        gy = gaussian_model(y, height, y0, sigma)
        z = gx * gy

        h = np.sum(z)
        # print("row=", row, "column=", column, "h=", h, z, x0, y0)
        return h, z

    radius = 0
    H = None
    z = None
    while True:
        # print("radius=", radius)
        if radius == 0:
            h, z = line(0, 0)
            H = h
        else:
            max_h = 0

            ok = False
            row = -radius
            for column in range(-radius, radius+1):
                h, zz = line(row, column)
                z += zz
                max_h = max([h, max_h])

            column = -radius
            for row in range(-radius + 1, radius):
                h, zz = line(row, column)
                z += zz
                max_h = max([h, max_h])

            column = radius
            for row in range(-radius + 1, radius):
                h, zz = line(row, column)
                z += zz
                max_h = max([h, max_h])

            row = radius
            for column in range(-radius, radius+1):
                h, zz = line(row, column)
                z += zz
                max_h = max([h, max_h])

            if max_h < (H/10000):
                # print("Radius=", radius)
                break

        radius += 1

    x = do_x(0)
    y = do_y(0)
    y = y[:, np.newaxis]  # transpose y
    return x, y, z

space = None
grid = 1000
for i in range(400):
    x0 = np.random.random()
    y0 = np.random.random()
    if i % 10 == 0:
        print("install expansion zone", i, x0, y0)
    x, y, z = build_mountain(x0=x0,
                             y0=y0,
                             height=np.random.random()/2,
                             width=np.random.random()/2,
                             grid=grid)
    if space is None:
        space = z
    else:
        space += z

#axe1 = plt.subplot2grid((1, 2), (0, 0))
#axe2 = plt.subplot2grid((1, 2), (0, 1))

fig, axe1 = plt.subplots(1, 1, subplot_kw={'projection':'3d', 'aspect':'equal'})
axe2 = plt.subplot2grid((1, 1), (0, 0))
# ax.plot_wireframe(x, y, space, color='r')
# ax.scatter(xi, yi, zi, s=1, c='r', zorder=1)

print("space range initial", np.min(space), np.max(space))
space -= np.min(space)
space /= np.max(space)
print("normalized space range", np.min(space), np.max(space))
space = 1 - space
print("inverse space range", np.min(space), np.max(space), space.shape)
space *= space
space *= space
space *= space

axe1.plot_surface(x, y, space, color='r')

ox = []
oy = []

n = 0
for i in range(1000000):
    x = int(np.random.random() * grid)
    y = int(np.random.random() * grid)
    z = space[y, x]
    p = np.random.random()
    if p > z:
        continue

    ox.append(x)
    oy.append(y)

    n += 1

    if (n % 100) == 0:
        print("generating", n)
        axe2.scatter(ox, oy, color='k', s=1)
        plt.pause(0.00001)



plt.show()
