

"""
Generate mountains

"""

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d


def gaussian_model(matrix, maxvalue, meanvalue, sigma):
    """
    populate a matrix with a gaussian distribution
    :param matrix: the matrix
    :param maxvalue: height
    :param meanvalue: location of the peak within the matrix
    :param sigma: width
    :return: the filled matrix
    """
    return maxvalue * np.exp(-(matrix - meanvalue)**2 / (2 * sigma**2))


# create vectors for the grid
x_vector = lambda grid_column, grid_size: (np.arange(0, grid_size, 1, float) / grid_size) + grid_column
y_vector = lambda grid_row, grid_size: (np.arange(0, grid_size, 1, float) / grid_size) + grid_row


def build_mountain(x_peak, y_peak, height, width, grid_size):
    """
    install a mountains in a circular 2D space. [0, 1[ x [0, 1[
    the space is filled as a square grid
    The mountain is propagated circularly left-right and top-bottom

    :param x_peak: X location of the peak
    :param y_peak: Y location of the peak
    :param height: Peak height
    :param width: Peak width
    :param grid_size:
    :return: the sum of the grid, the filled space grid
    """

    sigma = width / 4.0 / np.sqrt(2.0)

    """
    to make it circular we extend the model to all neighbour regions around until the max value is lower
    than epsilon 
    """

    def create_grid(g_row, g_column):
        vx = x_vector(g_column, grid_size)
        vy = y_vector(g_row, grid_size)
        vy = vy[:, np.newaxis]  # transpose y

        new_grid = gaussian_model(vx, height, x_peak, sigma) * gaussian_model(vy, height, y_peak, sigma)

        # print("row=", row, "column=", column, "h=", grid_sum, grid, x_peak, y_peak)

        return new_grid, np.sum(new_grid)

    radius = 0
    central_sum = None
    central_grid = None
    while True:
        # print("radius=", radius)
        if radius == 0:
            central_grid, central_sum = create_grid(0, 0)
        else:
            max_sum = 0

            # bottom line
            row = -radius
            for column in range(-radius, radius+1):
                local_grid, local_sum = create_grid(row, column)
                central_grid += local_grid
                max_sum = max([local_sum, max_sum])

            column = -radius
            for row in range(-radius + 1, radius):
                local_grid, local_sum = create_grid(row, column)
                central_grid += local_grid
                max_sum = max([local_sum, max_sum])

            column = radius
            for row in range(-radius + 1, radius):
                local_grid, local_sum = create_grid(row, column)
                central_grid += local_grid
                max_sum = max([local_sum, max_sum])

            row = radius
            for column in range(-radius, radius+1):
                local_grid, local_sum = create_grid(row, column)
                central_grid += local_grid
                max_sum = max([local_sum, max_sum])

            if max_sum < (central_sum/10000):
                # print("Radius=", radius)
                break

        radius += 1

    return central_grid


class Object(object):
    def __init__(self, object_id, object_x, object_y):
        self.id = object_id
        self.x = object_x
        self.y = object_y


space = None
grid = 1000
for i in range(40):
    x0 = np.random.random()
    y0 = np.random.random()
    if i % 10 == 0:
        print("install expansion zone", i, x0, y0)

    z = build_mountain(x_peak=x0,
                       y_peak=y0,
                       height=np.random.random()/2,
                       width=np.random.random()/2,
                       grid_size=grid)
    if space is None:
        space = z
    else:
        space += z

fig, axe1 = plt.subplots(1, 1, subplot_kw={'projection': '3d', 'aspect': 'equal'})
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

axe1.plot_surface(x_vector(0, grid), y_vector(0, grid)[:, np.newaxis], space, color='r')
plt.show()

axe2 = plt.subplot2grid((1, 1), (0, 0))

Os = []
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

    o = Object(object_id=n, object_x=x, object_y=y)
    Os.append(o)

    ox.append(x)
    oy.append(y)

    n += 1

    if (n % 100) == 0:
        print("generating", n)
        axe2.scatter(ox, oy, color='k', s=1)
        plt.pause(0.00001)

plt.show()
