

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


class CellIterator(object):
    def __init__(self):
        self.radius = 0
        self.row = 0
        self.column = 0

    def initialize(self):
        print("initialize>", self.radius)

    def iterate(self):
        print("iterate>", self.radius)

    def test_stop(self):
        print("test_stop>", self.radius)
        return True

    def run(self):
        self.radius = 0
        while True:
            # print("radius=", radius)
            if self.radius == 0:
                self.initialize()
            else:
                # bottom line
                self.row = -self.radius
                for self.column in range(-self.radius, self.radius + 1):
                    self.iterate()

                self.column = -self.radius
                for self.row in range(-self.radius + 1, self.radius):
                    self.iterate()

                self.column = self.radius
                for self.row in range(-self.radius + 1, self.radius):
                    self.iterate()

                self.row = self.radius
                for self.column in range(-self.radius, self.radius + 1):
                    self.iterate()

                if self.test_stop():
                    break

            self.radius += 1

        return self.radius


class MountainBuilder(CellIterator):
    def __init__(self, x_peak, y_peak, height, width, grid_size):
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

        CellIterator.__init__(self)

        self.x_peak = x_peak
        self.y_peak = y_peak
        self.height = height
        self.width = width
        self.grid_size = grid_size
        self.sigma = width / 4.0 / np.sqrt(2.0)

        self.epsilon = 1.0/1000
        self.central_sum = None
        self.max_sum = 0
        self.central_grid = None

    def initialize(self):
        self.central_grid, self.central_sum = self.create_grid(0, 0)

    def iterate(self):
        # print("myiterate>", self.radius, "col=", self.column, "row=", self.row, "max_sum=", self.max_sum)
        local_grid, local_sum = self.create_grid(self.row, self.column)
        self.central_grid += local_grid
        self.max_sum = max([local_sum, self.max_sum])

    def test_stop(self):
        # print("mytest_stop>", self.radius, "col=", self.column, "row=", self.row)
        ok = self.max_sum < (self.central_sum * self.epsilon)
        self.max_sum = 0
        return ok

    def run(self):
        CellIterator.run(self)
        return self.central_grid

    def create_grid(self, g_row, g_column):
        """
        to make it circular we extend the model to all neighbour regions around until the max value is lower
        than epsilon
        """

        vx = x_vector(g_column, self.grid_size)
        vy = y_vector(g_row, self.grid_size)
        vy = vy[:, np.newaxis]  # transpose y

        gx = gaussian_model(vx, self.height, self.x_peak, self.sigma)
        gy = gaussian_model(vy, self.height, self.y_peak, self.sigma)
        new_grid = gx * gy

        # print("row=", row, "column=", column, "h=", grid_sum, grid, x_peak, y_peak)

        return new_grid, np.sum(new_grid)


def build_space(fields, space_grid_size):

    """
    Construction of the cyclic spherical space, continuous left-riht & top-bottom
    install a set of repulsive fields
    each field is gaussian shaped, with random parameters (location, height, width)
    """

    space = None

    for i in range(fields):
        x0 = np.random.random()
        y0 = np.random.random()
        if i % 10 == 0:
            print("install expansion zone", i, x0, y0)

        builder = MountainBuilder(x_peak=x0,
                                  y_peak=y0,
                                  height=np.random.random()/2,
                                  width=np.random.random(),
                                  grid_size=space_grid_size)

        z = builder.run()

        if space is None:
            space = z
        else:
            space += z

    print("space range initial", np.min(space), np.max(space))
    space -= np.min(space)
    space /= np.max(space)
    print("normalized space range", np.min(space), np.max(space))
    space = 1 - space
    print("inverse space range", np.min(space), np.max(space), space.shape)
    space *= space
    space *= space
    space *= space

    return space

