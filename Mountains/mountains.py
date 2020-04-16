

"""
Generate mountains

"""

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d


space = None
"""
we create a base grid 
then we divide the base grid in a grid of cells
by definition: cell_grid_size < grid
"""
base_grid_size = 1000
cell_grid_size = 100


def cell_id(grid_size, x, y):
    """ compute a cell id from a x, y coordinate """
    return int(x * grid_size) + grid_size * int(y * grid_size)

def x_cell(cell_id, grid_size):
    """ compute the x column from the cell_id """
    return cell_id % grid_size


def y_cell(cell_id, grid_size):
    """ compute the y row from the cell_id """
    return cell_id / grid_size


def neighbour(grid_size, cell1, cell2):
    """
    test if the two cells (cell1 & cell2) are neighbours inside the square grid with grid_size
    neighbouring is true if cells are close by 1 left-right , top-bottom, or by corners
    the grid is cyclic by left-right , top-bottom, or by corners
    """
    t1 = cell1 == cell2

    dx = abs(x_cell(cell_id=cell1, grid_size=grid_size) - x_cell(cell_id=cell2, grid_size=grid_size))
    dy = abs(y_cell(cell_id=cell1, grid_size=grid_size) - y_cell(cell_id=cell2, grid_size=grid_size))

    t2 = (dx == 0) & (dy == 1)
    t3 = (dx == 0) & (dy == grid_size - 1)
    t4 = (dx == 1) & (dy == 0)
    t5 = (dx == grid_size - 1) & (dy == 0)

    t6 = (dx == 1) & (dy == 1)
    t7 = (dx == 1) & (dy == grid_size - 1)
    t8 = (dx == grid_size - 1) & (dy == 1)
    t9 = (dx == grid_size - 1) & (dy == grid_size - 1)

    # print(t1, t2, t3, t4, t5, t6, t7, t8, t9)
    return t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9


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

def cell_iterator(initialize, function, test_stop):
    radius = 0
    while True:
        # print("radius=", radius)
        if radius == 0:
            initialize()
        else:
            # bottom line
            row = -radius
            for column in range(-radius, radius + 1):
                function()

            column = -radius
            for row in range(-radius + 1, radius):
                function()

            column = radius
            for row in range(-radius + 1, radius):
                function()

            row = radius
            for column in range(-radius, radius+1):
                function()

            if test_stop():
                break

        radius += 1

    return radius


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


class MyIterator(CellIterator):
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
        # print("myinitialize>", self.radius, "col=", self.column, "row=", self.row)
        self.central_grid, self.central_sum = self.create_grid(0, 0)

    def iterate(self):
        # print("myiterate>", self.radius, "col=", self.column, "row=", self.row, "max_sum=", self.max_sum)
        self.local_grid, self.local_sum = self.create_grid(self.row, self.column)
        self.central_grid += self.local_grid
        self.max_sum = max([self.local_sum, self.max_sum])

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



class Object(object):
    def __init__(self, object_id, object_x, object_y):
        self.id = object_id
        self.x = object_x
        self.y = object_y
        self.cell_id = cell_id(grid_size=cell_grid_size,
                               x=self.x % cell_grid_size,
                               y=self.y % cell_grid_size)

    def dist(self, other):
        return np.sqrt(np.power(self.x - other.x, 2) + np.power(self.y - other.y, 2))


for i in range(40):
    x0 = np.random.random()
    y0 = np.random.random()
    if i % 10 == 0:
        print("install expansion zone", i, x0, y0)

    cell_iterator = MyIterator(x_peak=x0,
                               y_peak=y0,
                               height=np.random.random()/2,
                               width=np.random.random()/2,
                               grid_size=base_grid_size)

    z = cell_iterator.run()

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

axe1.plot_surface(x_vector(0, base_grid_size), y_vector(0, base_grid_size)[:, np.newaxis], space, color='r')
plt.show()

axe2 = plt.subplot2grid((1, 1), (0, 0))

Objects = []
ox = []
oy = []

n = 0
for i in range(1000000):
    x = int(np.random.random() * base_grid_size)
    y = int(np.random.random() * base_grid_size)
    z = space[y, x]
    p = np.random.random()
    if p > z:
        continue

    o = Object(object_id=n, object_x=x, object_y=y)
    Objects.append(o)

    """
    we now look for neighbour objects:
    - we only consider neighbour cells up to a max cell distance (in cell measure)
    - and we connect neighbour objects up to a max space distance (in space measure)
    """

    ox.append(x)
    oy.append(y)

    n += 1

    if (n % 100) == 0:
        print("generating", n)
        axe2.scatter(ox, oy, color='k', s=1)
        plt.pause(0.00001)

plt.show()
