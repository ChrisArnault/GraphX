

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
space_grid_size = 1000   # division of space in points

cell_grid_size = 100   # division of space in cells
cell_width = space_grid_size / cell_grid_size # cell width in space points
distance_max = 0.01


def cell_id(grid_size, _x, _y):
    """ compute a cell id from a x, y coordinate """
    return int(_x * grid_size) + grid_size * int(_y * grid_size)


def x_cell_from_id(_cell_id, grid_size):
    """ compute the x column from the cell_id """
    return int(_cell_id % grid_size)


def y_cell_from_id(_cell_id, grid_size):
    """ compute the y row from the cell_id """
    return int(_cell_id / grid_size)


def x_cell_from_coord(_x, grid_size):
    """ compute the x column from the cell_id """
    return _x * grid_size


def y_cell_from_coord(_y, grid_size):
    """ compute the y row from the cell_id """
    return _y * grid_size


def neighbour(grid_size, cell1, cell2):
    """
    test if the two cells (cell1 & cell2) are neighbours inside the square grid with grid_size
    neighbouring is true if cells are close by 1 left-right , top-bottom, or by corners
    the grid is cyclic by left-right , top-bottom, or by corners
    """
    t1 = cell1 == cell2

    xc1 = x_cell_from_id(_cell_id=cell1, grid_size=grid_size)
    xc2 = x_cell_from_id(_cell_id=cell2, grid_size=grid_size)
    
    yc1 = y_cell_from_id(_cell_id=cell1, grid_size=grid_size)
    yc2 = y_cell_from_id(_cell_id=cell2, grid_size=grid_size)

    dx = abs(xc1 - xc2)
    dy = abs(yc1 - yc2)

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


class ObjectConnector(CellIterator):
    def __init__(self, dist_max, cell0):
        """
        """
        CellIterator.__init__(self)
        self.dist_max = dist_max
        self.cell0 = int(cell0)
        self.cell_column0 = 0
        self.cell_row0 = 0
        self.cell = self.cell0
        self.cell_column = 0
        self.cell_row = 0
        self.step = 0
        self.largest_distance = 0

    def locate_cell(self):
        self.cell_column = int(self.cell_column0 + self.column)
        self.cell_row = int(self.cell_row0 + self.row)

        column = self.cell_column % cell_grid_size
        row = self.cell_row % cell_grid_size

        self.cell = column + cell_grid_size * row

    def distance(self):
        cell_dist = np.sqrt(np.power(self.cell_column - self.cell_column0, 2) + np.power(self.cell_row - self.cell_row0, 2))
        dist = cell_dist * cell_width / space_grid_size
        return dist

    def initialize(self):
        self.cell_column0 = x_cell_from_id(self.cell0, cell_grid_size)
        self.cell_row0 = y_cell_from_id(self.cell0, cell_grid_size)
        self.locate_cell()
        self.largest_distance = self.distance()
        self.step = 0

        print("=============================init cell=", self.cell, "x=", self.cell_column, "y=", self.cell_row, "step=", self.step)

        if self.cell in Objects:
            objects_in_cell = Objects[self.cell]
            print("look for all objects in cell", self.cell, len(objects_in_cell))


    def iterate(self):
        self.step += 1
        self.locate_cell()
        dist = self.distance()
        self.largest_distance = max(dist, self.largest_distance)

        """
        print("iterate cell=", self.cell,
              "cell col=", self.cell_column, "cell row=", self.cell_row,
              "col=", self.column, "row=", self.row, "dist=", dist, self.largest_distance)
        """

    def test_stop(self):
        self.locate_cell()
        # print("===== test stop cell=", self.cell, "x=", self.cell_column, "y=", self.cell_row, "step=", self.step, "largest dist=", self.largest_distance)
        return self.largest_distance > (2 * self.dist_max)

    def run(self):
        CellIterator.run(self)


Objects = dict()


class Object(object):
    def __init__(self, object_id, object_x, object_y):
        self.id = object_id
        self.x = object_x
        self.y = object_y
        self.cell_id = cell_id(grid_size=cell_grid_size, _x=self.x, _y=self.y)

    def dist(self, other):
        return np.sqrt(np.power(self.x - other.x, 2) + np.power(self.y - other.y, 2))


for i in range(40):
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

axe1.plot_surface(x_vector(0, space_grid_size), y_vector(0, space_grid_size)[:, np.newaxis], space, color='r')
# plt.show()

axe2 = plt.subplot2grid((1, 1), (0, 0))

ox = []
oy = []

n = 0
for i in range(1000000):
    x = np.random.random()
    y = np.random.random()
    z = space[int(y * space_grid_size), int(x * space_grid_size)]
    p = np.random.random()
    if p > z:
        continue

    o = Object(object_id=n, object_x=x, object_y=y)


    if o.cell_id in Objects:
        objects_in_cell = Objects[o.cell_id]
    else:
        objects_in_cell = []

    print("cell for object", o.cell_id, len(objects_in_cell))

    objects_in_cell.append(o)
    Objects[o.cell_id] = objects_in_cell

    connector = ObjectConnector(dist_max=distance_max, cell0=o.cell_id)
    connector.run()

    """
    we now look for neighbour objects:
    - we only consider neighbour cells up to a max cell distance (in cell measure)
    - and we connect neighbour objects up to a max space distance (in space measure)
    """

    ox.append(x)
    oy.append(y)

    n += 1

    if (n % 1000) == 0:
        print("generating", n)
        axe2.scatter(ox, oy, color='k', s=1)
        plt.pause(0.00001)

plt.show()
