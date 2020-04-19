

"""
Generate mountains

"""

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d

import mountain_space

"""
Now the space is filled with a repulsive field
We start populating it with objects
"""


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


class ObjectConnector(mountain_space.CellIterator):
    def __init__(self, dist_max, obj):
        mountain_space.CellIterator.__init__(self)
        self.dist_max = dist_max
        self.object = obj
        self.cell0 = obj.cell_id
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
        cell_dist = np.sqrt(np.power(self.cell_column - self.cell_column0, 2) +
                            np.power(self.cell_row - self.cell_row0, 2))
        # dist = cell_dist * cell_width / space_grid_size
        dist = cell_dist * cell_width
        return dist

    def connect(self):
        if self.cell not in Objects:
            # print("connect> current cell not in Objects", self.cell)
            return

        objects_in_cell = Objects[self.cell]
        # print("look for all objects in cell", self.cell, len(objects_in_cell))
        for _o in objects_in_cell:
            if self.object.id == _o.id:
                continue
            if self.object.dist(_o) > distance_max:
                print("connect> current object", self.object.id, "not neighbour of", _o.id)
                continue
            print("connect o=", self.object.id, self.object.x, self.object.y, "to=", _o.id, _o.x, _o.y)
            edgex = (self.object.x, _o.x)
            edgey = (self.object.y, _o.y)
            self.axe.plot(edgex, edgey, "r")

            self.object.connect(_o)

    def initialize(self):
        self.cell_column0 = x_cell_from_id(self.cell0, cell_grid_size)
        self.cell_row0 = y_cell_from_id(self.cell0, cell_grid_size)
        self.locate_cell()
        self.largest_distance = self.distance()
        self.step = 0
        self.connect()

        """
        print("=============================init cell=", self.cell,
              "x=", self.cell_column, "y=", self.cell_row,
              "step=", self.step)
        """


    def iterate(self):
        self.step += 1
        self.locate_cell()
        dist = self.distance()
        self.largest_distance = max(dist, self.largest_distance)
        self.connect()

        print("iterate cell=", self.cell,
              "cell col=", self.cell_column, "cell row=", self.cell_row,
              "col=", self.column, "row=", self.row, "dist=", dist, self.largest_distance)

    def test_stop(self):
        self.locate_cell()
        # print("===== test stop cell=", self.cell, "x=", self.cell_column, "y=", self.cell_row,
        # "step=", self.step, "largest dist=", self.largest_distance)
        return self.largest_distance > (2 * self.dist_max)

    def run(self, axe):
        self.axe = axe
        mountain_space.CellIterator.run(self)


Objects = dict()


class Object(object):
    def __init__(self, object_id, object_x, object_y):
        self.id = object_id
        self.x = object_x
        self.y = object_y
        self.cell_id = cell_id(grid_size=cell_grid_size, _x=self.x, _y=self.y)
        self.declare()
        self.edges = []

    def declare(self):
        if self.cell_id in Objects:
            objects_in_cell = Objects[o.cell_id]
        else:
            objects_in_cell = []

        # print("cell for object cell=", o.cell_id, len(objects_in_cell))

        objects_in_cell.append(self)
        Objects[self.cell_id] = objects_in_cell

    def dist(self, other):
        return np.sqrt(np.power(self.x - other.x, 2) + np.power(self.y - other.y, 2))

    def connect(self, other):
        self.edges.append(other)


def interpolate(from_space, _x, _y):
    """
    Compute the field @ x, y by interpolating the quantized field in space
    """
    g = from_space.shape[0]
    x1 = int(_x*g)
    y1 = int(_y*g)
    xx2 = (x1 + 1) % g
    yy2 = (y1 + 1) % g

    z1 = from_space[y1, x1]
    z2 = from_space[y1, xx2]
    z3 = from_space[yy2, x1]

    b = z2 - z1
    a = z3 - z1
    c = z1 - a*x1 - b*y1

    _z = (a*_x*g + b*_y*g + c)

    # print("interpolation ", x, y, "x, y=", x1, y1, "xx, yy=", xx2, yy2, "zi=", z1, z2, z3, z4, "z=", z)

    return _z


space = mountain_space.build_space(fields=40, space_grid_size=100)
space_grid_size = space.shape[0]

cell_grid_size = 100                              # division of space in cells
cell_width = space_grid_size / cell_grid_size   # cell width in space points
distance_max = cell_width

fig, axe1 = plt.subplots(1, 1, subplot_kw={'projection': '3d', 'aspect': 'equal'})
# ax.plot_wireframe(x, y, space, color='r')
# ax.scatter(xi, yi, zi, s=1, c='r', zorder=1)

axe1.plot_surface(mountain_space.x_vector(0, space_grid_size),
                  mountain_space.y_vector(0, space_grid_size)[:, np.newaxis], space, color='r')
# plt.show()

axe2 = plt.subplot2grid((1, 1), (0, 0))

ox = []
oy = []

n = 0
for i in range(1000000):
    x = np.random.random()
    y = np.random.random()
    z = interpolate(from_space=space, _x=x, _y=y)
    p = np.random.random()
    if p > z:
        continue

    o = Object(object_id=n, object_x=x, object_y=y)

    connector = ObjectConnector(dist_max=distance_max, obj=o)
    connector.run(axe2)

    """
    we now look for neighbour objects:
    - we only consider neighbour cells up to a max cell distance (in cell measure)
    - and we connect neighbour objects up to a max space distance (in space measure)
    """

    ox.append(x)
    oy.append(y)

    n += 1

    if (n % 10) == 0:

        for row in range(cell_grid_size):
            axe2.plot((0, 1), (row/cell_grid_size, row/cell_grid_size), "g")
            for col in range(cell_grid_size):
                axe2.plot((col / cell_grid_size, col / cell_grid_size), (0, 1), "g")

        print("generating", n)
        axe2.scatter(ox, oy, color='k', s=1)
        plt.pause(0.1)

plt.show()
