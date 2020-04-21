

"""
Generate mountains

"""

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d

import mountain_cells
import mountain_space

"""
Now the space is filled with a repulsive field
We start populating it with objects
"""


def cell_id(cells, _x, _y):
    """ compute a cell id from a x, y coordinate """

    _row = int(_y * cells)
    _col = int(_x * cells)
    return _col + cells * _row


def cell_id_from_rc(cells, _col, _row):
    """ compute a cell id from a x, y coordinate """

    return (_col % cells) + cells * (_row % cells)


def col_from_cell_id(cells, _cell_id):
    """ compute the x column from the cell_id """
    return int(_cell_id % cells)


def row_from_cell_id(cells, _cell_id):
    """ compute the y row from the cell_id """
    return int(_cell_id / cells)


def neighbour(cells, cell1, cell2):
    """
    test if the two cells (cell1 & cell2) are neighbours inside the square grid with cells
    neighbouring is true if cells are close by 1 left-right , top-bottom, or by corners
    the grid is cyclic by left-right , top-bottom, or by corners
    """
    t1 = cell1 == cell2
    if t1:
        return True

    xc1 = col_from_cell_id(cells=cells, _cell_id=cell1)
    xc2 = col_from_cell_id(cells=cells, _cell_id=cell2)

    yc1 = row_from_cell_id(cells=cells, _cell_id=cell1)
    yc2 = row_from_cell_id(cells=cells, _cell_id=cell2)

    dx = abs(xc1 - xc2)
    dy = abs(yc1 - yc2)

    t2 = (dx == 0) & (dy == 1)
    if t2:
        return True
    t3 = (dx == 0) & (dy == cells - 1)
    if t3:
        return True
    t4 = (dx == 1) & (dy == 0)
    if t4:
        return True
    t5 = (dx == cells - 1) & (dy == 0)
    if t5:
        return True

    t6 = (dx == 1) & (dy == 1)
    if t6:
        return True
    t7 = (dx == 1) & (dy == cells - 1)
    if t7:
        return True
    t8 = (dx == cells - 1) & (dy == 1)
    if t8:
        return True

    t9 = (dx == cells - 1) & (dy == cells - 1)
    if t9:
        return True

    # print(t1, t2, t3, t4, t5, t6, t7, t8, t9)
    return False


lines = []


def show_cell(cells, col, row):
    global lines

    cell_width = 1.0/cells
    w = cell_width
    x = (col % cells) * w
    y = (row % cells) * w
    # print('col=', self.column, 'row=', self.row, 'lcol=', self.local_col, 'lrow=', self.local_row)

    epsilon = w/20.
    a = plt.plot((x+epsilon, x+w-epsilon, x+w-epsilon, x+epsilon, x+epsilon),
                 (y+epsilon, y+epsilon, y+w-epsilon, y+w-epsilon, y+epsilon), 'r')

    plt.pause(0.0001)

    lines.append(a)


def erase_cells():
    global lines
    for line in lines:
        for s in line:
            s.remove()
    lines = []

def draw_connector(x1, y1, x2, y2, case):
    X = x2
    Y = y2
    if case == 0:
        pass
    elif case == 1:
        X += 1
    elif case == 2:
        X += 1
        Y += 1
    elif case == 3:
        Y += 1
    elif case == 4:
        X -= 1
        Y += 1
    elif case == 5:
        X -= 1
    elif case == 6:
        X -= 1
        Y -= 1
    elif case == 7:
        Y -= 1
    elif case == 8:
        X += 1
        Y -= 1

    a = (y1 - Y) / (x1 - X)
    b = y1 - (a * x1)

    if case == 0:
        plt.plot((x1, x2), (y1, y2), 'g')
    elif case == 1:
        xc = 1
        yc = a*xc + b
        plt.plot((x1, 1), (y1, yc), 'g')
        plt.plot((0, x2), (yc, y2), 'g')
    elif case == 2:
        xc = 1
        yc = a*xc + b
        if yc >= 1 or yc < 0:
            yc = 1
            xc = (yc - b)/a
            plt.plot((x1, xc), (y1, 1), 'g')
            plt.plot((xc, x2), (0, y2), 'g')
        else:
            plt.plot((x1, 1), (y1, yc), 'g')
            plt.plot((0, x2), (yc, y2), 'g')
    elif case == 3:
        yc = 1
        xc = (yc - b)/a
        plt.plot((x1, xc), (y1, 1), 'g')
        plt.plot((xc, x2), (0, y2), 'g')
    elif case == 4:
        xc = 0
        yc = a*xc + b
        if yc >= 1 or yc < 0:
            yc = 1
            xc = (yc - b)/a
            plt.plot((x1, xc), (y1, 1), 'g')
            plt.plot((xc, x2), (0, y2), 'g')
        else:
            plt.plot((x1, 0), (y1, yc), 'g')
            plt.plot((1, x2), (yc, y2), 'g')
    elif case == 5:
        xc = 0
        yc = a*xc + b
        plt.plot((x1, 0), (y1, yc), 'g')
        plt.plot((1, x2), (yc, y2), 'g')
    elif case == 6:
        xc = 0
        yc = a*xc + b
        if yc >= 1 or yc < 0:
            yc = 0
            xc = (yc - b)/a
            plt.plot((x1, xc), (y1, 0), 'g')
            plt.plot((xc, x2), (1, y2), 'g')
        else:
            plt.plot((x1, 0), (y1, yc), 'g')
            plt.plot((1, x2), (yc, y2), 'g')
    elif case == 7:
        yc = 0
        xc = (yc - b)/a
        plt.plot((x1, xc), (y1, 0), 'g')
        plt.plot((xc, x2), (1, y2), 'g')
    elif case == 8:
        xc = 1
        yc = a*xc + b
        if yc >= 1 or yc < 0:
            yc = 0
            xc = (yc - b)/a
            plt.plot((x1, xc), (y1, 0), 'g')
            plt.plot((xc, x2), (1, y2), 'g')
        else:
            plt.plot((x1, 1), (y1, yc), 'g')
            plt.plot((0, x2), (yc, y2), 'g')


class ObjectConnector(mountain_cells.CellIterator):
    def __init__(self, cells, obj, limit):
        # print("==================================================================================")
        mountain_cells.CellIterator.__init__(self)
        self.cells = cells
        self.cell_width = 1.0/self.cells
        self.limit = limit
        self.max_dist = 0
        self.object = obj
        self.start_x = obj.x
        self.start_y = obj.y

        self.start_col = int(self.start_x * self.cells)
        self.start_row = int(self.start_y * self.cells)

        self.cell_id = self.start_row * self.cells + self.start_col

        self.local_col = self.start_col
        self.local_row = self.start_row

        erase_cells()

    def dist(self):
        return np.sqrt(np.power((self.local_col - self.start_col) * self.cell_width, 2) +
                       np.power((self.local_row - self.start_row) * self.cell_width, 2))

    def connect(self):
        if self.cell_id not in Objects:
            # print("connect> current cell not in Objects", self.cell)
            return

        objects_in_cell = Objects[self.cell_id]
        # print("look for all objects in cell", self.cell_id, "len=", len(objects_in_cell))
        # plt.pause(5)
        for _o in objects_in_cell:
            if self.object.id == _o.id:
                continue
            # a = plt.scatter(self.object.x, self.object.y, color='y', s=40)
            # b = plt.scatter(_o.x, _o.y, color='b', s=40)
            d, indices = self.object.dist(_o)
            if d > self.limit:
                # print("connect> current object", self.object.id, "not neighbour of", _o.id, "d=", d, "indices=", indices)
                # plt.pause(1)
                # a.remove()
                # b.remove()
                continue
            # print("connect o=", self.object.id, self.object.x, self.object.y, "to=", _o.id, _o.x, _o.y, "d=", d, "indices=", indices)
            case = indices[0]
            draw_connector(self.object.x, self.object.y, _o.x, _o.y, case)
            # plt.pause(5)
            # a.remove()
            # b.remove()

            self.object.connect(_o)

    def initialize(self):
        self.local_col = self.start_col
        self.local_row = self.start_row
        self.max_dist = 0
        self.cell_id = cell_id_from_rc(self.cells, self.local_col, self.local_row)
        # show_cell(self.cells, self.local_col, self.local_row)
        self.connect()

    def iterate(self):
        self.local_col = self.column + self.start_col
        self.local_row = self.row + self.start_row
        self.cell_id = cell_id_from_rc(self.cells, self.local_col, self.local_row)
        self.max_dist = max(self.max_dist, self.dist())
        # show_cell(self.cells, self.local_col, self.local_row)
        self.connect()

        # print("iterate> r=", self.radius, "col=", self.column, "row=", self.row, "d=", self.max_dist)

    def test_stop(self):
        # print("mytest_stop> r=", self.radius, "col=", self.column, "row=", self.row, "d=", self.max_dist)
        return self.max_dist > self.limit



Objects = dict()


class Object(object):
    def __init__(self, cells, object_id, object_x, object_y):
        self.cells = cells
        self.id = object_id
        self.x = object_x
        self.y = object_y
        self.cell_id = cell_id(cells=cells, _x=self.x, _y=self.y)
        # a = plt.scatter(object_x, object_y, color='y', s=40)

        col = int(self.cell_id % cells)
        row = int(self.cell_id / cells)
        # show_cell(cells, col, row)
        # plt.pause(5)
        # a.remove()

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
        x1 = self.x
        y1 = self.y
        x2 = other.x
        y2 = other.y

        d = lambda _x1, _x2, _y1, _y2 : np.sqrt(np.power(_x1 - _x2, 2) + np.power(_y1 - _y2, 2))

        cases = np.array([d(x1, x2,     y1, y2),
                          d(x1, x2 + 1, y1, y2),
                          d(x1, x2 + 1, y1, y2 + 1),
                          d(x1, x2,     y1, y2 + 1),
                          d(x1, x2 - 1, y1, y2 + 1),
                          d(x1, x2 - 1, y1, y2),
                          d(x1, x2 - 1, y1, y2 - 1),
                          d(x1, x2,     y1, y2 - 1),
                          d(x1, x2 + 1, y1, y2 - 1) ])
        dist_min = min(cases)
        indices = [i for i, v in enumerate(cases) if v == dist_min]
        return dist_min, indices

    def connect(self, other):
        self.edges.append(other)


text = None

def onclick(event):
    global text

    if text is not None:
        text.remove()
        text = None

    # print("Click occured in axe at x={} y={}".format(event.xdata, event.ydata))
    if event.xdata is None:
        return
    x = event.xdata
    y = event.ydata
    if x < 0 and x > 1 and y < 0 and y > 1:
        return
    col = int(x * cells)
    row = int(y * cells)
    cell_id = row*cells + col
    # print("cell=", cell_id, "col=", col, "row=", row)

    if cell_id not in Objects:
        # print("connect> current cell not in Objects", self.cell)
        return

    objects_in_cell = Objects[cell_id]
    for o in objects_in_cell:
        ox = o.x
        oy = o.y
        if abs(ox - x) < 0.01 and (oy - y) < 0.01:
            text = plt.text(x, y, "{}|{}|{}".format(o.id, col, row), fontsize=14)

space = mountain_space.build_space(fields=40, space_grid_size=100)
space_grid_size = space.shape[0]

fig, axe1 = plt.subplots(1, 1, subplot_kw={'projection': '3d', 'aspect': 'equal'})
# ax.plot_wireframe(x, y, space, color='r')
# ax.scatter(xi, yi, zi, s=1, c='r', zorder=1)

fig.canvas.mpl_connect('motion_notify_event', onclick)
fig.canvas.mpl_connect('button_press_event', onclick)

axe1.plot_surface(mountain_space.x_vector(0, space_grid_size),
                  mountain_space.y_vector(0, space_grid_size)[:, np.newaxis], space, color='r')

# =========================================================================================

# now to handle objects, we devide the space in cells
cells = 2 * 10 + 1                             # division of space in cells
cell_width = 1.0 / cells
limit = 0.05

ox = []
oy = []

axe2 = plt.subplot2grid((1, 1), (0, 0))

"""
for row in range(cells + 1):
    plt.plot((0, 1), (row * cell_width, row * cell_width), 'g')
    for col in range(cells + 1):
        plt.plot((col * cell_width, col * cell_width), (0, 1), 'g')
"""

n = 0
for i in range(100000):
    x = np.random.random()
    y = np.random.random()
    z = mountain_space.interpolate(from_space=space, _x=x, _y=y)
    p = np.random.random()
    if p > z:
        continue

    ox.append(x)
    oy.append(y)

    axe2.scatter(ox, oy, color='k', s=1)

    o = Object(cells=cells, object_id=n, object_x=x, object_y=y)
    it = ObjectConnector(cells=cells, obj=o, limit=limit)
    it.run()

    plt.pause(0.0001)

    n += 1

plt.show()
