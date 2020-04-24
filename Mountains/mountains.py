

"""
Generate mountains

"""

import numpy as np
from matplotlib import pyplot as plt

import mountain_cells
import mountain_space

"""
Now the space is filled with a repulsive field
We start populating it with objects
"""


def cell_id(_cells: int, _x: float, _y: float) -> int:
    """ compute a cell id from a x, y coordinate """
    _row = int(_y * _cells)
    _col = int(_x * _cells)
    return _col + _cells * _row


def cell_id_from_rc(_cells: int, _col: int, _row: int) -> int:
    """ compute a cell id from a row, column index """
    return (_col % _cells) + _cells * (_row % _cells)


def col_from_cell_id(_cells: int, _cell_id: int) -> int:
    """ compute the x column from the cell_id """
    return int(_cell_id % _cells)


def row_from_cell_id(_cells: int, _cell_id: int) -> int:
    """ compute the y row from the cell_id """
    return int(_cell_id / _cells)


def neighbour(_cells: int, cell1: int, cell2: int) -> bool:
    """
    test if the two cells (cell1 & cell2) are neighbours inside complete space with cells
    neighbouring is true if cells are close by 1 left-right , top-bottom, or by corners
    the grid is cyclic by left-right , top-bottom, or by corners
    """
    t1 = cell1 == cell2
    if t1:
        return True

    xc1 = col_from_cell_id(_cells=_cells, _cell_id=cell1)
    xc2 = col_from_cell_id(_cells=_cells, _cell_id=cell2)

    yc1 = row_from_cell_id(_cells=_cells, _cell_id=cell1)
    yc2 = row_from_cell_id(_cells=_cells, _cell_id=cell2)

    dx = abs(xc1 - xc2)
    dy = abs(yc1 - yc2)

    t2 = (dx == 0) & (dy == 1)
    if t2:
        return True
    t3 = (dx == 0) & (dy == _cells - 1)
    if t3:
        return True
    t4 = (dx == 1) & (dy == 0)
    if t4:
        return True
    t5 = (dx == _cells - 1) & (dy == 0)
    if t5:
        return True

    t6 = (dx == 1) & (dy == 1)
    if t6:
        return True
    t7 = (dx == 1) & (dy == _cells - 1)
    if t7:
        return True
    t8 = (dx == _cells - 1) & (dy == 1)
    if t8:
        return True

    t9 = (dx == _cells - 1) & (dy == _cells - 1)
    if t9:
        return True

    # print(t1, t2, t3, t4, t5, t6, t7, t8, t9)
    return False


"""
Temporary storage for graphs used tu draw cells during an iteration
"""


class CellIteration(object):
    def __init__(self) -> None:
        self.lines = []

    def __del__(self) -> None:
        for line in self.lines:
            for s in line:
                s.remove()
        self.lines = []

    def show_cell(self, _cells: int, col: int, row: int) -> None:
        w: float = 1.0/_cells
        _x = (col % _cells) * w
        _y = (row % _cells) * w
        # print('col=', self.column, 'row=', self.row, 'lcol=', self.local_col, 'lrow=', self.local_row)

        epsilon = w / 20.
        a = plt.plot((_x + epsilon, _x + w - epsilon, _x + w - epsilon, _x + epsilon, _x + epsilon),
                     (_y + epsilon, _y + epsilon, _y + w - epsilon, _y + w - epsilon, _y + epsilon), 'r')

        plt.pause(0.0001)

        self.lines.append(a)


def create_1_segment(x1: float, y1: float, x2: float, y2: float, _1, _2, _3, _4) -> None:
    plt.plot((x1, x2), (y1, y2), 'g')


def create_2_segments_x(x1: float, y1: float, x2: float, y2: float, edge_x: int, _1, a: float, b: float) -> None:
    # crossing x edge 1=right 0=left
    xc1 = edge_x
    yc1 = a * xc1 + b

    plt.plot((x1, edge_x), (y1, yc1), 'g')
    plt.plot((1 - edge_x, x2), (yc1, y2), 'g')


def create_2_segments_y(x1: float, y1: float, x2: float, y2: float, _1, edge_y: int, a: float, b: float) -> None:
    # crossing y edge 0=bottom 1=top
    yc2 = edge_y
    xc2 = (yc2 - b) / a

    plt.plot((x1, xc2), (y1, edge_y), 'g')
    plt.plot((xc2, x2), (1 - edge_y, y2), 'g')


def create_3_segments(x1: float, y1: float, x2: float, y2: float, edge_x: int, edge_y: int, a: float, b: float) -> None:
    # crossing x edge
    xc1 = edge_x
    yc1 = a * xc1 + b

    # crossing y edge
    yc2 = edge_y
    xc2 = (yc2 - b) / a

    # test whether the y crossing is out the [0, 1[ range to identify the quadrant of the intermediate segment
    y_out = yc1 >= 1 or yc1 < 0

    shifted_x = xc2
    shifted_y = yc1

    if y_out:
        # frst cross the X edge
        shifted_y %= 1
        plt.plot((x1, shifted_x), (y1, edge_y), 'g')
        plt.plot((shifted_x, edge_x), (1 - edge_y, shifted_y), 'g')
        plt.plot((1 - edge_x, x2), (shifted_y, y2), 'g')
    else:
        # frst cross the Y edge
        shifted_x %= 1
        plt.plot((x1, edge_x), (y1, shifted_y), 'g')
        plt.plot((1 - edge_x, shifted_x), (shifted_y, edge_y), 'g')
        plt.plot((shifted_x, x2), (1 - edge_y, y2), 'g')


def draw_connector(x1: float, y1: float, x2: float, y2: float, case: int) -> None:
    """
    Draw connector
    We handle here the situation where the connection between the two objects cross the edges

    (the case "0" is when the connection does not cross the edges)

    :param x1: coordinatex for first object
    :param y1:
    :param x2:: coordinatex for second object
    :param y2:
    :param case: situation of the objects in the spherical space (given by the distance computation)

    0: 1 @ center      2 @ center  (no crossing of the edges)
    1: 1 @ right side  2 @ left side
    2: 1 @ TR corner   2 @ BL corner
    3: 1 @ top side    2 @ bottom side
    4: 1 @ TL corner   2 @ BR corner
    5: 1 @ left side   2 @ right side
    6: 1 @ BL corner   2 @ BR corner
    7: 1 @ bottom side 2 @ top side
    8: 1 @ BR corner  2 @ TL corner
    """

    # keep_center = (0, )
    # keep_corners = (2, 4, 6, 8)
    # keep_sides = (1, 3, 5, 7)
    # ignore side-by-side crossing

    # if case not in keep_center:
    #     return

    #
    # we start by changing the coordinates of the dest object to consider the spheric space.
    # when the two object are closer by crossing the edges we consider that the connector
    # crosses the edge
    #
    # crossing sides makes 2 segments for the connector
    #
    # crossing the corners makes 3 segments, and 2 situations according to which intermediate quadrant is crossed
    #
    #
    # objectify the case handling:
    #   0: shift in X
    #   1: shift in Y
    #   2: draw segments function
    #
    cases = [(0, 0, create_1_segment),
             (1, 0, create_2_segments_x),
             (1, 1, create_3_segments),
             (0, 1, create_2_segments_y),
             (-1, 1, create_3_segments),
             (-1, 0, create_2_segments_x),
             (-1, -1, create_3_segments),
             (0, -1, create_2_segments_y),
             (1, -1, create_3_segments),
             ]

    shifted_x = x2
    shifted_y = y2

    # shift quadrant for the destination object to make it closer

    shift_x = cases[case][0]
    shift_y = cases[case][1]
    shifted_x += shift_x
    shifted_y += shift_y

    # the linear equation y = ax + b for the connector
    a = (y1 - shifted_y) / (x1 - shifted_x)
    b = y1 - (a * x1)

    f = cases[case][2]
    edge_x = (shift_x + 1) / 2
    edge_y = (shift_y + 1) / 2
    f(x1, y1, x2, y2, edge_x, edge_y, a, b)


class Object(object):
    def __init__(self, _cells: int, _object_id: int, object_x: float, object_y: float) -> None:
        self.cells = _cells
        self.id = _object_id
        self.x = object_x
        self.y = object_y
        self.cell_id = cell_id(_cells=_cells, _x=self.x, _y=self.y)
        # a = plt.scatter(object_x, object_y, color='y', s=40)

        # col = int(self.cell_id % _cells)
        # row = int(self.cell_id / _cells)
        # show_cell(cells, col, row)
        # plt.pause(5)
        # a.remove()

        self.declare()
        self.edges = []

    def declare(self) -> None:
        if self.cell_id in Objects:
            objects_in_cell = Objects[o.cell_id]
        else:
            objects_in_cell = []

        # print("cell for object cell=", o.cell_id, len(objects_in_cell))

        objects_in_cell.append(self)
        Objects[self.cell_id] = objects_in_cell

    def dist(self, other: "Object") -> (float, bytearray):
        x1 = self.x
        y1 = self.y
        x2 = other.x
        y2 = other.y

        d = lambda _x1, _x2, _y1, _y2: np.sqrt(np.power(_x1 - _x2, 2) + np.power(_y1 - _y2, 2))

        """
        in order to understand the respective situation of the two objects we have to connect
        we compute the 9 distances
        We deduce a case topology

        0: 1 @ center      2 @ center  (no crossing of the edges)
        1: 1 @ right side  2 @ left side
        2: 1 @ TR corner   2 @ BL corner
        3: 1 @ top side    2 @ bottom side
        4: 1 @ TL corner   2 @ BR corner
        5: 1 @ left side   2 @ right side
        6: 1 @ BL corner   2 @ BR corner
        7: 1 @ bottom side 2 @ top side
        8: 1 @ BR corner   2 @ TL corner
        """

        cases = np.array([d(x1, x2,     y1, y2),
                          d(x1, x2 + 1, y1, y2),
                          d(x1, x2 + 1, y1, y2 + 1),
                          d(x1, x2,     y1, y2 + 1),
                          d(x1, x2 - 1, y1, y2 + 1),
                          d(x1, x2 - 1, y1, y2),
                          d(x1, x2 - 1, y1, y2 - 1),
                          d(x1, x2,     y1, y2 - 1),
                          d(x1, x2 + 1, y1, y2 - 1)])

        dist_min = min(cases)
        indices = [i for i, v in enumerate(cases) if v == dist_min]
        return dist_min, indices

    def connect(self, other: "Object", _case: int) -> None:
        self.edges.append((other, _case))

    def draw_connections(self):
        for edge in self.edges:
            _o = edge[0]
            _case = edge[1]
            draw_connector(self.x, self.y, _o.x, _o.y, _case)


class ObjectConnector(mountain_cells.CellIterator):
    """
    iterate over neighbour cells up to max distance
    then within each neighbour cell, we connect objects according their real distance
    """
    def __init__(self, _cells: int, obj: Object, _limit: float) -> None:
        # print("==================================================================================")
        mountain_cells.CellIterator.__init__(self)
        self.cells = _cells
        self.cell_width = 1.0/self.cells
        self.limit = _limit
        self.max_dist = 0
        self.object = obj
        self.start_x = obj.x
        self.start_y = obj.y

        self.start_col = int(self.start_x * self.cells)
        self.start_row = int(self.start_y * self.cells)

        self.cell_id = self.start_row * self.cells + self.start_col

        self.local_col = self.start_col
        self.local_row = self.start_row

        self.cell_iterator = CellIteration()

    def cell_dist(self) -> float:
        return np.sqrt(np.power((self.local_col - self.start_col) * self.cell_width, 2) +
                       np.power((self.local_row - self.start_row) * self.cell_width, 2))

    def connect(self) -> None:
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
                # print("connect> current object", self.object.id,
                # "not neighbour of", _o.id,
                # "d=", d, "indices=", indices)

                # plt.pause(1)
                # a.remove()
                # b.remove()
                continue
            # print("connect o=", self.object.id, self.object.x, self.object.y,
            # "to=", _o.id, _o.x, _o.y, "d=", d, "indices=", indices)
            case = indices[0]
            # draw_connector(self.object.x, self.object.y, _o.x, _o.y, case)
            # plt.pause(5)
            # a.remove()
            # b.remove()

            self.object.connect(_o, case)

    def initialize(self) -> None:
        self.local_col = self.start_col
        self.local_row = self.start_row
        self.max_dist = 0
        self.cell_id = cell_id_from_rc(self.cells, self.local_col, self.local_row)
        # self.cell_iterator.show_cell(self.cells, self.local_col, self.local_row)
        self.connect()

    def iterate(self) -> None:
        self.local_col = self.column + self.start_col
        self.local_row = self.row + self.start_row
        self.cell_id = cell_id_from_rc(self.cells, self.local_col, self.local_row)
        self.max_dist = max(self.max_dist, self.cell_dist())
        # self.cell_iterator.show_cell(self.cells, self.local_col, self.local_row)
        self.connect()

        # print("iterate> r=", self.radius, "col=", self.column, "row=", self.row, "d=", self.max_dist)

    def test_stop(self) -> bool:
        # print("mytest_stop> r=", self.radius, "col=", self.column, "row=", self.row, "d=", self.max_dist)
        return self.max_dist > self.limit


Objects = dict()
text = None

class DrawSpace(object):
    def __init__(self, fig, _cells):
        self.cells = _cells
        self.text = None
        fig.canvas.mpl_connect('motion_notify_event', self.onclick)
        fig.canvas.mpl_connect('button_press_event', self.onclick)

    def onclick(self, event):
        if self.text is not None:
            self.text.remove()
            self.text = None

        # print("Click occured in axe at x={} y={}".format(event.xdata, event.ydata))
        if event.xdata is None:
            return
        _x: float = event.xdata
        _y: float = event.ydata
        if _x < 0 and _x > 1:
            if _y < 0 and _y > 1:
                return
        col = int(_x * self.cells)
        row = int(_y * self.cells)
        _cell_id = row * self.cells + col
        # print("cell=", cell_id, "col=", col, "row=", row)

        if _cell_id not in Objects:
            # print("connect> current cell not in Objects", self.cell)
            return

        objects_in_cell = Objects[_cell_id]
        for _o in objects_in_cell:
            _ox = _o.x
            _oy = _o.y
            if abs(_ox - _x) < 0.01 and (_oy - _y) < 0.01:
                self.text = plt.text(_x, _y, "{}|{}|{}".format(_o.id, col, row), fontsize=14)

# ==========================================================================================
# Building the space with the set of repulsive fields
#

space = mountain_space.build_space(fields=150, space_grid_size=200, width_factor=0.12)
space_grid_size = space.shape[0]

fig, axe1 = plt.subplots(1, 1, subplot_kw={'projection': '3d', 'aspect': 'equal'})

axe1.plot_surface(mountain_space.x_vector(0, space_grid_size),
                  mountain_space.y_vector(0, space_grid_size)[:, np.newaxis], space, color='r')

plt.show()

# =========================================================================================

# now to handle objects, we devide the space in cells
cells = 2 * 100 + 1                             # division of space in cells
# cells = 2 * 3 + 1                             # division of space in cells
limit = 0.02
# limit = 0.2

# ds = DrawSpace(fig=fig, _cells=cells)

cell_width = 1.0 / cells

ox = []
oy = []

axe2 = plt.subplot2grid((1, 1), (0, 0))

"""
for row in range(cells + 1):
    plt.plot((0, 1), (row * cell_width, row * cell_width), 'g')
    for col in range(cells + 1):
        plt.plot((col * cell_width, col * cell_width), (0, 1), 'g')
"""

for object_id in range(100000):
    """
    x = np.random.random() * 2 * cell_width - cell_width
    if x < 0:
        x = 1 + x
    y = np.random.random() * 2 * cell_width - cell_width
    if y < 0:
        y = 1 + y
    """
    x = np.random.random()
    y = np.random.random()
    z = mountain_space.interpolate(from_space=space, _x=x, _y=y)
    p = np.random.random()
    if p > z:
        continue

    ox.append(x)
    oy.append(y)

    axe2.scatter(ox, oy, color='k', s=1)

    o = Object(_cells=cells, _object_id=object_id, object_x=x, object_y=y)
    it = ObjectConnector(_cells=cells, obj=o, _limit=limit)
    it.run()

    o.draw_connections()

    plt.pause(0.0001)

    object_id += 1

plt.show()
