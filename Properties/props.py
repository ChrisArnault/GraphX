
import random
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d

N = 100
P = 20

randx = lambda: np.random.random()
randy = lambda: np.random.random()

# Create objects

object_properties = lambda : [p for p in random.sample(range(P), int(random.randint(0, P))) ]
objects = {o: (randx(), randy(), object_properties()) for o in range(N)}

# for o in objects:
#    print("object=", o, objects[o])

# Create property base

properties = dict()

for o in objects:
    plist = objects[o][2]
    for p in plist:
        if p in properties:
            olist = properties[p]
        else:
            olist = []
        olist.append(o)
        properties[p] = olist

zones = dict()

def zoneid(plist):
    a = plist
    a.sort()
    return frozenset(a)

for o in objects:
    zone = zoneid(objects[o][2])
    # print("zone:", a, b, pset)
    if zone not in zones:
        zones[zone] = True

"""
for id in zones:
    print("zones", id)
"""

def zdist(z1, z2):
    if len(z1) + len(z2) == 0:
        return 0.0
    else:
        return float(len(z1.symmetric_difference(z2)))/float(len(z1) + len(z2))

# print(edges)

# ===============================================
# Graphics

# fig, axe2 = plt.subplots(1, 1)
fig, (axe1, axe2) = plt.subplots(1, 2)
# fig, (axe1, axe2) = plt.subplots(1, 2, subplot_kw={'projection': '3d', 'aspect': 'equal'})

texts = None
def onclick(event):
    global texts
    if texts is not None:
        for text in texts:
            text.remove()
        texts = None

    if event.xdata is None:
        return

    _x: float = event.xdata
    _y: float = event.ydata
    if _x < 0 and _x > 1:
        if _y < 0 and _y > 1:
            return

    epsilon = 0.01

    texts = []

    dx = 0
    for _o in objects:
        obj = objects[_o]
        _ox = obj[0]
        _oy = obj[1]
        plist = obj[2]

        if abs(_ox - _x) < epsilon and abs(_oy - _y) < epsilon:
            plist = obj[2]
            # print("o=", _o, "plist=", plist)
            texts.append(axe1.text(_x+dx, _y+0.02, "{}".format(_o), fontsize=14))
            dx += 0.1

    dx = 0.03
    for e in edges:
        o1 = objects[e[0]]
        o2 = objects[e[1]]
        x1 = o1[0]
        y1 = o1[1]
        x2 = o2[0]
        y2 = o2[1]
        a = (y1 - y2)/(x1 - x2)
        b1 = y1 - a*x1
        b2 = y2 - a*x2
        _ey1 = a*_x + b1
        _ey2 = a*_x + b2
        if (abs(_ey1 - _y) < epsilon and
                (abs(_ey2 - _y) < epsilon) and
                (_x >= min(x2, x1)) and
                (_x <= max(x2, x1))):
            # print("e=", e)
            z1 = zoneid(o1[2])
            z2 = zoneid(o2[2])
            d = list(z1.symmetric_difference(z2))
            texts.append(axe1.text(_x+dx, _y-0.05, "{}{}".format(e, d), fontsize=14))
            dx += 0.15

    fig.canvas.draw()

fig.canvas.mpl_connect('motion_notify_event', onclick)
# fig.canvas.mpl_connect('button_press_event', onclick)


"""
xs = []
ys = []
zs = []

s = len(zones.keys())

x_vector = lambda grid_column, grid_size: (np.arange(0, grid_size, 1, float) / grid_size) + grid_column
y_vector = lambda grid_row, grid_size: (np.arange(0, grid_size, 1, float) / grid_size) + grid_row

vx = np.zeros(s)
vy = np.zeros(s)
vy = vy[:, np.newaxis]  # transpose y
matrix = vx * vy
"""
zarray = []
for i in range(100000):
    zs = random.sample(zones.keys(), 2)
    z1 = zs[0]
    z2 = zs[1]
    z = zdist(z1, z2)
    zarray.append(z)
    # print(z1, z2, z)
    # axe2.scatter(z1, z2, zdist(z1, z2))

min_dist = min(zarray)
print("min dist", min_dist)

a = np.array(zarray)
y, bins = np.histogram(a, P)
x = bins[:-1] + 0.5 * (bins[1] - bins[0])
mean = np.sum(x * y) / a.size
axe2.plot(x, y, 'b-', label='data')

edges = []

for o in objects:
    obj1 = objects[o]
    zone1 = zoneid(obj1[2])
    for other in objects:
        if other == o:
            continue
        obj2 = objects[other]
        zone2 = zoneid(obj2[2])
        try:
            d = zdist(zone1, zone2)
            if d == min_dist and (other, o) not in edges:
                edges.append((o, other))
        except:
            print("zero")

# Draw links between objects

for e in edges:
    o1 = e[0]
    o2 = e[1]
    src = objects[o1]
    dst = objects[o2]
    axe1.plot((src[0], dst[0]), (src[1], dst[1]))

# Draw objects
"""
for o in objects:
    obj = objects[o]
    axe1.scatter(obj[0], obj[1], s=3)
"""

"""
    xs.append(float(z1))
    ys.append(float(z2))

    matrix[z1, z2] = zdist(z1, z2)
"""

# print(np.array(xs).shape, np.array(ys).shape, matrix.shape)
# axe2.plot_surface(x_vector(0, s), y_vector(0, s)[:, np.newaxis], matrix, color='r')

plt.show()


