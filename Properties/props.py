
import random
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d

N = 100
P = 20

randx = lambda: np.random.random()
randy = lambda: np.random.random()

# Create objects

class Alert(object):
    def __init__(self, x, y, properties):
        self.x = x
        self.y = y
        self.properties = properties

    def __str__(self):
        return "x={} y={} p={}".format(self.x, self.y, self.properties)

object_properties = lambda : [p for p in random.sample(range(P), int(random.randint(0, P))) ]
Alerts = {a_id: Alert(randx(), randy(), object_properties()) for a_id in range(N)}

# for a_id in Alerts:
#    print("object=", a_id, Alerts[a_id])

# Create property base

properties = dict()

for a_id in Alerts:
    plist = Alerts[a_id].properties
    for p in plist:
        if p in properties:
            olist = properties[p]
        else:
            olist = []
        olist.append(a_id)
        properties[p] = olist

zones = dict()

def zoneid(plist):
    copy = plist
    copy.sort()
    return frozenset(copy)

for a_id in Alerts:
    zone = zoneid(Alerts[a_id].properties)
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
    for _a_id in Alerts:
        alert = Alerts[_a_id]
        _ox = alert.x
        _oy = alert.y
        plist = alert.properties

        if abs(_ox - _x) < epsilon and abs(_oy - _y) < epsilon:
            # print("o=", _o, "plist=", plist)
            texts.append(axe1.text(_x + dx, _y + 0.02, "{}".format(_a_id), fontsize=14))
            dx += 0.1

    dx = 0.03
    for e in edges:
        a1 = Alerts[e[0]]
        a2 = Alerts[e[1]]
        x1 = a1.x
        y1 = a1.y
        x2 = a2.x
        y2 = a2.x
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
            z1 = zoneid(a1.properties)
            z2 = zoneid(a2.properties)
            d = list(z1.symmetric_difference(z2))
            texts.append(axe1.text(_x + dx, _y - 0.05, "{}{}".format(e, d), fontsize=14))
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

# simulate many random region couples to evaluate all possible distances

zarray = []
for i in range(100000):
    zs = random.sample(zones.keys(), 2)
    z1 = zs[0]
    z2 = zs[1]
    z = zdist(z1, z2)
    zarray.append(z)
    # print(z1, z2, z)
    # axe2.scatter(z1, z2, zdist(z1, z2))

zarray = []
for i in range(P):
    for j in range(P-i, 1, -1):
        z = float(i) / float(j)
        zarray.append(z)

min_dist = min(zarray)
print("min dist", min_dist)

# plot the distribution of all ossible distances beween regions

a = np.array(zarray)
y, bins = np.histogram(a, P)
x = bins[:-1] + 0.5 * (bins[1] - bins[0])
mean = np.sum(x * y) / a.size
axe2.plot(x, y, 'b-', label='data')

# construct edges between alerts at minimum distance

edges = []

for a_id1 in Alerts:
    alert1 = Alerts[a_id1]
    zone1 = zoneid(alert1.properties)
    for a_id2 in Alerts:
        if a_id2 == a_id1:
            continue
        alert2 = Alerts[a_id2]
        zone2 = zoneid(alert2.properties)
        try:
            d = zdist(zone1, zone2)
            if d == min_dist and (a_id2, a_id1) not in edges:
                edges.append((a_id1, a_id2))
        except:
            print("zero")

# Draw links between Alerts

for e in edges:
    a1 = e[0]
    a2 = e[1]
    src = Alerts[a1]
    dst = Alerts[a2]
    axe1.plot((src.x, dst.x), (src.y, dst.y))

"""
# Draw Alerts

for a_id in Alerts:
    alert = Alerts[a_id]
    axe1.scatter(alert.x, alert.y, s=3)
"""

"""
    xs.append(float(z1))
    ys.append(float(z2))

    matrix[z1, z2] = zdist(z1, z2)
"""

# print(np.array(xs).shape, np.array(ys).shape, matrix.shape)
# axe2.plot_surface(x_vector(0, s), y_vector(0, s)[:, np.newaxis], matrix, color='r')

plt.show()


