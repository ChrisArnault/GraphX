
import random
import numpy as np
from matplotlib import pyplot as plt

N = 50
P = 10

randx = lambda: np.random.random()
randy = lambda: np.random.random()

# Create objects

object_properties = lambda : [p for p in random.sample(range(P), int(random.randint(0, P)/3)) ]
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

zone_id = 0
zones = dict()

for o in objects:
    plist = objects[o][2]
    plist.sort()
    found = False
    for id in zones:
        zone = zones[id]
        if zone == plist:
            found = True
            break
    if not found:
        zones[zone_id] = plist
        zone_id += 1

for id in zones:
    print("zones", id, zones[id])

edges = []
for p in properties:
    # print("property=", p, properties[p])
    olist = properties[p]
    def create_edge(olist):
        if len(olist) <= 1:
            return []
        head = [(olist[0], o) for o in olist[1:] if (olist[0], o) not in edges]
        head += create_edge(olist[1:])
        return head
    edges += create_edge(olist)

# print(edges)

# ===============================================
# Graphics

fig, axe1 = plt.subplots(1, 1)

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

    for _o in objects:
        obj = objects[_o]
        _ox = obj[0]
        _oy = obj[1]

        if abs(_ox - _x) < epsilon and abs(_oy - _y) < epsilon:
            plist = obj[2]
            # print("o=", _o, "plist=", plist)
            texts.append(axe1.text(_x, _y+0.02, "{}".format(_o), fontsize=14))

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
            texts.append(axe1.text(_x+dx, _y-0.05, "{}".format(e), fontsize=14))
            dx += 0.12

    fig.canvas.draw()

fig.canvas.mpl_connect('motion_notify_event', onclick)
# fig.canvas.mpl_connect('button_press_event', onclick)

# Draw objects

for o in objects:
    obj = objects[o]
    axe1.scatter(obj[0], obj[1], s=3)

# Draw links between objects

for e in edges:
    o1 = e[0]
    o2 = e[1]
    src = objects[o1]
    dst = objects[o2]
    axe1.plot((src[0], dst[0]), (src[1], dst[1]))

plt.show()


