
import random
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.colors as mcolors
# from mpl_toolkits.mplot3d import axes3d
from typing import List, Tuple, Optional

N = 50
P = 20

randx = lambda: np.random.random()
randy = lambda: np.random.random()


# Create database alert


class Alert(object):
    def __init__(self, x: float, y: float, _properties: List[int]):
        self.x = x
        self.y = y
        self.properties = _properties

    def __str__(self):
        return "x={} y={} p={}".format(self.x, self.y, self.properties)


object_properties = lambda: [_p for _p in random.sample(range(P), int(random.randint(0, P)))]
Alerts = {a_id: Alert(randx(), randy(), object_properties()) for a_id in range(N)}


# Create property database base


properties = dict()


for alert_id in Alerts:
    plist = Alerts[alert_id].properties
    for p in plist:
        if p in properties:
            olist = properties[p]
        else:
            olist = []
        olist.append(alert_id)
        properties[p] = olist


# Create zone database base


def zoneid(_plist: List[int]) -> frozenset:
    copy = _plist
    copy.sort()
    return frozenset(copy)


def zdist(z1: frozenset, z2: frozenset) -> float:
    if len(z1) + len(z2) == 0:
        return 0.0
    else:
        return float(len(z1.symmetric_difference(z2)))/float(len(z1) + len(z2))


class Zone(object):
    def __init__(self):
        self.alerts = []

    def add_alert(self, _alert_id: int) -> None:
        if _alert_id not in self.alerts:
            self.alerts.append(_alert_id)


class Zones(object):
    def __init__(self):
        self.db = dict()
        self.distances = []

    def get_zone(self, _zone_id: frozenset) -> Optional[Zone]:
        if _zone_id in self.db:
            return self.db[_zone_id]
        return None

    def fill_db(self, alerts: Alerts) -> None:
        for _alert_id in alerts:
            zone = zoneid(alerts[_alert_id].properties)
            # print("zone:", a, b, pset)
            if zone not in self.db:
                self.db[zone] = Zone()
            self.db[zone].add_alert(_alert_id=_alert_id)

    def print_db(self) -> None:
        for zone_id in self.db:
            print("zones", zone_id)

    def simulate_distances(self) -> List[float]:
        self.distances = []
        for i in range(100000):
            zs = random.sample(self.db.keys(), 2)
            z1 = zs[0]
            z2 = zs[1]
            z = zdist(z1, z2)
            if z not in self.distances:
                self.distances.append(z)
            # print(z1, z2, z)
        self.distances.sort()

        return self.distances

    def all_distances(self) -> List[float]:
        self.distances = []
        for i in range(1, P + 1):
            for j in range(2*P, i - 1, -1):
                z = float(i) / float(j)
                if z not in self.distances:
                    self.distances.append(z)

        self.distances.sort()
        print(len(self.distances), self.distances)

        return self.distances

    def min_distances(self) -> float:
        return min(self.distances)

    def plot_distances(self, axe: plt.Axes) -> None:
        self.simulate_distances()
        a = np.array(self.distances)
        y, bins = np.histogram(a, P)
        x = bins[:-1] + 0.5 * (bins[1] - bins[0])
        # x = bins[:-1]
        # mean = np.sum(x * y) / a.size
        axe.set(title="Distances (simulation)")
        axe.plot(x, y, 'b-', label='data')
        self.all_distances()


def get_colors_names():
    colors = mcolors.CSS4_COLORS
    by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))), name) for name, color in colors.items())
    col = [name for hsv, name in by_hsv if hsv[2] <= 0.7]
    return random.sample(col, len(col))


zones = Zones()
zones.fill_db(Alerts)


color_names = get_colors_names()
zarray = zones.all_distances()
distance_color = {d: color_names[i % len(color_names)] for i, d in enumerate(zarray)}


class Edges(object):
    def __init__(self):
        self.db = []
        self.segments = []

    def erase_segments(self) -> None:
        for segment in self.segments:
            for line in segment:
                line.remove()
        self.segments = []

    def fill_db_from_alerts(self, alerts: Alerts, _min_dist: float, _max_dist: float) -> List[Tuple[int, int, str]]:
        self.db = []

        for alert_id1 in alerts:
            alert1 = alerts[alert_id1]
            zone1 = zoneid(alert1.properties)
            for alert_id2 in alerts:
                if alert_id2 == alert_id1:
                    continue

                alert2 = alerts[alert_id2]
                zone2 = zoneid(alert2.properties)
                d = zdist(zone1, zone2)
                color = distance_color[d]
                if (d >= _min_dist) and \
                        (d <= _max_dist) and \
                        ((alert_id2, alert_id1, color) not in self.db):
                    self.db.append((alert_id1, alert_id2, color))

        return self.db

    def fill_db(self, _zones: Zones, _min_dist: float, _max_dist: float) -> List[Tuple[int, int, str]]:
        self.db = []

        for zone_id1 in _zones.db:
            for zone_id2 in _zones.db:
                if zone_id2 == zone_id1:
                    continue
                d = zdist(zone_id1, zone_id2)
                if (d < _min_dist) or (d > _max_dist):
                    continue
                zone1 = _zones.get_zone(zone_id1)
                zone2 = _zones.get_zone(zone_id2)
                color = distance_color[d]
                for _alert_id1 in zone1.alerts:
                    for _alert_id2 in zone2.alerts:
                        if _alert_id2 == _alert_id1:
                            continue
                        if (_alert_id2, _alert_id1, color) not in self.db:
                            self.db.append((_alert_id1, _alert_id2, color))

        return self.db

    def plot_db(self, axe: plt.Axes) -> None:
        self.erase_segments()
        for e in self.db:
            a1 = e[0]
            a2 = e[1]
            color = e[2]
            src = Alerts[a1]
            dst = Alerts[a2]
            self.segments.append(axe.plot((src.x, dst.x), (src.y, dst.y), color=color))


edges = Edges()


# ===============================================
# Graphics
# ===============================================

class UpdateSlider(object):
    def __init__(self, fig, slider, other, axe):
        self.fig = fig
        self.slider = slider
        self.other = other
        self.axe = axe


class UpdateMaxSlider(UpdateSlider):
    def __call__(self, val):
        if val < self.other.val:
            val = self.other.val
            self.slider.set_val(val)
        # print("max slider changed", val)

        edges.fill_db(_zones=zones, _min_dist=self.other.val, _max_dist=self.slider.val)
        edges.plot_db(axe=self.axe)

        self.fig.canvas.draw()
        self.fig.canvas.draw_idle()


class UpdateMinSlider(UpdateSlider):
    def __call__(self, val):
        if val > self.other.val:
            val = self.other.val
            self.slider.set_val(val)
        # print("min slider changed", val)

        edges.fill_db(_zones=zones, _min_dist=self.slider.val, _max_dist=self.other.val)
        edges.plot_db(axe=self.axe)

        self.fig.canvas.draw()
        self.fig.canvas.draw_idle()


class Graphics(object):
    def __init__(self):
        self.fig = plt.figure()
        self.axe1 = plt.axes([0.08,0.1,0.4,0.72])
        self.axe2 = plt.axes([0.55,0.1,0.4,0.85])
        # self.fig, (self.axe1, self.axe2) = plt.subplots(1, 2, subplot_kw={'projection': '3d', 'aspect': 'equal'})
        self.texts = []
        self.fig.canvas.mpl_connect('motion_notify_event', self.onclick)
        axcolor = 'lightgoldenrodyellow'
        self.slider_max_axe = plt.axes([0.12, 0.94, 0.33, 0.03], facecolor=axcolor)
        self.slider_min_axe = plt.axes([0.12, 0.89, 0.33, 0.03], facecolor=axcolor)

        self.slider_max = Slider(self.slider_max_axe, 'Max dist', 0.0, 1.0, valinit=0)
        self.slider_min = Slider(self.slider_min_axe, 'Min dist', 0.0, 1.0, valinit=0)
        self.update_slider_max = UpdateMaxSlider(fig=self.fig,
                                                 slider=self.slider_max,
                                                 other=self.slider_min,
                                                 axe=self.axe1)
        self.update_slider_min = UpdateMinSlider(fig=self.fig,
                                                 slider=self.slider_min,
                                                 other=self.slider_max,
                                                 axe=self.axe1)
        self.slider_max.on_changed(self.update_slider_max)
        self.slider_min.on_changed(self.update_slider_min)

    def erase_texts(self) -> None:
        for text in self.texts:
            text.remove()
        self.texts = []

    def draw_alerts(self) -> None:
        for _alert_id in Alerts:
            alert = Alerts[_alert_id]
            _alert_x = alert.x
            _alert_y = alert.y
            self.axe1.set(title="Alerts & correlation with distance")
            self.axe1.scatter(_alert_x, _alert_y, s=1)

    def onclick(self, event) -> None:
        self.erase_texts()

        if event.xdata is None:
            return

        _x: float = event.xdata
        _y: float = event.ydata
        if (_x < 0) and (_x > 1):
            if (_y < 0) and (_y > 1):
                return

        epsilon = 0.01

        dx = 0
        for _alert_id in Alerts:
            alert = Alerts[_alert_id]
            _alert_x = alert.x
            _alert_y = alert.y
            # plist = alert.properties

            if abs(_alert_x - _x) < epsilon and abs(_alert_y - _y) < epsilon:
                # print("alert=", alert, "plist=", plist)
                self.texts.append(self.axe1.text(_x + dx, _y + 0.02, "{}".format(_alert_id), fontsize=14))
                dx += 0.1

        dx = 0.03
        dy = 0
        for e in edges.db:
            alert1 = Alerts[e[0]]
            alert2 = Alerts[e[1]]
            x1 = alert1.x
            y1 = alert1.y
            x2 = alert2.x
            y2 = alert2.y
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
                z1 = zoneid(alert1.properties)
                z2 = zoneid(alert2.properties)
                d = list(z1.symmetric_difference(z2))
                eshort = (e[0], e[1])
                t = "{}{}".format(eshort, d)
                self.texts.append(self.axe1.text(_x + dx, _y - 0.05 - dy, t, fontsize=12))
                dy += 0.03

        self.fig.canvas.draw()


graphics = Graphics()

min_dist = zones.min_distances()
print("min dist", min_dist)

graphics.slider_max.set_val(min_dist)
graphics.slider_min.set_val(min_dist)

# plot the distribution of all possible distances between regions

zones.plot_distances(axe=graphics.axe2)

graphics.draw_alerts()

# construct edges between alerts at minimum distance

edges.fill_db(_zones=zones, _min_dist=min_dist, _max_dist=min_dist)
edges.plot_db(axe=graphics.axe1)

# print(np.array(xs).shape, np.array(ys).shape, matrix.shape)
# axe2.plot_surface(x_vector(0, s), y_vector(0, s)[:, np.newaxis], matrix, color='r')

# plt.tight_layout()
plt.show()


