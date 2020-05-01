import numpy as np
import time
import os
import sys
import gc
import collections
import re

class Conf(object):
    def __init__(self):
        self.graphs_base = "/user/chris.arnault/graphs"
        self.name = "c:/tmp/triangles.csv"

    def set(self):
        run = True

        for i, arg in enumerate(sys.argv[1:]):
            a = arg.split("=")
            # print(i, arg, a)
            key = a[0]
            if key == "name" or key == "F" or key == "f":
                self.name = a[1]
            elif key == "Args" or key == "args" or key == "A" or key == "a":
                run = False
            elif key[:2] == "-h" or key[0] == "h":
                print('''
> python check_triangles.py 
  name|F|f = "test"
                ''')
                exit()

        [print(a, "=", getattr(self, a)) for a in dir(self) if a[0] != '_']

        if not run:
            exit()

conf = Conf()
conf.set()

items1 = collections.OrderedDict([
    ("batch=", 1),
    ("vertices=", 3),
    ("edges=", 5),
    ("partial", 7)
])

items2 = collections.OrderedDict([
    ("batch=", 1),
    ("vertices=", 3),
    ("edges=", 5),
    ("total=", 7),
    ("partial", 9)
])

total_time = 0
total_triangles = 0
with open(conf.name) as f:
    for line in f:
        line = line.strip()
        if line == '':
            continue
        words = line.split()
        if words[0] == 'batch=':
            if 'total=' in words:
                t = ", ".join(["{}{}".format(k, words[items2[k]]) for k in items2])
                print(t)
                partial = int(words[items2["partial"]])
            else:
                t = ", ".join(["{}{}".format(k, words[items1[k]]) for k in items1])
                print(t)
                partial = int(words[items1["partial"]])

            total_triangles += partial

        elif "triangleCount" in words:
            i = words.index('|')
            t = words[i+1]
            m = re.match("([0-9]*)h([0-9]*)m([0-9.]*)s", t)
            if m is not None:
                s = float(m[3]) + 60.0 * float(m[2]) + 60.0 * float(m[1])
                print("time", t, s)
                total_time += s

print("total triangles", total_triangles)

h = int(total_time/3600)
total_time -= h*3600
m = int(total_time/60)
total_time -= m*60
s = total_time
print("{}h{}m{}s".format(h, m, s))
