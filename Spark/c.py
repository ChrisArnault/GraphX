import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import SQLContext

import random
import time

import graphframes
# from graphframes.examples import Graphs

spark = SparkSession.builder.appName("GraphX").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

sqlContext = SQLContext(spark.sparkContext)

num_vertices = 1000000
num_edges =    1000000
degree_max =  100
distance_max = 0.1

x = lambda : np.random.random()
y = lambda : np.random.random()

print("creating vertices")
v = sqlContext.createDataFrame([(v_id, x(), y()) for v_id in range(num_vertices)], ["id", "x", "y"])

"""
    for vertex in vertices:
        vertex.out_vertices = [v for v in random.sample(vertices, np.random.randint(0, np.sqrt(len(vertices)))) if vertex.dist(v) < distance_max]
"""

class Stepper(object):
    previous_time = None

    def __init__(self):
        self.previous_time = time.time()

    def show_step(self, label='Initial time'):
        now = time.time()
        delta = now - self.previous_time

        if delta < 60:
            t = '0h0m{:.3f}s'.format(delta)
        elif delta < 3600:
            m = int(delta / 60)
            s = delta - (m*60)
            t = '0h{}m{:.3f}s'.format(m, s)
        else:
            h = int(delta / 3600)
            d = delta - (h*3600)
            m = int(d / 60)
            s = d - (m*60)
            t = '{}h{}h{:.3f}s'.format(h, m, s)

        print('--------------------------------', label, '|', t)

        self.previous_time = now

        return delta

def edge_it(n, last):
    v = 0
    while v < n:
        m = random.randint(0, int(degree_max))
        j = 0
        while j < m:
            j += 1
            w = random.randint(0, n)
            # print(v, w)
            if v > last:
                break
            yield (v, w)
        if v > last:
            break
        v += 1



# s = [(v, w) for v, w in edge_it(1000000000, 10)]
# print(s)

print("creating edges")
e = sqlContext.createDataFrame([(v_id, w_id) for v_id, w_id in edge_it(num_vertices, num_edges)], ["src", "dst"])

print("Create a GraphFrame")
g = graphframes.GraphFrame(v, e)

print("Display the vertex and edge DataFrames")
g.vertices.show(10)
g.edges.show(10)

print("Get a DataFrame with columns id and degree")
vertexDegrees = g.degrees
vertexDegrees.show()

# Find the youngest user's age in the graph.
# This queries the vertex DataFrame.
# g.vertices.groupBy().min("x").show()

# Count the number of "follows" in the graph.
# This queries the edge DataFrame.
# numFollows = g.edges.filter("relationship = 'follow'").count()

# print("numFollows=", numFollows)






