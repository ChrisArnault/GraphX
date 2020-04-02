import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import SQLContext

import random
import time

import graphframes

from simple_model_conf import *

spark = SparkSession.builder.appName("GraphX").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

sqlContext = SQLContext(spark.sparkContext)

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



x = lambda : np.random.random()
y = lambda : np.random.random()

s = Stepper()

vertices = sqlContext.createDataFrame([(v_id, x(), y()) for v_id in range(num_vertices)], ["id", "x", "y"])
s.show_step("creating vertices")

edges = sqlContext.createDataFrame([(v_id, w_id) for v_id, w_id in edge_it(num_vertices, num_edges)], ["src", "dst"])
s.show_step("creating edges")

g = graphframes.GraphFrame(vertices, edges)
s.show_step("Create a GraphFrame")

g.vertices.write.parquet(home + "/vertices")
g.edges.write.parquet(home + "/edges")
s.show_step("Save the GraphFrame")
