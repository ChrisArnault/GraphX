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

s = Stepper()

vertices = sqlContext.read.parquet(home + "/vertices")
edges = sqlContext.read.parquet(home + "/edges")
s.show_step("Load the vertices and edges back.")

# Create an identical GraphFrame.
g = graphframes.GraphFrame(vertices, edges)
s.show_step("Create a GraphFrame")

g.vertices.show(10)
g.edges.show(10)
s.show_step("Display the vertex and edge DataFrames")

vertexDegrees = g.degrees
vertexDegrees.count()
vertexDegrees.show()
s.show_step("Get a DataFrame with columns id and degree")

triangle = g.triangleCount()
triangle.count()
triangle.show()
s.show_step("Count triangles")

# Find the youngest user's age in the graph.
# This queries the vertex DataFrame.
# g.vertices.groupBy().min("x").show()

# Count the number of "follows" in the graph.
# This queries the edge DataFrame.
# numFollows = g.edges.filter("relationship = 'follow'").count()

# print("numFollows=", numFollows)
