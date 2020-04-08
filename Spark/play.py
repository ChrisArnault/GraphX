import random
import time
import os
import sys
import subprocess

import numpy as np

has_spark = os.name != 'nt'

if has_spark:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import *
    from pyspark.sql.types import *
    from pyspark.sql import SQLContext

    import graphframes

if has_spark:
    spark = SparkSession.builder.appName("GraphX").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    spark.sparkContext.setCheckpointDir("/tmp")
    sqlContext = SQLContext(spark.sparkContext)
    spark.conf.set("spark.sql.crossJoin.enabled", True)

N = 1000 * 100
D = 1000
Grid = 10000
partitions = 100

x = lambda : np.random.random()
y = lambda : np.random.random()
g = int(np.sqrt(Grid))
area = lambda x, y: int(x*g) + g * int(y*g)


base_vertex_values = lambda : [(v, x(), y()) for v in range(N)]
vertex_values = lambda : [(v[0], v[1], v[2], area(v[1], v[2])) for v in base_vertex_values()]

vertices = sqlContext.createDataFrame(vertex_values(), ["id", "x", "y", "area"])
n = vertices.rdd.getNumPartitions()
print("partitions=", n)

vertices = vertices.repartition(partitions, "area")
n = vertices.rdd.getNumPartitions()
print("partitions=", n)

vertices.show()


def edge_it(vertices, range_vertices, degree_max):
    for v in range_vertices:
        m = random.randint(0, int(degree_max))
        for j in range(m):
            w = random.randint(0, vertices)
            # print(v, w)
            yield (v, w)




edge_values = lambda : [(i, e[0], e[1]) for i, e in enumerate(edge_it(N, range(0, N), D))]

edges = sqlContext.createDataFrame(edge_values(), ["id", "src", "dst"])
edges = edges.repartition(partitions, "id")

edges.show()

src = vertices.alias("src")
df = src.join(edges, (src.id == edges.src), how="inner")
df.show()

dst = vertices.alias("dst")
df = dst.join(df, [(dst.id == df.dst)&(dst.area == df.area)], how="inner")

df.show()


spark.sparkContext.stop()


