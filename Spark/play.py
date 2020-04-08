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

N = 1000 * 1000
x = lambda : np.random.random()
y = lambda : np.random.random()
partitions = 100
g = int(np.sqrt(partitions))
area = lambda x, y: int(x*g) + g * int(y*g)


base_vertex_values = lambda : [(v, x(), y()) for v in range(N)]
vertex_values = lambda : [(v[0], v[1], v[2], area(v[1], v[2])) for v in base_vertex_values()]

df = sqlContext.createDataFrame(vertex_values(), ["id", "x", "y", "area"])
n = df.rdd.getNumPartitions()
print("partitions=", n)

df = df.repartition(partitions, "area")
n = df.rdd.getNumPartitions()
print("partitions=", n)

df.show()

spark.sparkContext.stop()


