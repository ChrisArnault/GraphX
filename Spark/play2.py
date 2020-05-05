import random
import time
import os
import sys
import subprocess

import numpy as np
import matplotlib.pyplot as plt

has_spark = os.name != 'nt'

if has_spark:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import *
    from pyspark.sql.types import *
    from pyspark.sql import SQLContext
    from pyspark.sql.functions import monotonically_increasing_id
    import graphframes

if has_spark:
    spark = SparkSession.builder.appName("GraphX").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    spark.sparkContext.setCheckpointDir("/tmp")
    sqlContext = SQLContext(spark.sparkContext)
    spark.conf.set("spark.sql.crossJoin.enabled", True)

def wait():
    while True:
        print("wait")
        time.sleep(3)

class Conf(object):
    def __init__(self):
        self.partitions = 300
        self.graphs_base = "/user/chris.arnault/graphs"
        self.name = "test"
        self.batches_for_triangles = 1
        self.batch_at_restart = 0
        self.count_at_restart = 0
        self.degrees = True
        self.triangles = True
        self.graphs = ""
    def set(self) -> None:
        run = True
        for i, arg in enumerate(sys.argv[1:]):
            a = arg.split("=")
            # print(i, arg, a)
            key = a[0]
            if key == "partitions" or key == "P" or key == "p":
                self.partitions = int(a[1])
            elif key == "name" or key == "F" or key == "f":
                self.name = a[1]
            elif key == "D" or key == "d":
                # only degree
                self.degrees = True
                self.triangles = False
            elif key == "BT" or key == "bt":
                # batches for triangles
                self.batches_for_triangles = int(a[1])
            elif key == "BS" or key == "bs":
                # Batch number at restart
                self.batch_at_restart = int(a[1])
            elif key == "BC" or key == "bc":
                # Triangle count at Batch restart
                self.count_at_restart = int(a[1])
            elif key == "Args" or key == "args" or key == "A" or key == "a":
                run = False
            elif key[:2] == "-h" or key[0] == "h":
                print('''
> python create_graphfames.py 
  partitions|P|p = 300
  BT|bt = 1          (batches for triangles)
  BS|bs = 0          (for triangles: restart from batch number)
  BC|bc = 0          (count for triangles at restart)
  D|d = 0            (only degrees)
  name|F|f = "test"
                ''')
                exit()
        [print(a, "=", getattr(self, a)) for a in dir(self) if a[0] != '_']
        if not run:
            exit()

conf = Conf()
conf.set()

conf.name = 'test_N100000_BN10_BE1_D100000_G10000'

file_name = conf.graphs_base + "/" + conf.name

print("file_name=", file_name)

vertices = sqlContext.read.parquet(file_name + "/vertices")
edges = sqlContext.read.parquet(file_name + "/edges")

# Create an identical GraphFrame.
g = graphframes.GraphFrame(vertices, edges)

vertexDegrees = g.degrees

p = vertexDegrees.toPandas()
p.hist("degree", bins=30)

plt.show()

spark.sparkContext.stop()


