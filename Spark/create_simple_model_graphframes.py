import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import SQLContext

import random
import time
import os
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


def get_file_size(dir, file):
    import subprocess
    cmd = "hdfs dfs -du -h {} | egrep {}".format(dir, file)
    result = subprocess.check_output(cmd, shell=True).decode().split("\n")
    for line in result:
        if file in line:
            a = line.split()
            size = float(a[0])
            scale = a[1]
            if scale == 'K':
                size *= 1.0/1024.0
            elif scale == 'M':
                size *= 1
            elif scale == 'G':
                size *= 1024
            return size
    return 0


def edge_it(n, range=None):
    v = 0
    number = 0
    finished = False
    while v < n:
        m = random.randint(0, int(degree_max))
        j = 0
        while j < m:
            j += 1
            w = random.randint(0, n)
            # print(v, w)
            if range is None or (range is not None and number in range):
                yield (v, w)

            if range is not None and number >= range.stop:
                finished = False
                break

            number += 1
        v += 1
        if finished:
            break


def batch_create(dir, file, build_values, columns):
    previous_size = 0

    s = Stepper()

    for batch in range(loops):
        values = [(v, x(), y()) for v in range(num_vertices)]
        df = sqlContext.createDataFrame(values, ["id", "x", "y"])
        df = df.cache()
        df.count()
        s.show_step("building the dataframe")

        file_name = "{}/{}".format(home, file)
        if batch == 0:
            df.write.format("parquet").save(file_name)
        else:
            df.write.format("parquet").mode("append").save(file_name)
        s.show_step("Write block")

        new_size = get_file_size(file)
        increment = new_size - previous_size
        previous_size = new_size

        print("file_size={} increment={}".format(new_size, increment))

    df = spark.read.format(conf.file_format).load(conf.dest)
    s.show_step("Read full file")

    return df


os.system("hdfs dfs -rm -r -f {}/{}".format(home, "vertices"))
os.system("hdfs dfs -rm -r -f {}/{}".format(home, "edges"))

x = lambda : np.random.random()
y = lambda : np.random.random()

vertex_values = lambda : [(v, x(), y()) for v in range(num_vertices)]
s = Stepper()
vertices = batch_create(home, "vertices", vertex_values, ["id", "x", "y"]):
s.show_step("creating vertices")

"""
not finished: accumulate vertices and edges by batches
...
"""

edges = None

edge_values = lambda : [(v, w) for v, w in edge_it(num_vertices)]
edges = batch_create(home, "edges", edge_values, ["src", "dst"]):
s.show_step("creating edges")

g = graphframes.GraphFrame(vertices, edges)
s.show_step("Create a GraphFrame")

g.vertices.write.parquet(home + "/vertices")
g.edges.write.parquet(home + "/edges")
s.show_step("Save the GraphFrame")
