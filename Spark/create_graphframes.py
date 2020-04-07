
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

    # .set("spark.local.dir", "/tmp/spark-temp");

    sqlContext = SQLContext(spark.sparkContext)

class Conf(object):
    def __init__(self):
        self.vertices = 1000
        self.edges = 1000
        self.batches_vertices = 1
        self.batches_edges = 1
        self.degree_max = 100
        self.partitions = 1000
        self.file_format = "parquet"
        self.graphs_base = "/user/chris.arnault/graphs"
        self.name = "test"
        self.graphs = ""
        self.read_vertices = False

    def set(self):
        for i, arg in enumerate(sys.argv[1:]):
            a = arg.split("=")
            # print(i, arg, a)
            key = a[0]
            if key == "vertices" or key == "N" or key == "n":
                self.vertices = int(a[1])
            elif key == "edges" or key == "E" or key == "e":
                self.edges = int(a[1])
            elif key == "batches_vertices" or key == "BN" or key == "bn":
                self.batches_vertices = int(a[1])
            elif key == "batches_edges" or key == "BE" or key == "be":
                self.batches_edges = int(a[1])
            elif key == "degree_max" or key == "D" or key == "d":
                self.degree_max = int(a[1])
            elif key == "partitions" or key == "P" or key == "p":
                self.partitions = int(a[1])
            elif key == "name" or key == "F" or key == "f":
                self.name = a[1]
            elif key == "read_vertices" or key == "R" or key == "r":
                self.read_vertices = True
            elif key[:2] == "-h" or key[0] == "h":
                print('''
> python create_graphfames.py 
  vertices|N = 1000
  edges|E = 1000
  batches_vertices|BN = 1
  batches_edges|BE = 1
  degree_max|D = 100
  partitions|P = 1000
  name|F = "test"
  read_vertices|R = False
                ''')
                exit()


        self.graphs = "{}/{}_N{}_BN{}_BE{}_D{}".format(self.graphs_base,
                                                       self.name,
                                                       self.vertices,
                                                       self.batches_vertices,
                                                       self.batches_edges,
                                                       self.degree_max)
        print("graphs={}".format(self.graphs))

        [print(a, "=", getattr(conf, a)) for a in dir(conf) if a[0] != '_']

        cmd = "hdfs dfs -mkdir {}".format(self.graphs)

        try:
            result = subprocess.check_output(cmd, shell=True).decode().split("\n")
            print(result)
        except:
            pass


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


def edge_it(vertices, range_vertices, degree_max):
    for v in range_vertices:
        m = random.randint(0, int(degree_max))
        for j in range(m):
            w = random.randint(0, vertices)
            # print(v, w)
            yield (v, w)


def batch_create(dir, file, build_values, columns, total_rows, batches):
    os.system("hdfs dfs -rm -r -f {}/{}".format(dir, file))

    print("batch_create> ", dir, file, "total_rows=", total_rows, "batches=", batches)
    file_name = "{}/{}".format(dir, file)

    previous_size = 0

    loops = batches
    rows = int(total_rows / loops)
    row = 0
    
    s = Stepper()

    for batch in range(loops):
        print("batch> ", batch, " range ", row, row + rows)
        df = sqlContext.createDataFrame(build_values(row, row + rows), columns)
        df = df.cache()
        df.count()
        s.show_step("building the dataframe")

        if batch == 0:
            df.write.format("parquet").save(file_name)
        else:
            df.write.format("parquet").mode("append").save(file_name)
        s.show_step("Write block")

        new_size = get_file_size(dir, file)
        increment = new_size - previous_size
        previous_size = new_size
        row += rows
        print("file_size={} increment={}".format(new_size, increment))

    df = spark.read.format("parquet").load(file_name)
    s.show_step("Read full file")

    return df

conf = Conf()
conf.set()

if not has_spark:
    exit()

x = lambda : np.random.random()
y = lambda : np.random.random()

vertex_values = lambda start, stop: [(v, x(), y()) for v in range(start, stop)]
s = Stepper()

if conf.read_vertices:
    vertices = spark.read.format("parquet").load("{}/{}".format(conf.graphs, "vertices"))
else:
    vertices = batch_create(conf.graphs, "vertices", vertex_values, ["id", "x", "y"], conf.vertices, conf.batches_vertices)

s.show_step("creating vertices")

"""
not finished: accumulate vertices and edges by batches
...
"""

edges = None

edge_values = lambda start, stop : [(v, w) for v, w in edge_it(conf.vertices, range(start, stop), conf.degree_max)]
edges = batch_create(conf.graphs, "edges", edge_values, ["src", "dst"], conf.vertices, conf.batches_edges







                     )
s.show_step("creating edges")

g = graphframes.GraphFrame(vertices, edges)
s.show_step("Create a GraphFrame")

print("count:", g.vertices.count(), g.edges.count())
s.show_step("count GraphFrame")

g.vertices.show()
g.edges.show()

