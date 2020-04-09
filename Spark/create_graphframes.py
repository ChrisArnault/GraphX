
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
    spark.conf.set("spark.sql.crossJoin.enabled", True)

    sqlContext = SQLContext(spark.sparkContext)

class Conf(object):
    def __init__(self):
        self.vertices = 1000
        self.batches_vertices = 1
        self.batches_edges = 1
        self.degree_max = 100
        self.partitions = 300
        self.file_format = "parquet"
        self.graphs_base = "/user/chris.arnault/graphs"
        self.name = "test"
        self.graphs = ""
        self.read_vertices = False
        self.grid = 10000

    def set(self):

        run = True

        for i, arg in enumerate(sys.argv[1:]):
            a = arg.split("=")
            # print(i, arg, a)
            key = a[0]
            if key == "vertices" or key == "N" or key == "n":
                self.vertices = int(a[1])
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
            elif key == "grid" or key == "G" or key == "g":
                self.grid = int(a[1])
            elif key == "Args" or key == "args" or key == "A" or key == "a":
                run = False
            elif key[:2] == "-h" or key[0] == "h":
                print('''
> python create_graphfames.py 
  vertices|N|n = 1000
  batches_vertices|BN|bn = 1
  batches_edges|BE|be = 1
  degree_max|D|d = 100
  partitions|P|p = 300
  grid|G|g = 10000
  name|F|f = "test"
  read_vertices|R|r = False
  Args|args|A|a  print only args (no run)
                ''')
                exit()


        print("graphs={}".format(self.graphs))
        self.graphs = "{}/{}_N{}_BN{}_BE{}_D{}_G{}".format(self.graphs_base,
                                                           self.name,
                                                           self.vertices,
                                                           self.batches_vertices,
                                                           self.batches_edges,
                                                           self.degree_max,
                                                           self.grid)

        [print(a, "=", getattr(conf, a)) for a in dir(conf) if a[0] != '_']

        if not run:
            exit()

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
        j = 0
        while j < m:
            w = random.randint(0, vertices)
            if w != v:
                # print(v, w)
                yield (v, w)
                j += 1


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


def batch_update(dir, file, df):
    print("batch_update> ", dir, file)

    os.system("hdfs dfs -rm -r -f {}/{}".format(dir, file))

    file_name = "{}/{}".format(dir, file)

    df.write.format("parquet").save(file_name)


def xc(c, g):
    return c % g

def yc(c, g):
    return c / g

def neighbour(g, c1, c2):
    t1 = c1 == c2

    dx = abs(xc(c1, g) - xc(c2, g))
    dy = abs(yc(c1, g) - yc(c2, g))

    t2 = (dx == 0) & (dy == 1)
    t3 = (dx == 0) & (dy == g - 1)
    t4 = (dx == 1) & (dy == 0)
    t5 = (dx == g - 1) & (dy == 0)

    t6 = (dx == 1) & (dy == 1)
    t7 = (dx == 1) & (dy == g - 1)
    t8 = (dx == g - 1) & (dy == 1)
    t9 = (dx == g - 1) & (dy == g - 1)

    # print(t1, t2, t3, t4, t5, t6, t7, t8, t9)
    return t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9

import signal
import time

def handler(signum, frame):
    print("CTL-C received")
    spark.sparkContext.stop()
    exit()


signal.signal(signal.SIGINT, handler)


conf = Conf()
conf.set()

if not has_spark:
    exit()

"""
Generic lambda functions
"""

x = lambda : np.random.random()
y = lambda : np.random.random()

g = int(np.sqrt(conf.grid))                     # row or column number of the square grid
G = g*g                                         # grid size
cell = lambda x, y: int(x*g) + g * int(y*g)     # cell index

# -------------- Vertices

base_vertex_values = lambda start, stop: [(v, x(), y()) for v in range(start, stop)]
vertex_values = lambda start, stop: [(v[0], v[1], v[2], cell(v[1], v[2])) for v in base_vertex_values(start, stop)]

s = Stepper()

if conf.read_vertices:
    vertices = spark.read.format("parquet").load("{}/{}".format(conf.graphs, "vertices"))
else:
    vertices = batch_create(dir=conf.graphs,
                            file="vertices",
                            build_values=vertex_values,
                            columns=["id", "x", "y", "cell"],
                            total_rows=conf.vertices,
                            batches=conf.batches_vertices)

s.show_step("creating vertices")

original_partitions = vertices.rdd.getNumPartitions()
print("original partitions # =", original_partitions)

vertices = vertices.repartition(conf.partitions, "cell")
effective_partitions = vertices.rdd.getNumPartitions()
print("effective partitions # =", effective_partitions)

# ---------- edges

edges = None

edge_values = lambda start, stop : [(i, e[0], e[1]) for i, e in enumerate(edge_it(conf.vertices,
                                                                                  range(start, stop),
                                                                                  conf.degree_max))]

edges = batch_create(dir=conf.graphs,
                     file="edges_temp",
                     build_values=edge_values,
                     columns=["eid", "src", "dst"],
                     total_rows=conf.vertices,
                     batches=conf.batches_edges)

edges = edges.repartition(conf.partitions, "eid")
s.show_step("creating edges")

print("count before filter: vertices=", vertices.count(), "edges=", edges.count())

# ---------- filter edges by cell neighbourhood

src = vertices.alias("src")  # "id", "x", "y", "cell"
filtered_src = src.join(edges, (src.id == edges.src), how="inner") # "id", "x", "y", "cell", "eid", "src", "dst"

print("==== filtered_src>")
filtered_src.show()
s.show_step("join src")

dst = vertices.alias("dst")    # "id", "x", "y", "cell"

filtered = dst.join(filtered_src,
                    (dst.id != filtered_src.id) & (dst.id == filtered_src.dst) & neighbour(g, dst.cell, filtered_src.cell),
                    how="inner")

s.show_step("join dst")

print("==== filtered>")
filtered.show()

filtered_edges = filtered.select("eid", "src", "dst")

print("==== filtered_edges>")
filtered_edges.show()

batch_update(dir=conf.graphs,
             file="edges",
             df=filtered_edges)

g = graphframes.GraphFrame(vertices, filtered_edges)
s.show_step("Create a GraphFrame")

print("count: vertices=", g.vertices.count(), "edges=", g.edges.count())
s.show_step("count GraphFrame")

g.vertices.show()
g.edges.show()

spark.sparkContext.stop()


