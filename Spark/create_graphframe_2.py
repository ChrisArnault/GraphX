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
    from pyspark.sql.functions import monotonically_increasing_id
    from matplotlib import pyplot as plt
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
        self.vertices = 1000
        self.batches_vertices = 1
        self.batches_edges = 1
        self.degree_max = self.vertices
        self.partitions = 1000
        self.file_format = "parquet"
        self.graphs_base = "/user/chris.arnault/graphs"
        self.name = "test"
        self.graphs = ""
        self.read_vertices = False
        self.grid = 10000
        self.g = int(np.sqrt(self.grid))  # row or column number of the square grid
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
  partitions|P|p = 1000
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
        [print(a, "=", getattr(self, a)) for a in dir(self) if a[0] != '_']
        if not run:
            exit()
        cmd = "hdfs dfs -mkdir {}".format(self.graphs)
        try:
            result = subprocess.check_output(cmd, shell=True).decode().split("\n")
            print(result)
        except:
            pass


def histo(df, col):
    p = df.toPandas()
    p.hist(col)
    plt.show()


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


def cell_iterator(start_row, start_col, max_radius, _cells):
    radius = 0
    while radius < max_radius:
        if radius == 0:
            lrow = start_row
            lcol = start_col
            if lrow >= 0 and lrow < _cells and lcol >= 0 and lcol < _cells:
                yield radius, lrow, lcol
        else:
            row = -radius
            for column in range(- radius, radius + 1):
                lrow = start_row + row
                lcol = start_col + column
                if lrow >= 0 and lrow < _cells and lcol >= 0 and lcol < _cells:
                    yield radius, lrow, lcol
            column = radius
            for row in range(-radius + 1, radius + 1):
                lrow = start_row + row
                lcol = start_col + column
                if lrow >= 0 and lrow < _cells and lcol >= 0 and lcol < _cells:
                    yield radius, lrow, lcol
            row = radius
            for column in range(radius - 1, -radius - 1, -1):
                lrow = start_row + row
                lcol = start_col + column
                if lrow >= 0 and lrow < _cells and lcol >= 0 and lcol < _cells:
                    yield radius, lrow, lcol
            column = -radius
            for row in range(radius - 1, -radius, -1):
                lrow = start_row + row
                lcol = start_col + column
                if lrow >= 0 and lrow < _cells and lcol >= 0 and lcol < _cells:
                    yield radius, lrow, lcol
        radius += 1


conf = Conf()
conf.set()

partitions = conf.partitions
grid_size = conf.g
all_cells = conf.grid

randx = lambda: np.random.random()
randy = lambda: np.random.random()

cell = lambda x, y: int(x*conf.g) + conf.g * int(y*conf.g)     # cell index

# -------------- Vertices

base_vertex_values = lambda start, stop: [(v, randx(), randy()) for v in range(start, stop)]
vertex_values = lambda start, stop: [(v[0], v[1], v[2], cell(v[1], v[2])) for v in base_vertex_values(start, stop)]

columns = ["id", "x", "y", "cell"]

vertices = sqlContext.createDataFrame(vertex_values(0, conf.vertices), columns).repartition(conf.grid, "cell")
n = vertices.rdd.getNumPartitions()
print("partitions=", n)

degree = np.random.randint(0, conf.degree_max)
fraction = float(degree) / conf.vertices

dst = vertices. \
    withColumnRenamed("id", "dst_id"). \
    withColumnRenamed("cell", "dst_cell"). \
    sample(False, fraction)

v2 = vertices.select("id", "cell", (vertices.cell/conf.g).alias("row"), (vertices.cell % conf.g).alias("col"))
v3 = v2.rdd.map(lambda x: [(x[0], x[1], _row, _col) for r, _row, _col in cell_iterator(int(x[2]), x[3], 2, conf.g)]).flatMap(lambda x: x)
v4 = v3.toDF().toDF("src_id", "src_cell", "src_row", "src_col")
src = v4.repartition(all_cells, "src_cell")
dst = dst.repartition(all_cells, "dst_cell")

e1 = src.join(dst, (src.src_id != dst.dst_id))
e2 = e1.repartition(all_cells, "src_cell")

e3 = e2.filter(neighbour(conf.g, src.src_cell, dst.dst_cell)).select('src_id', 'dst_id', 'src_cell', 'dst_cell').distinct()
e4 = e3.repartition(all_cells, "src_cell")
edges = e4.withColumnRenamed("src_id", "src").withColumnRenamed("dst_id", "dst").withColumn('id', monotonically_increasing_id()).select("id", "src", "dst")

conf.graphs = "{}/{}_N{}_D{}".format(conf.graphs_base,
                                     "test2",
                                     conf.vertices,
                                     conf.degree_max)

directory = conf.graphs

file_name = "{}/{}".format(directory, "vertices")
os.system("hdfs dfs -rm -r -f {}".format(file_name))
vertices.write.format("parquet").save(file_name)

file_name = "{}/{}".format(directory, "edges")
os.system("hdfs dfs -rm -r -f {}".format(file_name))
edges.write.format("parquet").save(file_name)

graph = graphframes.GraphFrame(vertices, edges)
deg = graph.degrees
p = deg.filter(deg.degree > 1).toPandas()

p.hist("degree", bins=30)
