import random
import time
import os
import sys
import subprocess
import gc

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
        self.partitions = 100
        self.file_format = "parquet"
        self.graphs_base = "/user/chris.arnault/graphs"
        self.name = "test2"
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
> python create_graphfames_2.py 
  vertices|N|n = 1000
  degree_max|D|d = 1000
  partitions|P|p = 1000
  grid|G|g = 10000
  name|F|f = "test2"
  read_vertices|R|r = False
  Args|args|A|a  print only args (no run)
                ''')
                exit()
        print("graphs={}".format(self.graphs))
        conf.graphs = "{}/{}_N{}_D{}".format(conf.graphs_base,
                                             conf.name,
                                             conf.vertices,
                                             conf.degree_max)
        [print(a, "=", getattr(self, a)) for a in dir(self) if a[0] != '_']
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


def histo(df, col):
    _p = df.toPandas()
    _p.hist(col)
    plt.show()


def xc(c, g):
    return c % g


def yc(c, g):
    return c / g


def cyclic_neighbour(g, c1, c2) -> bool:
    """
    - A square cyclic matrix of g cells
    - test if the two cells c1 & c2 are neighbours
    - cases:
       + C center
       + T top
       + B bottom
       + L left
       + R right
       + TL, TR, BL, BR
    """
    t: bool = c1 == c2
    if t:
        return True                 # C, C
    dx = abs(xc(c1, g) - xc(c2, g))
    dy = abs(yc(c1, g) - yc(c2, g))
    t = (dx == 0) & (dy == 1)
    if t:
        return True                 # C, C
    t = (dx == 0) & (dy == g - 1)
    if t:
        return True                 # T, B
    t = (dx == 1) & (dy == 0)
    if t:
        return True                 # C, C
    t = (dx == g - 1) & (dy == 0)
    if t:
        return True                 # L, R
    t = (dx == 1) & (dy == 1)
    if t:
        return True                 # C, C
    t = (dx == 1) & (dy == g - 1)
    if t:
        return True                 # T, B
    t = (dx == g - 1) & (dy == 1)
    if t:
        return True                 # L, R
    t = (dx == g - 1) & (dy == g - 1)
    return t                        # TL, TR, BL, BR


def neighbour(g, c1, c2) -> bool:
    """
    - A square non-cyclic matrix of g cells
    - test if the two cells c1 & c2 are neighbours
    - cases:
       + C center
       + T top
       + B bottom
       + L left
       + R right
       + TL, TR, BL, BR
    """
    t: bool = c1 == c2
    if t:
        return True                 # C, C
    dx = abs(xc(c1, g) - xc(c2, g))
    dy = abs(yc(c1, g) - yc(c2, g))
    t = (dx == 0) & (dy == 1)
    if t:
        return True                 # C, C
    t = (dx == 1) & (dy == 0)
    if t:
        return True                 # C, C
    t = (dx == 1) & (dy == 1)
    return t                        # TL, TR, BL, BR


def cell_iterator(start_row, start_col, max_radius, _cells):
    """
    - A square matrix of cells
    - iterate increasing radius around a starting cell (start_row, start_col) counter-clock wise
    - at each radius start from BL corner
    - up to a max_radius
    """
    radius = 0
    while radius < max_radius:
        if radius == 0:
            lrow = start_row
            lcol = start_col
            if (lrow >= 0) and (lrow < _cells) and (lcol >= 0) and (lcol < _cells):
                yield radius, lrow, lcol
        else:
            row = -radius
            for column in range(- radius, radius + 1):
                lrow = start_row + row
                lcol = start_col + column
                if (lrow >= 0) and (lrow < _cells) and (lcol >= 0) and (lcol < _cells):
                    yield radius, lrow, lcol
            column = radius
            for row in range(-radius + 1, radius + 1):
                lrow = start_row + row
                lcol = start_col + column
                if (lrow >= 0) and (lrow < _cells) and (lcol >= 0) and (lcol < _cells):
                    yield radius, lrow, lcol
            row = radius
            for column in range(radius - 1, -radius - 1, -1):
                lrow = start_row + row
                lcol = start_col + column
                if (lrow >= 0) and (lrow < _cells) and (lcol >= 0) and (lcol < _cells):
                    yield radius, lrow, lcol
            column = -radius
            for row in range(radius - 1, -radius, -1):
                lrow = start_row + row
                lcol = start_col + column
                if (lrow >= 0) and (lrow < _cells) and (lcol >= 0) and (lcol < _cells):
                    yield radius, lrow, lcol
        radius += 1


conf = Conf()
conf.set()

partitions = conf.partitions
grid_size = conf.g

stepper = Stepper()

randx = lambda: np.random.random()
randy = lambda: np.random.random()

cell = lambda x, y: int(x*conf.g) + conf.g * int(y*conf.g)     # cell index

# =============================== Vertices

base_vertex_values = lambda start, stop: [(v, randx(), randy()) for v in range(start, stop)]
vertex_values = lambda start, stop: [(v[0], v[1], v[2], cell(v[1], v[2])) for v in base_vertex_values(start, stop)]

columns = ["id", "x", "y", "cell"]

vertices = sqlContext.createDataFrame(vertex_values(0, conf.vertices), columns).repartition(partitions, "id")
stepper.show_step("creating vertices")

n = vertices.rdd.getNumPartitions()
print("partitions=", n)

# save vertices
file_name = "{}/{}".format(conf.graphs, "vertices")
os.system("hdfs dfs -rm -r -f {}".format(file_name))
vertices.write.format("parquet").save(file_name)

stepper.show_step("save vertices")

# =============================== edges

degree = np.random.randint(0, conf.degree_max)
fraction = float(degree) / conf.vertices

# select a fraction of all vertices to make them destination edges
dst = vertices. \
    withColumnRenamed("id", "dst_id"). \
    withColumnRenamed("cell", "dst_cell"). \
    sample(False, fraction)

iterate_cells = lambda x: [(x[0], x[1], _row, _col) for r, _row, _col in cell_iterator(int(x[2]), x[3], 2, conf.g)]

partitions = 100
edges = None

while True:
    try:
        print("try with partitions=", partitions)
        # cleanup GC
        gc.collect()
        print("split the space into a matrix of square [g x g] cells")
        stepper.show_step("start edges")
        v2 = vertices.select("id", "cell", (vertices.cell/conf.g).alias("row"), (vertices.cell % conf.g).alias("col"))
        print("iterate around all cell (radius max = 2 => only one layer of neighbour cells)")
        v3 = v2.rdd.map(iterate_cells).flatMap(lambda x: x)
        print("rename columns")
        v4 = v3.toDF().toDF("src_id", "src_cell", "src_row", "src_col")
        print("repartitions")
        src = v4.repartition(partitions, "src_id")
        dst = dst.repartition(partitions, "dst_id")
        print("join src & dst to create edges")
        e1 = src.join(dst, (src.src_id != dst.dst_id))
        print("select only neighbour cells")
        e2 = e1.withColumn("xc1", xc(e1.src_cell, conf.g)).\
            withColumn("yc1", yc(e1.src_cell, conf.g)).\
            withColumn("xc2", xc(e1.dst_cell, conf.g)).\
            withColumn("yc2", yc(e1.dst_cell, conf.g))
        e3 = e2.withColumn("dx", abs(e2.xc1 - e2.xc2)).\
            withColumn("dy", abs(e2.yc1 - e2.yc2))
        e4 = e3.filter((e3.src_cell == e3.dst_cell) |
                       ((e3.dx == 0) & (e3.dy == 1)) |
                       ((e3.dx == 1) & (e3.dy == 0)) |
                       ((e3.dx == 1) & (e3.dy == 1)))
        e5 = e4.select('src_id', 'dst_id', 'src_cell', 'dst_cell').distinct()
        e6 = e5.repartition(partitions, "src_id")
        print("format edges")
        edges = e6.withColumnRenamed("src_id", "src").\
            withColumnRenamed("dst_id", "dst").\
            withColumn('id', monotonically_increasing_id()).\
            select("id", "src", "dst")
        stepper.show_step("created edges")
        file_name = "{}/{}".format(conf.graphs, "edges")
        os.system("hdfs dfs -rm -r -f {}".format(file_name))
        edges.write.format("parquet").save(file_name)
        stepper.show_step("saved edges")
        break
    except:
        partitions *= 2
        if partitions > int(conf.vertices/10):
            print("Memory exhausted partitions=", partitions)
            break


if edges is not None:
    graph = graphframes.GraphFrame(vertices, edges)
    deg = graph.degrees
    stepper.show_step("count degrees")
    p = deg.filter(deg.degree > 1).toPandas()
    p.hist("degree", bins=30)
    stepper.show_step("histo edges")


spark.sparkContext.stop()
