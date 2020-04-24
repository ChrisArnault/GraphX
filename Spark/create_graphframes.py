
import random
import time
import os
import sys
import subprocess
import signal
import time
import numpy as np
import gc

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


def get_file_size(directory, file):
    import subprocess
    cmd = "hdfs dfs -du -h {} | egrep {}".format(directory, file)
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


def edge_it(vertex_number, range_vertices, degree_max):
    for v in range_vertices:
        m = random.randint(0, int(degree_max))
        j = 0
        while j < m:
            w = random.randint(0, vertex_number)
            if w != v:
                # print(v, w)
                yield (v, w)
                j += 1


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


def CellIterator(start_row, start_col, max_radius, _cells):
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


def batch_create(directory, file, build_values, columns, total_rows, batches, vertices=None, grid_size=None):
    os.system("hdfs dfs -rm -r -f {}/{}".format(directory, file))

    print("batch_create> ", directory, file, "total_rows=", total_rows, "batches=", batches)
    file_name = "{}/{}".format(directory, file)

    previous_size = 0

    loops = batches
    rows = int(total_rows / loops)
    row = 0
    
    local_stepper = Stepper()

    for batch in range(loops):
        print("batch> ", batch, " range ", row, row + rows)

        gc.collect()

        if vertices is None:
            df = sqlContext.createDataFrame(build_values(row, row + rows), columns)
            local_stepper.show_step("create dataframe")
        else:

            df = None
            for row0 in range(grid_size):
                for col0 in range(grid_size):
                    cell0 = col0 + grid_size * row0

                    src = vertices.filter(vertices.cell == cell0).alias("src").\
                        withColumnRenamed("id", "src_id").\
                        withColumnRenamed("cell", "src_cell")

                    if src.count() == 0:
                        continue

                    for r, row, col in CellIterator(row0, col0, 2, grid_size):
                        cell = col + grid_size * row
                        dst = vertices.filter(vertices.cell == cell).\
                            withColumnRenamed("id", "dst_id"). \
                            withColumnRenamed("cell", "dst_cell")

                        if src.count() == 0:
                            continue

                        # "src_id", "x", "y", "src_cell"
                        # "dst_id", "x", "y", "dst_cell"
                        # "eid", "src", "dst"

                        edges = src.join(dst, (dst.dst_id != src.src_id), how="inner").\
                                select("eid", "src", "dst")

                        if df is None:
                            df = edges
                        else:
                            df = df.union(edges)

                        """
                        df = sqlContext.createDataFrame(build_values(row, row + rows), columns).\
                            repartition(1000, "eid")
                        
                        df = df.join(src, (src.src_id == df.src), how="inner"). \
                                join(dst, (dst.dst_id == df.dst) &
                                          (dst.dst_id != src.src_id) &
                                           neighbour(grid_size, dst.dst_cell, src.src_cell),
                                     how="inner").\
                                select("eid", "src", "dst")
                        """

            local_stepper.show_step("create dataframe and join")

        if batch == 0:
            df.write.format("parquet").save(file_name)
        else:
            df.write.format("parquet").mode("append").save(file_name)

        local_stepper.show_step("Write block")

        new_size = get_file_size(directory, file)
        increment = new_size - previous_size
        previous_size = new_size
        row += rows
        print("file_size={}M increment={}M".format(new_size, increment))

    df = spark.read.format("parquet").load(file_name)
    local_stepper.show_step("Read full file")

    return df


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

randx = lambda: np.random.random()
randy = lambda: np.random.random()

cell = lambda x, y: int(x*conf.g) + conf.g * int(y*conf.g)     # cell index

# -------------- Vertices

base_vertex_values = lambda start, stop: [(v, randx(), randy()) for v in range(start, stop)]
vertex_values = lambda start, stop: [(v[0], v[1], v[2], cell(v[1], v[2])) for v in base_vertex_values(start, stop)]

stepper = Stepper()

if conf.read_vertices:
    vertices_df = spark.read.format("parquet").load("{}/{}".format(conf.graphs, "vertices"))
else:
    vertices_df = batch_create(directory=conf.graphs,
                               file="vertices",
                               build_values=vertex_values,
                               columns=["id", "x", "y", "cell"],
                               total_rows=conf.vertices,
                               batches=conf.batches_vertices)

stepper.show_step("creating vertices")

original_partitions = vertices_df.rdd.getNumPartitions()
print("original partitions # =", original_partitions)

vertices_df = vertices_df.repartition(conf.partitions, "cell")
effective_partitions = vertices_df.rdd.getNumPartitions()
print("effective partitions # =", effective_partitions)

# ---------- edges

edge_values = lambda start, stop: [(i, e[0], e[1]) for i, e in enumerate(edge_it(conf.vertices,
                                                                                 range(start, stop),
                                                                                 conf.degree_max))]

edges = batch_create(directory=conf.graphs,
                     file="edges",
                     build_values=edge_values,
                     columns=["eid", "src", "dst"],
                     total_rows=conf.vertices,
                     batches=conf.batches_edges,
                     vertices=vertices_df,
                     grid_size=conf.g)

graph = graphframes.GraphFrame(vertices_df, edges)
stepper.show_step("Create a GraphFrame")

print("count: vertices=", graph.vertices.count(), "edges=", graph.edges.count())
stepper.show_step("count GraphFrame")

graph.vertices.show()
graph.edges.show()

spark.sparkContext.stop()

