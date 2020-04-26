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

N = 10000
D = N
partitions = 100
grid_size = 50
all_cells = grid_size*grid_size

x = lambda : np.random.random()
y = lambda : np.random.random()
cell_id = lambda x, y: int(x*grid_size) + grid_size * int(y*grid_size)


base_vertex_values = lambda : [(v, x(), y()) for v in range(N)]
vertex_values = lambda : [(v[0], v[1], v[2], cell_id(v[1], v[2])) for v in base_vertex_values()]

columns = ["id", "x", "y", "cell"]
vertices = sqlContext.createDataFrame(vertex_values(), columns).repartition(all_cells, "cell")
n = vertices.rdd.getNumPartitions()
print("partitions=", n)

vertices.show()

def CellIterator(row0, col0, max_radius, grid_size):
    radius = 0
    while radius < max_radius:
        if radius == 0:
            lrow = row0
            lcol = col0
            if lrow >= 0 and lrow < grid_size and lcol >= 0 and lcol < grid_size:
                yield radius, lrow, lcol
        else:
            row = -radius
            for column in range(- radius, radius + 1):
                lrow = row0 + row
                lcol = col0 + column
                if lrow >= 0 and lrow < grid_size and lcol >= 0 and lcol < grid_size:
                    yield radius, lrow, lcol
            column = radius
            for row in range(-radius + 1, radius + 1):
                lrow = row0 + row
                lcol = col0 + column
                if lrow >= 0 and lrow < grid_size and lcol >= 0 and lcol < grid_size:
                    yield radius, lrow, lcol
            row = radius
            for column in range(radius - 1, -radius - 1, -1):
                lrow = row0 + row
                lcol = col0 + column
                if lrow >= 0 and lrow < grid_size and lcol >= 0 and lcol < grid_size:
                    yield radius, lrow, lcol
            column = -radius
            for row in range(radius - 1, -radius, -1):
                lrow = row0 + row
                lcol = col0 + column
                if lrow >= 0 and lrow < grid_size and lcol >= 0 and lcol < grid_size:
                    yield radius, lrow, lcol
        radius += 1

cell_width = 1/grid_size
dist_max = 0.1
maxr = dist_max/cell_width

def func(p_list):
    yield p_list

vertices_rdd = vertices.rdd.mapPartitions(func)

degree = np.random.randint(0, D)
fraction = float(degree) / N

dst = vertices. \
    withColumnRenamed("id", "dst_id"). \
    withColumnRenamed("cell", "dst_cell"). \
    sample(False, fraction)

j_id = columns.index("id")
j_x = columns.index("x")
j_y = columns.index("y")
# j_row = columns.index("row")
# j_col = columns.index("col")
j_cell = columns.index("cell")

f_src_id = lambda i: i[j_id]
f_x = lambda i: i[j_x]
f_y = lambda i: i[j_y]
f_cell = lambda i: i[j_cell]
f_row = lambda i: int(i[j_cell] / grid_size)
f_col = lambda i: i[j_cell] % grid_size
#                cell  row             col
visit_cells = lambda x: [(f_src_id(i), f_x(i), f_y(i), f_cell(i), f_row(i), f_col(i)) for i in x]

z = vertices_rdd.map(lambda x: visit_cells(x))

# for i in z.collect():
#    print(i)

d = z.flatMap(lambda x : x)

# for i in d.collect():
#     print(i)

visit_neighbour_cells = lambda x: [[(src, xx, yy, cell_src, row*grid_size+col) for r, row, col in CellIterator(row, col, 2, grid_size)]
                                   for src, xx, yy, cell_src, row, col in visit_cells(x)]



full_visit = vertices_rdd.map(lambda x: visit_neighbour_cells(x))

# for i in full_visit.collect():
#    print(i)

d = full_visit.flatMap(lambda x : x)

# for i in d.collect():
#     print(i)

all_visited_cells = full_visit.flatMap(lambda x : x).flatMap(lambda x : x)

# for i in all_visited_cells.collect():
#     print(i)

dst = vertices. \
    withColumnRenamed("id", "dst_id"). \
    withColumnRenamed("cell", "dst_cell"). \
    withColumnRenamed("x", "dst_x"). \
    withColumnRenamed("y", "dst_y"). \
    sample(False, fraction)

all_edges = sqlContext.createDataFrame(all_visited_cells, ['src_id', 'src_x', 'src_y', 'src_cell', 'dst_cell'])
# ddf = ddf.join(dst, dst.dst_cell == ddf.dst_cell).select('src_id', 'dst_id')
# ddf = ddf.withColumn('id', monotonically_increasing_id())
# df = all_edges.join(dst, dst.dst_cell == all_edges.dst_cell).select('src_id', 'dst_id')

degree = np.random.randint(0, D)
fraction = float(degree) / N

df = all_edges.join(dst, (dst.dst_cell == all_edges.dst_cell) & (all_edges.src_id != dst.dst_id)). \
    withColumnRenamed("src_id", "src"). \
    withColumnRenamed("dst_id", "dst"). \
    withColumn('id', monotonically_increasing_id())


points = vertices.toPandas()
x_points = points["x"]
y_points = points["y"]

edges = df.toPandas()

e_src_x = edges["src_x"]
e_src_y = edges["src_y"]
e_dst_x = edges["dst_x"]
e_dst_y = edges["dst_y"]

plt.scatter(x_points, y_points, s=1)
e = [plt.plot((e_src_x[i], e_dst_x[i]), (e_src_y[i], e_dst_y[i])) for i, x in enumerate(e_src_x)]

plt.show()
