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

N = 1000
D = 1000
partitions = 100
cells = 10
all_cells = cells*cells

x = lambda : np.random.random()
y = lambda : np.random.random()
cell_id = lambda x, y: int(x*cells) + cells * int(y*cells)


base_vertex_values = lambda : [(v, x(), y()) for v in range(N)]
vertex_values = lambda : [(v[0], v[1], v[2], cell_id(v[1], v[2])) for v in base_vertex_values()]

vertices = sqlContext.createDataFrame(vertex_values(), ["id", "x", "y", "cell"])
n = vertices.rdd.getNumPartitions()
print("partitions=", n)

vertices = vertices.repartition(partitions, "cell")
n = vertices.rdd.getNumPartitions()
print("partitions=", n)

vertices.show()

"""
for c in range(all_cells):
    count = vertices.filter(vertices.cell == c).count()
    if count > 0:
        print(c, count)
"""

def CellIterator(row0, col0, max_radius, cells):
    radius = 0
    while radius < max_radius:
        if radius == 0:
            lrow = row0
            lcol = col0
            if lrow >= 0 and lrow < cells and lcol >= 0 and lcol < cells:
                yield radius, lrow, lcol
        else:
            row = -radius
            for column in range(- radius, radius + 1):
                lrow = row0 + row
                lcol = col0 + column
                if lrow >= 0 and lrow < cells and lcol >= 0 and lcol < cells:
                    yield radius, lrow, lcol
            column = radius
            for row in range(-radius + 1, radius + 1):
                lrow = row0 + row
                lcol = col0 + column
                if lrow >= 0 and lrow < cells and lcol >= 0 and lcol < cells:
                    yield radius, lrow, lcol
            row = radius
            for column in range(radius - 1, -radius - 1, -1):
                lrow = row0 + row
                lcol = col0 + column
                if lrow >= 0 and lrow < cells and lcol >= 0 and lcol < cells:
                    yield radius, lrow, lcol
            column = -radius
            for row in range(radius - 1, -radius, -1):
                lrow = row0 + row
                lcol = col0 + column
                if lrow >= 0 and lrow < cells and lcol >= 0 and lcol < cells:
                    yield radius, lrow, lcol
        radius += 1

"""
for r, row, col in CellIterator(5, 8, 3, 8):
    print(r, row, col)
"""

cell_width = 1/cells
dist_max = 0.1
maxr = dist_max/cell_width

total = 0
eid = 0
df = None
for row0 in range(cells):
    for col0 in range(cells):
        cell0 = col0 + cells * row0
        src = vertices.filter(vertices.cell == cell0).\
            withColumnRenamed("id", "src_id"). \
            withColumnRenamed("cell", "src_cell")
        if src.count() == 0:
            continue
        # src.show()
        for r, row, col in CellIterator(row0, col0, 2, cells):
            cell = col + cells * row
            degree = np.random.randint(0, D)
            fraction = float(degree)/N
            print("fraction=", fraction)
            dst = vertices.filter(vertices.cell == cell).\
                withColumnRenamed("id", "dst_id"). \
                withColumnRenamed("cell", "dst_cell").\
                sample(False, fraction)
            count = 0
            if dst.count() == 0:
                continue
            # dst.show()
            edges = src.join(dst, (dst.dst_id != src.src_id), how="inner"). \
                select('src_id', "dst_id").\
                withColumnRenamed("src_id", "src").\
                withColumnRenamed("dst_id", "dst")
            if df is None:
                df = edges
            else:
                df = df.union(edges)
            count = edges.count()
            total += count
            print("r0=", row0, "c0=", col0, "r=", row, "c=", col, "count=", count, "total=", total)
        df.show()


"""
for i in l:
    yield i
"""

def func(p_list):
    yield p_list

x = vertices.repartition(partitions, vertices.id % 10)
y = x.rdd.mapPartitions(func)

degree = np.random.randint(0, D)
fraction = float(degree) / N

dst = vertices. \
    withColumnRenamed("id", "dst_id"). \
    withColumnRenamed("cell", "dst_cell"). \
    sample(False, fraction)

#                cell  row             col
f1 = lambda x: [(i[0], i[3], int(i[3]/cells), i[3]%cells) for i in x]
f2 = lambda x: [[(src, cell_src, row*cells+col) for r, row, col in CellIterator(row, col, 2, cells)] for src, cell_src, row, col in f1(x)]
z = y.map(lambda x: f2(x))
c = z.collect()
d = z.flatMap(lambda x : x).flatMap(lambda x : x)
ddf = sqlContext.createDataFrame(d, ['src_id', 'src_cell', 'dst_cell'])
# ddf = ddf.join(dst, dst.dst_cell == ddf.dst_cell).select('src_id', 'dst_id')
# ddf = ddf.withColumn('id', monotonically_increasing_id())
df = ddf.join(dst, dst.dst_cell == ddf.dst_cell).select('src_id', 'dst_id'). \
    withColumnRenamed("src_id", "src"). \
    withColumnRenamed("dst_id", "dst"). \
    withColumn('id', monotonically_increasing_id())

for ic in c:
    if len(ic) == 0:
        continue
    print("---------")
    for jc in ic:
        print("a", jc)

"""
[[], [], [], [], [], 
[Row(id=35, x=0.07413584590669997, y=0.0005590032288271818, cell=0), Row(id=566, x=0.0871836469225078, y=0.06884292555826566, cell=0), Row(id=736, x=0.06935023370130589, y=0.09140834057995562, cell=0), Row(id=738, x=0.04855167720496423, y=0.025231678227210508, cell=0), Row(id=46, x=0.014870555986308709, y=0.027513331049781042, cell=0), Row(id=102, x=0.09684039402638045, y=0.008138223824768032, cell=0), Row(id=848, x=0.015988584529946115, y=0.004414593917120846, cell=0), Row(id=577, x=0.061422084104769126, y=0.08981124326951317, cell=0), Row(id=632, x=0.08776387926398022, y=0.047861308012999815, cell=0)], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
"""

mylist = np.random.random(20)
rdd = sc.parallelize(mylist)
t = rdd.mapPartitions(func)
for i in t.collect():
    print(i)


def edge_it(vertices, range_vertices, degree_max):
    for v in range_vertices:
        m = random.randint(0, int(degree_max))
        for j in range(m):
            w = random.randint(0, vertices)
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


