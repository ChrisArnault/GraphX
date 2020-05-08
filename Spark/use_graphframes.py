import numpy as np
from matplotlib import pyplot as plt
import time
import os
import sys
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
    # spark.sparkContext.setCheckpointDir("/lsst/data/tmp")

    sqlContext = SQLContext(spark.sparkContext)


class Conf(object):
    def __init__(self):
        self.partitions = 300
        self.graphs_base = "/user/chris.arnault/graphs"
        self.name = "test2"
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



class Stepper(object):
    previous_time = None

    def __init__(self):
        self.previous_time = time.time()

    def show_step(self, label='Initial time') -> float:
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


def do_degrees(g: graphframes.GraphFrame, s: Stepper) -> None:
    vertexDegrees = g.degrees
    degrees = vertexDegrees.count()
    # vertexDegrees.show()
    histo(vertexDegrees, "degree")

    s.show_step("Get a DataFrame with columns id and degree")
    meandf = vertexDegrees.agg({"degree": "mean"})
    meandf.show()
    mean = meandf.toPandas()["avg(degree)"][0]
    s.show_step("mean degree {}".format(mean))


def do_triangles(conf: Conf, g: graphframes.GraphFrame, s: Stepper, vertices_count: int) -> None:
    """
    Pattern for batch oriented iteration

    - we split the graph into batches using the filterVertices mechanism
    - we mark the total count of triangles and the partial count
    - in case of error:
       * we double the number of batches and the batch number
       * we restart the iteration at this point with smaller subgraph
    """

    full_set = vertices_count
    batches = conf.batches_for_triangles
    total_triangles = conf.count_at_restart
    batch = conf.batch_at_restart
    subset = int(full_set / batches)

    while batch < batches:
        st = Stepper()
        count = 0
        try:
            print("try batches=", batches, "subset=", subset, "at batch=", batch)
            gc.collect()
            # g1 = g.filterVertices("int(cell/{}) == {}".format(subset, batch))
            g1 = g.filterVertices("int(id/{}) == {}".format(subset, batch))
            triangles = g1.triangleCount()
            st.show_step("partial triangleCount")
            gc.collect()
            count = triangles.agg({"cell": "sum"}).toPandas()["sum(cell)"][0]
            st.show_step("partial triangleCount sum")

            total_triangles += count

            print("batch=", batch,
                  "vertices=", g1.vertices.count(),
                  "edges=", g1.edges.count(),
                  "total=", total_triangles,
                  "partial", count)
        except:
            print("memory error")
            batches *= 2
            batch *= 2
            subset = int(full_set / batches)
            print("restarting with batches=", batches, "subset=", subset, "at batch=", batch)
            if subset >= 1:
                continue
            break

        batch += 1

    s.show_step("triangleCount")
    print("total=", total_triangles)


s = Stepper()

conf = Conf()
conf.set()

file_name = conf.graphs_base + "/" + conf.name

print("file_name=", file_name)

vertices = sqlContext.read.parquet(file_name + "/vertices")
edges = sqlContext.read.parquet(file_name + "/edges")
s.show_step("Load the vertices and edges back.")

# Create an identical GraphFrame.
g = graphframes.GraphFrame(vertices, edges)
s.show_step("Create a GraphFrame")

g.vertices.show(10)
g.edges.show(10)
s.show_step("Display the vertex and edge DataFrames")

vertices_count = vertices.count()
print("vertices=", vertices_count)

if conf.degrees:
    do_degrees(g=g, s=s)

if conf.triangles:
    do_triangles(conf=conf, g=g, s=s, vertices_count=vertices_count)

spark.sparkContext.stop()

