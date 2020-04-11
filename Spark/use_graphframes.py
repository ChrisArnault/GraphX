import numpy as np
import time
import os

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

    sqlContext = SQLContext(spark.sparkContext)


class Conf(object):
    def __init__(self):
        self.partitions = 300
        self.graphs_base = "/user/chris.arnault/graphs"
        self.name = "test"
        self.graphs = ""

    def set(self):
        run = True

        for i, arg in enumerate(sys.argv[1:]):
            a = arg.split("=")
            # print(i, arg, a)
            key = a[0]
            if key == "partitions" or key == "P" or key == "p":
                self.partitions = int(a[1])
            elif key == "name" or key == "F" or key == "f":
                self.name = a[1]
            elif key == "Args" or key == "args" or key == "A" or key == "a":
                run = False
            elif key[:2] == "-h" or key[0] == "h":
                print('''
> python create_graphfames.py 
  partitions|P|p = 300
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

s = Stepper()

conf = Conf()
conf.set()

file_name = conf.graphs_base + "/" + conf.name

exit()

vertices = sqlContext.read.parquet(file_name + "/vertices")
edges = sqlContext.read.parquet(file_name + "/edges")
s.show_step("Load the vertices and edges back.")

# Create an identical GraphFrame.
g = graphframes.GraphFrame(vertices, edges)
s.show_step("Create a GraphFrame")

g.vertices.show(10)
g.edges.show(10)
s.show_step("Display the vertex and edge DataFrames")

vertexDegrees = g.degrees
vertexDegrees.count()
vertexDegrees.show()
s.show_step("Get a DataFrame with columns id and degree")

triangles = g.triangleCount()
c = triangles.count()
triangles.show()
s.show_step("Get triangle count", c)

spark.sparkContext.stop()
