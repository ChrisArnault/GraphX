import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import SQLContext

import graphframes
# from graphframes.examples import Graphs

spark = SparkSession.builder.appName("GraphX").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

sqlContext = SQLContext(spark.sparkContext)

num_vertices = 1000
distance_max = 0.1

x = lambda : np.random.random()
y = lambda : np.random.random()

v = sqlContext.createDataFrame([(v_id, x(), y()) for v_id in range(num_vertices)], ["id", "x", "y"])

"""
    for vertex in vertices:
        vertex.out_vertices = [v for v in random.sample(vertices, np.random.randint(0, np.sqrt(len(vertices)))) if vertex.dist(v) < distance_max]
"""

e = sqlContext.createDataFrame([(v_id, x(), y()) for v_id in range(num_vertices)], ["src", "dst"])

# Create a GraphFrame
g = graphframes.GraphFrame(v, e)

# Display the vertex and edge DataFrames
g.vertices.show()
g.edges.show()

# Get a DataFrame with columns "id" and "inDegree" (in-degree)
vertexInDegrees = g.inDegrees

# Find the youngest user's age in the graph.
# This queries the vertex DataFrame.
g.vertices.groupBy().min("age").show()

# Count the number of "follows" in the graph.
# This queries the edge DataFrame.
numFollows = g.edges.filter("relationship = 'follow'").count()

print("numFollows=", numFollows)






