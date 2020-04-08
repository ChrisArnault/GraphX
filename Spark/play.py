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
    sqlContext = SQLContext(spark.sparkContext)

x = lambda : np.random.random()
y = lambda : np.random.random()
vertex_values = lambda : [(v, x(), y()) for v in range(100)]
df = sqlContext.createDataFrame(vertex_values, ["id", "x", "y"])
