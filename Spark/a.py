import matplotlib.pyplot as plt
import matplotlib

import numpy as np

import sys
import os
import random
import time

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import SQLContext

from graphframes import GraphFrame
from graphframes.examples import Graphs

spark = SparkSession.builder.appName("GraphX").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

sqlContext = SQLContext(spark.sparkContext)

g = Graphs(sqlContext).friends()  # Get example graph

g.vertices.show()







