from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import SQLContext

from graphframes.examples import Graphs

spark = SparkSession.builder.appName("GraphX").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

sqlContext = SQLContext(spark.sparkContext)

g = Graphs(sqlContext).friends()

g.vertices.show()
g.edges.show()







