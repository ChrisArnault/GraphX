from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import SQLContext

from graphframes.examples import Graphs

import os
import time


num_vertices = 1000
num_edges = num_vertices
degree_max =  100
distance_max = 0.1
home = '/user/chris.arnault/graphs/testbatch'
file = "mydf"

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

def get_file_size(file):
    import subprocess
    cmd = "hdfs dfs -du -h {}} | egrep {}/{}".format(home, file)
    result = subprocess.check_output(cmd, shell=True).decode().split("\n")
    for line in result:
        if conf.dest in line:
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


spark = SparkSession.builder.appName("GraphX").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
sqlContext = SQLContext(spark.sparkContext)

os.system("hdfs dfs -rm -r -f {}/{}".format(home, file))

"""
    for batch in range(conf.loops):
        print("batch #{}".format(batch))

        s = Stepper()
        values = [(ra_value(), dec_value(), z_value()) for i in range(batch_size)]
        df = spark.createDataFrame(values, ['ra', 'dec', 'z'])
        df = df.repartition(conf.partitions, "ra")
        df = df.cache()
        df.count()
        s.show_step("building the dataframe")

        s = Stepper()
        if batch == 0:
            df.write.format(conf.file_format).save(conf.dest)
        else:
            df.write.format(conf.file_format).mode("append").save(conf.dest)
        s.show_step("Write block")

        new_size = get_file_size(conf)
        increment = new_size - previous_size
        previous_size = new_size

        print("file_size={} increment={}".format(new_size, increment))
"""

s = Stepper()

x = lambda : np.random.random()
y = lambda : np.random.random()
loops = 100
previous_size = 0

for batch in range(loops):
    values = [(v, x(), y()) for v_id in range(num_vertices)]
    df = sqlContext.createDataFrame(values, ["id", "x", "y"])
    df = df.cache()
    df.count()
    s.show_step("building the dataframe")

    if batch == 0:
        df.write.format("parquet").save(home)
    else:
        df.write.format("parquet").mode("append").save(home)
    s.show_step("Write block")

    new_size = get_file_size(file)
    increment = new_size - previous_size
    previous_size = new_size

    print("file_size={} increment={}".format(new_size, increment))

