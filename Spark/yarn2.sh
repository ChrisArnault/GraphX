
# 0.8.0-spark2.4-s_2.11

export SPARKCONF="--packages $LAL_SPARK_PACKAGE,graphframes:graphframes:0.7.0-spark2.4-s_2.11 \
  --jars $LAL_SPARK_JARS \
  --master yarn \
  --driver-memory 29g \
  --total-executor-cores 34 \
  --executor-cores 17 \
  --executor-memory 29g"


