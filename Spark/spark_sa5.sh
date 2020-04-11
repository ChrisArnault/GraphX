

export SPARKCONF="--packages $LAL_SPARK_PACKAGE,graphframes:graphframes:0.7.0-spark2.4-s_2.11 \
  --jars $LAL_SPARK_JARS \
  --master spark://134.158.75.222:7077 \
  --conf spark.sql.crossJoin.enabled=true\
  --driver-memory 29g \
  --total-executor-cores 85 \
  --executor-cores 17 \
  --executor-memory 29g"


