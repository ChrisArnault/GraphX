
export SPARK_MASTER=yarn
export SPARK_CORES=34

export SPARKCONF="--packages $SPARK_MONITORING_PACKAGES,graphframes:graphframes:0.7.0-spark2.4-s_2.11 \
  --jars $SPARK_MONITORING_JARS \
  $SPARK_MONITORING_CONF \
  --master $SPARK_MASTER \
  --conf spark.sql.crossJoin.enabled=true\
  --driver-memory 29g \
  --total-executor-cores $SPARK_CORES \
  --executor-cores 17 \
  --executor-memory 29g"

