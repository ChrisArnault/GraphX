
export SPARK_MASTER=spark://134.158.75.222:7077
export SPARK_MEMORY=28g

export SPARKCONF="--packages $SPARK_MONITORING_PACKAGES,graphframes:graphframes:0.7.0-spark2.4-s_2.11 \
  --jars $SPARK_MONITORING_JARS \
  $SPARK_MONITORING_CONF \
  --master $SPARK_MASTER \
  --conf spark.sql.crossJoin.enabled=true\
  --driver-memory $SPARK_MEMORY \
  --total-executor-cores $SPARK_CORES \
  --executor-cores 17 \
  --executor-memory $SPARK_MEMORY"

