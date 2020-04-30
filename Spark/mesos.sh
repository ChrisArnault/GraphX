
export SPARK_MASTER=mesos://134.158.75.63:5050

export MESOS_SANDBOX="/data/spark_local"
export MESOS_CONF="--conf spark.mesos.principal=lsst \
  --conf spark.mesos.secret=secret \
  --conf spark.mesos.role=lsst "

export SPARKCONF="--packages graphframes:graphframes:0.7.0-spark2.4-s_2.11 \
  $MESOS_CONF \
  --master $SPARK_MASTER \
  --conf spark.local.dir=/data/spark_local \
  --conf spark.sql.crossJoin.enabled=true\
  --driver-memory 29g \
  --total-executor-cores $SPARK_CORES \
  --executor-cores 17 \
  --executor-memory 29g"

