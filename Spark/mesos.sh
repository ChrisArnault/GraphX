
export SPARK_MASTER=spark://134.158.75.222:7077

export MESOS_SANDBOX="/spark_dir/spark_tmp"
export MESOS_SANDBOX="/lsst/data/tmp/"
export MESOS_CONF="--conf spark.mesos.principal=lsst \
  --conf spark.mesos.secret=secret \
  --conf spark.mesos.role=lsst "

export SPARKCONF="--packages graphframes:graphframes:0.7.0-spark2.4-s_2.11 \
  $MESOS_CONF \
  --master $SPARK_MASTER \
  --conf spark.local.dir=/lsst/data/tmp/ \
  --conf spark.sql.crossJoin.enabled=true\
  --driver-memory 29g \
  --total-executor-cores $SPARK_CORES \
  --executor-cores 17 \
  --executor-memory 29g"

