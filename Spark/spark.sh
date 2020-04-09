
export MESOS_SANDBOX="/spark_dir/spark_tmp"
export MESOS_SANDBOX="/lsst/data/tmp/"
export SPARKCONF="--packages $LAL_SPARK_PACKAGE,graphframes:graphframes:0.7.0-spark2.4-s_2.11 \
  --jars $LAL_SPARK_JARS \
  --conf spark.mesos.principal=lsst
  --conf spark.mesos.secret=secret \
  --conf spark.mesos.role=lsst \
  --conf spark.local.dir=/lsst/data/tmp/ \
  --conf spark.sql.crossJoin.enabled=true \
  --master mesos://vm-75063.lal.in2p3.fr:5050  \
  --driver-memory 29g \
  --total-executor-cores 85 \
  --executor-cores 17 \
  --executor-memory 29g"


