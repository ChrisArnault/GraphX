
export MESOS_SANDBOX="/spark_dir/spark_tmp"
export LOCAL_DIRS="/spark_dir/spark_tmp"
export SPARKCONF="--packages io.delta:delta-core_2.11:0.4.0,org.influxdb:influxdb-java:2.14,graphframes:graphframes:0.7.0-spark2.4-s_2.11 \
  --master yarn \
  --driver-memory 29g \
  --total-executor-cores 85 \
  --executor-cores 17 \
  --executor-memory 29g"


