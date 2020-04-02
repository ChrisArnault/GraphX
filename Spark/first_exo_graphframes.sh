
time spark-submit \
  --conf spark.mesos.principal=lsst \
  --conf spark.mesos.secret=secret \
  --conf spark.mesos.role=lsst \
  --packages io.delta:delta-core_2.11:0.4.0,org.influxdb:influxdb-java:2.14 \
  --master mesos://vm-75063.lal.in2p3.fr:5050  \
  --driver-memory 29g \
  --total-executor-cores 85 \
  --executor-cores 17 \
  --executor-memory 29g \
  first_exo_graphframes.py $*


