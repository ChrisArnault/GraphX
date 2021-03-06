
export SPARK_MASTER=yarn

export SPARKCONF="--packages graphframes:graphframes:0.7.0-spark2.4-s_2.11 \
  --master $SPARK_MASTER \
  --conf spark.sql.crossJoin.enabled=true\
  --driver-memory 15G \
  --num-executors 4 \
  --executor-cores 1 \
  --executor-memory 1500M"

