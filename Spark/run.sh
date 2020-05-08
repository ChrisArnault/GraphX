
source mesos4.sh
spark-submit $SPARKCONF create_graphframe_2.py n=1000 >output/test2_N1000_D1000.log
spark-submit $SPARKCONF create_graphframe_2.py n=10000 >output/test2_N10000_D10000.log
spark-submit $SPARKCONF create_graphframe_2.py n=100000 >output/test2_N100000_D100000.log
spark-submit $SPARKCONF create_graphframe_2.py n=1000000 >output/test2_N1000000_D1000000.log
spark-submit $SPARKCONF create_graphframe_2.py n=10000000 >output/test2_N10000000_D10000000.log

