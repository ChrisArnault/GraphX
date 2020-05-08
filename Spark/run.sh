
source mesos4.sh

export N=1000
rm -f output/test2_N${N}_D${N}.log
spark-submit $SPARKCONF create_graphframe_2.py n=${N} d=${N} >output/test2_N${N}_D${N}.log

export N=10000
rm -f output/test2_N${N}_D${N}.log
spark-submit $SPARKCONF create_graphframe_2.py n=${N} d=${N} >output/test2_N${N}_D${N}.log

export N=100000
rm -f output/test2_N${N}_D${N}.log
spark-submit $SPARKCONF create_graphframe_2.py n=${N} d=${N} >output/test2_N${N}_D${N}.log

export N=1000000
rm -f output/test2_N${N}_D${N}.log
spark-submit $SPARKCONF create_graphframe_2.py n=${N} d=${N} >output/test2_N${N}_D${N}.log

export N=10000000
rm -f output/test2_N${N}_D${N}.log
# spark-submit $SPARKCONF create_graphframe_2.py n=${N} d=${N} >output/test2_N${N}_D${N}.log

export N=100000000
rm -f output/test2_N${N}_D${N}.log
# spark-submit $SPARKCONF create_graphframe_2.py n=${N} d=${N} >output/test2_N${N}_D${N}.log
