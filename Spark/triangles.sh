
source sa4.sh

spark-submit $SPARKCONF use_graphframes.py f=test_N1000_BN10_BE1_D1000_G10000
spark-submit $SPARKCONF use_graphframes.py f=test_N10000_BN10_BE1_D10000_G10000
spark-submit $SPARKCONF use_graphframes.py f=test_N100000_BN10_BE1_D100000_G10000
spark-submit $SPARKCONF use_graphframes.py f=test_N1000000_BN10_BE1_D1000000_G10000
spark-submit $SPARKCONF use_graphframes.py f=test_N10000000_BN10_BE1_D10000000_G10000
spark-submit $SPARKCONF use_graphframes.py f=test_N100000000_BN50_BE1_D100000000_G10000
