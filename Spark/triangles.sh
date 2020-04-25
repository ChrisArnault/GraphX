
source sa4.sh

# spark-submit $SPARKCONF use_graphframes.py f=test_N1000_BN1_BE1_D100_G10000
# spark-submit $SPARKCONF use_graphframes.py f=test_N10000_BN10_BE10_D1000_G10000
# spark-submit $SPARKCONF use_graphframes.py f=test_N100000_BN10_BE200_D1000_G10000
# spark-submit $SPARKCONF use_graphframes.py f=test_N1000000_BN10_BE100_D1000_G10000
# spark-submit $SPARKCONF use_graphframes.py f=test_N1000000_BN10_BE200_D1000_G10000
spark-submit $SPARKCONF use_graphframes.py f=test_N1000000_BN10_BE500_D10000_G10000

