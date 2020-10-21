make clean
make -f Makefile

# ./Exp 10000 uniform 1
# ./Exp 20000 uniform 1
# ./Exp 160000 uniform 1

./Exp -c 1000000 -d uniform -s 1
# ./Exp 1000000 skewed 4
# ./Exp 2000000 skewed 4
# ./Exp 4000000 skewed 4
# ./Exp 8000000 skewed 4
# ./Exp 16000000 skewed 4
# ./Exp 32000000 skewed 4
# ./Exp 64000000 skewed 4
# ./Exp 128000000 skewed 4

# ./Exp 16000000 normal 1
# ./Exp 16000000 uniform 1
# ./Exp 17468292 real 1
# ./Exp 100000000 OSM 1
