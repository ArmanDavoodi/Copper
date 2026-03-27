ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
CURDIR=$(pwd)
cd $ROOT

SRCDIR=$ROOT/bench/datasets
OUTDIR=$ROOT/bench/datasets/bin
rm -r $OUTDIR/
mkdir -p $OUTDIR/

INC_PATHS="-I$ROOT/ -I$ROOT/include/"
COMM_FLAGS="-std=c++20 -g3 -fno-omit-frame-pointer -rdynamic -DBUILD=RELEASE -DENABLE_TEST_LOGGING"

g++ $COMM_FLAGS $* $SRCDIR/make_dataset.cpp -o $OUTDIR/make_dataset
g++ $COMM_FLAGS -fopenmp $INC_PATHS $* $SRCDIR/find_duplicates.cpp -o $OUTDIR/find_duplicates
g++ $COMM_FLAGS -fopenmp $INC_PATHS $* $SRCDIR/clustering.cpp -o $OUTDIR/clustering
g++ $COMM_FLAGS -fopenmp $INC_PATHS $* $SRCDIR/brute_force.cpp -o $OUTDIR/brute_force

cd $CURDIR

# Needs DBUILD, DVECTOR_TYPE, DCENTROID_TYPE, DDISTANCE_TYPE, DDIMENSION to be set.