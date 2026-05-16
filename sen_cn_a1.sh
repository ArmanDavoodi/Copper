# r 0 bench/datasets/bigann/index/ctypeu8/tree/sen/bigann32M_16LCAP_512ICAP_MAX4096_MI24 /mnt/homes/adavoodi/Copper/disaggregated_bench/out/logs/

ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
CURDIR=$(pwd)
cd $ROOT

INDEX_PATH=$ROOT/bench/datasets/bigann/index/ctypeu8/tree/sen/
LOG_PATH=$ROOT/disaggregated_bench/out/logs/
STAT_PATH=$ROOT/disaggregated_bench/out/stats/

INDEX_FILES=(
    bigann128M_2048LCAP_8ICAP_MAX512_MI24
    bigann128M_2048LCAP_128ICAP_MAX512_MI24
    bigann128M_2048LCAP_512ICAP_MAX512_MI24

    bigann128M_4096LCAP_8ICAP_MAX512_MI24
    bigann128M_4096LCAP_128ICAP_MAX512_MI24
    bigann128M_4096LCAP_512ICAP_MAX512_MI24

    bigann128M_8192LCAP_8ICAP_MAX512_MI24
    bigann128M_8192LCAP_32ICAP_MAX512_MI24
    bigann128M_8192LCAP_128ICAP_MAX512_MI24

    bigann128M_16384LCAP_8ICAP_MAX512_MI24
    bigann128M_16384LCAP_32ICAP_MAX512_MI24
    bigann128M_16384LCAP_64ICAP_MAX512_MI24
)

NICKNAME=(
    "2048L_8I"
    "2048L_128I"
    "2048L_512I"

    "4096L_8I"
    "4096L_128I"
    "4096L_512I"

    "8192L_8I"
    "8192L_32I"
    "8192L_128I"

    "16384L_8I"
    "16384L_32I"
    "16384L_64I"
)

# INDEX_FILES=(
#     bigann32M_8LCAP_8ICAP_MAX4096_MI24

#     bigann32M_16LCAP_512ICAP_MAX4096_MI24

#     bigann32M_128LCAP_8ICAP_MAX4096_MI24
#     bigann32M_128LCAP_32ICAP_MAX4096_MI24

#     bigann32M_256LCAP_8ICAP_MAX4096_MI24

#     bigann32M_512LCAP_8ICAP_MAX4096_MI24

#     bigann32M_1024LCAP_8ICAP_MAX4096_MI24
#     bigann32M_1024LCAP_128ICAP_MAX4096_MI24
#     bigann32M_1024LCAP_256ICAP_MAX4096_MI24
#     bigann32M_1024LCAP_512ICAP_MAX4096_MI24
#     bigann32M_1024LCAP_1024ICAP_MAX4096_MI24

#     bigann32M_2048LCAP_8ICAP_MAX4096_MI24
#     bigann32M_2048LCAP_32ICAP_MAX4096_MI24
#     bigann32M_2048LCAP_64ICAP_MAX4096_MI24
#     bigann32M_2048LCAP_128ICAP_MAX4096_MI24
#     bigann32M_2048LCAP_256ICAP_MAX4096_MI24
#     bigann32M_2048LCAP_512ICAP_MAX4096_MI24
#     bigann32M_2048LCAP_1024ICAP_MAX4096_MI24

#     bigann32M_4096LCAP_8ICAP_MAX4096_MI24
#     bigann32M_4096LCAP_16ICAP_MAX4096_MI24
#     bigann32M_4096LCAP_32ICAP_MAX4096_MI24
#     bigann32M_4096LCAP_128ICAP_MAX4096_MI24
#     bigann32M_4096LCAP_512ICAP_MAX4096_MI24
#     bigann32M_4096LCAP_1024ICAP_MAX4096_MI24
#     bigann32M_4096LCAP_2048ICAP_MAX4096_MI24

#     bigann32M_8192LCAP_8ICAP_MAX4096_MI24
# )

# NICKNAME=(
#     8L_8I

#     16L_512I

#     128L_8I
#     128L_32I

#     256L_8I

#     512L_8I

#     1024L_8I
#     1024L_128I
#     1024L_256I
#     1024L_512I
#     1024L_1024I

#     2048L_8I
#     2048L_32I
#     2048L_64I
#     2048L_128I
#     2048L_256I
#     2048L_512I
#     2048L_1024I

#     4096L_8I
#     4096L_16I
#     4096L_32I
#     4096L_128I
#     4096L_512I
#     4096L_1024I
#     4096L_2048I

#     8192L_8I
# )

NUM_COMMANDS=${#INDEX_FILES[@]}

# r 0 bench/datasets/bigann/index/ctypeu8/tree/sen/bigann32M_16LCAP_512ICAP_MAX4096_MI24 bench/datasets/bigann/exact/dtypeu32/32M_K100 disaggregated_bench/out/logs/ disaggregated_bench/out/stats/cn016_512.stat 1024 1073741824 4 128
> $ROOT/sen_cn_a1.log

for ((i=0; i<$NUM_COMMANDS; i++)); do
    sleep 480
    INDEX_FILE=${INDEX_FILES[i]}
    NICKNAME_FILE=${NICKNAME[i]}
    LOG_FILE_PATH=$LOG_PATH/$NICKNAME_FILE/
    STAT_FILE_PATH=$STAT_PATH/$NICKNAME_FILE.stat
    echo ============================================================================================================================================================================================================================================================================================================ >> $ROOT/sen_cn_a1.log
    echo "running search with index_file=$INDEX_FILE, nickname=$NICKNAME_FILE"
    echo "running search with index_file=$INDEX_FILE, nickname=$NICKNAME_FILE" >> $ROOT/sen_cn_a1.log
    #RECALL
    ./disaggregated_bench/build/bin/compute/ivfflar_bench_cn 0 $INDEX_PATH/$INDEX_FILE bench/datasets/bigann/exact/dtypeu32/100M_K100 $LOG_FILE_PATH $STAT_FILE_PATH 1024 2147483648 8 16 >> $ROOT/sen_cn_a1.log 2>&1
    #QPS
    # ./disaggregated_bench/build/bin/compute/ivfflar_bench_cn 0 $INDEX_PATH/$INDEX_FILE $LOG_FILE_PATH $STAT_FILE_PATH 1024 1073741824 32 16 >> $ROOT/sen_cn_a1.log 2>&1
    echo "done running search with index_file=$INDEX_FILE, nickname=$NICKNAME_FILE"
    echo "done running search with index_file=$INDEX_FILE, nickname=$NICKNAME_FILE" >> $ROOT/sen_cn_a1.log
    echo ============================================================================================================================================================================================================================================================================================================ >> $ROOT/sen_cn_a1.log
    sleep 120
done



cd $CURDIR