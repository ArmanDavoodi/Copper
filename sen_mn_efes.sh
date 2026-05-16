# r 0 bench/datasets/bigann/index/ctypeu8/tree/sen/bigann32M_16LCAP_512ICAP_MAX4096_MI24 /mnt/homes/adavoodi/Copper/disaggregated_bench/out/logs/

ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
CURDIR=$(pwd)
cd $ROOT

INDEX_PATH=$ROOT/bench/datasets/bigann/index/ctypeu8/tree/sen/
LOG_PATH=$ROOT/disaggregated_bench/out/logs/

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
> $ROOT/sen_mn_efes.log

for ((i=0; i<$NUM_COMMANDS; i++)); do
    INDEX_FILE=${INDEX_FILES[i]}
    NICKNAME_FILE=${NICKNAME[i]}
    LOG_FILE_PATH=$LOG_PATH/$NICKNAME_FILE/
    echo ============================================================================================================================================================================================================================================================================================================ >> $ROOT/sen_mn_efes.log
    echo "running search with index_file=$INDEX_FILE, nickname=$NICKNAME_FILE"
    echo "running search with index_file=$INDEX_FILE, nickname=$NICKNAME_FILE" >> $ROOT/sen_mn_efes.log
    ./disaggregated_bench/build/bin/memory/ivfflar_bench_mn 0 $INDEX_PATH/$INDEX_FILE $LOG_FILE_PATH >> $ROOT/sen_mn_efes.log 2>&1
    echo "done running search with index_file=$INDEX_FILE, nickname=$NICKNAME_FILE"
    echo "done running search with index_file=$INDEX_FILE, nickname=$NICKNAME_FILE" >> $ROOT/sen_mn_efes.log
    echo ============================================================================================================================================================================================================================================================================================================ >> $ROOT/sen_mn_efes.log
    sleep 120
done

cd $CURDIR