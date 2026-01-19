ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT=$(dirname $ROOT)
CURDIR=$(pwd)
cd $ROOT

CONF_FILE="bench/hnswlib_run.conf"

VAR_LOG_OUTPUT_PATH=$ROOT/bench/out/logs/

VAR_DEF_K=1

VAR_M=16
VAR_EF_CONSTRUCTION=200
VAR_EF_SEARCH=10

VAR_NUM_QUERY_THREADS=40

VAR_WRITE_RATIO=0 #0% of queries are updates
VAR_DELETE_RATIO=0
# VAR_DELETE_RATIO=50  #50% of update queries are deletions
# todo: what about deletions?

# VAR_BATCH_SIZE=$(( 4 )) #num embeddings read from disk at once during build
# VAR_BUILT_SIZE=$(( 64 )) #num embedings to insert during build
# VAR_WARMUP_TIME_SEC=10 #only search
# VAR_RUN_TIME_SEC=10 #real test used for stat collection
VAR_BATCH_SIZE=$(( 4096 )) #num embeddings read from disk at once during build
VAR_BUILT_SIZE=$(( 1024 * 1024 * 4 )) #num embedings to insert during build
VAR_WARMUP_TIME_SEC=300 #only search
VAR_RUN_TIME_SEC=300 #real test used for stat collection

# VAR_RUNTIME_THROUGHPUT_REPORT_SEC=0 #use 0 to disable
VAR_RUNTIME_THROUGHPUT_REPORT_SEC=5 #use 0 to disable
VAR_SHOW_RUNTIME_REPORT_FOR_BUILD_AND_WARMUP=1 #1 to show throughput report during build and warmup phases

VAR_COLLECT_AVG_DISTANCES=1 #1 to collect and print average distances during searches

#create the config file if it does not exists and clean it if it does
echo > $CONF_FILE

echo "log-path:$VAR_LOG_OUTPUT_PATH" >> $CONF_FILE

echo "default-k:$VAR_DEF_K" >> $CONF_FILE

echo "hnsw-M:$VAR_M" >> $CONF_FILE
echo "hnsw-ef-construction:$VAR_EF_CONSTRUCTION" >> $CONF_FILE
echo "hnsw-ef-search:$VAR_EF_SEARCH" >> $CONF_FILE

echo "num-client-threads:$VAR_NUM_QUERY_THREADS" >> $CONF_FILE

echo "write-ratio:$VAR_WRITE_RATIO" >> $CONF_FILE
echo "delete-ratio:$VAR_DELETE_RATIO" >> $CONF_FILE

echo "bench-batch-size:$VAR_BATCH_SIZE" >> $CONF_FILE
echo "build-size:$VAR_BUILT_SIZE" >> $CONF_FILE
echo "warmup-time:$VAR_WARMUP_TIME_SEC" >> $CONF_FILE
echo "run-time:$VAR_RUN_TIME_SEC" >> $CONF_FILE

echo "throughput-report-time:$VAR_RUNTIME_THROUGHPUT_REPORT_SEC" >> $CONF_FILE
echo "show-runtime-report-for-build-and-warmup:$VAR_SHOW_RUNTIME_REPORT_FOR_BUILD_AND_WARMUP" >> $CONF_FILE

echo "collect-avg-distances:$VAR_COLLECT_AVG_DISTANCES" >> $CONF_FILE

cd $CURDIR