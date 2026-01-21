ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT=$(dirname $ROOT)
CURDIR=$(pwd)
cd $ROOT

CONF_FILE="bench/ivfflat_run.conf"

VAR_LOG_OUTPUT_PATH=$ROOT/bench/out/logs/

VAR_BUILT_SIZE=$(( 1024 * 1024 * 4 )) #num embedings to insert during build
VAR_NUM_CLUSTERS=$((VAR_BUILT_SIZE / 1024))

VAR_DEF_K=1
VAR_N_PROBES=64
VAR_KMEANS_MAX_ITERS=32
VAR_INSERT_DUPLICATES=0 #1 to allow inserting duplicate vectors and 0 to not allow

VAR_NUM_QUERY_THREADS=60

VAR_WARMUP_TIME_SEC=10 #only search
VAR_RUN_TIME_SEC=30 #real test used for stat collection

VAR_RUNTIME_THROUGHPUT_REPORT_SEC=1 #use 0 to disable
VAR_SHOW_RUNTIME_REPORT_FOR_BUILD_AND_WARMUP=1 #1 to show throughput report during build and warmup phases


VAR_COLLECT_AVG_DISTANCES=1 #1 to collect and print average distances during searches

#create the config file if it does not exists and clean it if it does
echo > $CONF_FILE

echo "log-path:$VAR_LOG_OUTPUT_PATH" >> $CONF_FILE

echo "num-clusters:$VAR_NUM_CLUSTERS" >> $CONF_FILE
echo "n-probes:$VAR_N_PROBES" >> $CONF_FILE
echo "default-k:$VAR_DEF_K" >> $CONF_FILE
echo "kmeans-max-iters:$VAR_KMEANS_MAX_ITERS" >> $CONF_FILE
echo "insert-duplicates:$VAR_INSERT_DUPLICATES" >> $CONF_FILE

echo "num-client-threads:$VAR_NUM_QUERY_THREADS" >> $CONF_FILE

echo "build-size:$VAR_BUILT_SIZE" >> $CONF_FILE
echo "warmup-time:$VAR_WARMUP_TIME_SEC" >> $CONF_FILE
echo "run-time:$VAR_RUN_TIME_SEC" >> $CONF_FILE

echo "throughput-report-time:$VAR_RUNTIME_THROUGHPUT_REPORT_SEC" >> $CONF_FILE
echo "show-runtime-report-for-build-and-warmup:$VAR_SHOW_RUNTIME_REPORT_FOR_BUILD_AND_WARMUP" >> $CONF_FILE

echo "collect-avg-distances:$VAR_COLLECT_AVG_DISTANCES" >> $CONF_FILE

cd $CURDIR