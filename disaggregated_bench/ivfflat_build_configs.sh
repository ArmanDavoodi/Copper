ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT=$(dirname $ROOT)
CURDIR=$(pwd)
cd $ROOT

CN_CONF_FILE="disaggregated_bench/compute_node/run.conf"
MN_CONF_FILE="disaggregated_bench/memory_node/run.conf"

VAR_LOG_OUTPUT_PATH=$ROOT/disaggregated_bench/out/logs/

VAR_DEF_K=1
VAR_N_PROBES=1
VAR_BUILT_SIZE=$(( 1024 * 32 )) #num embedings to insert during build
VAR_AVG_NUM_VEC_PER_CLUSTER=$((512))
VAR_NUM_CLUSTERS=$((VAR_BUILT_SIZE / VAR_AVG_NUM_VEC_PER_CLUSTER))
VAR_KMEANS_MAX_ITERS=10
VAR_INSERT_DUPLICATES=0 #1 to allow inserting duplicate vectors and 0 to not allow

VAR_NUM_QUERY_THREADS=4
VAR_BUILD_NUM_THREADS=160

VAR_PAGE_SIZE=$((512 * 256)) # should be power of 2
VAR_POOL_SIZE=$((1024 * 1024 * 128)) #128MB

VAR_WARMUP_TIME_SEC=60 #only search
VAR_RUN_TIME_SEC=60 #real test used for stat collection

VAR_RUNTIME_THROUGHPUT_REPORT_SEC=5 #use 0 to disable
VAR_SHOW_RUNTIME_REPORT_FOR_BUILD_AND_WARMUP=1 #1 to show throughput report during build and warmup phases

VAR_COLLECT_AVG_DISTANCES=1 #1 to collect and print average distances during searches

#create the config file if it does not exists and clean it if it does
echo > $CN_CONF_FILE

echo "log-path:$VAR_LOG_OUTPUT_PATH" >> $CN_CONF_FILE

echo "n-probes:$VAR_N_PROBES" >> $CN_CONF_FILE
echo "default-k:$VAR_DEF_K" >> $CN_CONF_FILE
echo "num-client-threads:$VAR_NUM_QUERY_THREADS" >> $CN_CONF_FILE
echo "page-size:$VAR_PAGE_SIZE" >> $CN_CONF_FILE
echo "pool-size:$VAR_POOL_SIZE" >> $CN_CONF_FILE

echo "warmup-time:$VAR_WARMUP_TIME_SEC" >> $CN_CONF_FILE
echo "run-time:$VAR_RUN_TIME_SEC" >> $CN_CONF_FILE

echo "throughput-report-time:$VAR_RUNTIME_THROUGHPUT_REPORT_SEC" >> $CN_CONF_FILE
echo "show-runtime-report-for-build-and-warmup:$VAR_SHOW_RUNTIME_REPORT_FOR_BUILD_AND_WARMUP" >> $CN_CONF_FILE

echo "collect-avg-distances:$VAR_COLLECT_AVG_DISTANCES" >> $CN_CONF_FILE

#create the config file if it does not exists and clean it if it does
echo > $MN_CONF_FILE

echo "log-path:$VAR_LOG_OUTPUT_PATH" >> $MN_CONF_FILE
echo "num-clusters:$VAR_NUM_CLUSTERS" >> $MN_CONF_FILE
echo "build-size:$VAR_BUILT_SIZE" >> $MN_CONF_FILE
echo "kmeans-max-iters:$VAR_KMEANS_MAX_ITERS" >> $MN_CONF_FILE
echo "insert-duplicates:$VAR_INSERT_DUPLICATES" >> $MN_CONF_FILE
echo "num-build-threads:$VAR_BUILD_NUM_THREADS" >> $MN_CONF_FILE

cd $CURDIR