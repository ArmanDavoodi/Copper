ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT=$(dirname $ROOT)
CURDIR=$(pwd)
cd $ROOT

CN_CONF_FILE="disaggregated_bench/compute_node/run.conf"
MN_CONF_FILE="disaggregated_bench/memory_node/run.conf"

# CN_STAT_FILE=$ROOT/disaggregated_bench/out/stats/stats.stat

# VAR_LOG_OUTPUT_PATH=$ROOT/disaggregated_bench/out/logs/

# VAR_GT_PATH=$ROOT/bench/datasets/bigann/exact/dtypeu32/4M_K100
VAR_LATENCY_BASE=50
VAR_LATENCY_SAMPLE=1

VAR_DEF_K=10
VAR_NUM_QUERY_THREADS=64
# VAR_PAGE_SIZE=$((CONF_VAR_AVG_NUM_VEC_PER_CLUSTER * CONF_VAR_VECTOR_BYTES * 2)) # should be power of 2 -> will only be used for ivfflat.
# VAR_POOL_SIZE=$((1024 * 1024 * 1024)) #1GB

VAR_WARMUP_TIME_SEC=60 #only search
VAR_RUN_TIME_SEC=120 #real test used for stat collection

VAR_RUNTIME_THROUGHPUT_REPORT_SEC=5 #use 0 to disable
VAR_SHOW_RUNTIME_REPORT_FOR_BUILD_AND_WARMUP=1 #1 to show throughput report during build and warmup phases

VAR_COLLECT_AVG_DISTANCES=0 #1 to collect and print average distances during searches

#create the config file if it does not exists and clean it if it does
echo > $CN_CONF_FILE

# echo "log-path:$VAR_LOG_OUTPUT_PATH" >> $CN_CONF_FILE

# echo "ground-truth-path:$VAR_GT_PATH" >> $CN_CONF_FILE
echo "latency-sample-base:$VAR_LATENCY_BASE" >> $CN_CONF_FILE
echo "latency-sample-rate:$VAR_LATENCY_SAMPLE" >> $CN_CONF_FILE
echo "default-k:$VAR_DEF_K" >> $CN_CONF_FILE
echo "num-client-threads:$VAR_NUM_QUERY_THREADS" >> $CN_CONF_FILE

echo "warmup-time:$VAR_WARMUP_TIME_SEC" >> $CN_CONF_FILE
echo "run-time:$VAR_RUN_TIME_SEC" >> $CN_CONF_FILE

echo "throughput-report-time:$VAR_RUNTIME_THROUGHPUT_REPORT_SEC" >> $CN_CONF_FILE
echo "show-runtime-report-for-build-and-warmup:$VAR_SHOW_RUNTIME_REPORT_FOR_BUILD_AND_WARMUP" >> $CN_CONF_FILE

echo "collect-avg-distances:$VAR_COLLECT_AVG_DISTANCES" >> $CN_CONF_FILE
# echo "stat-file:$CN_STAT_FILE" >> $CN_CONF_FILE

#create the config file if it does not exists and clean it if it does
echo > $MN_CONF_FILE

# echo "log-path:$VAR_LOG_OUTPUT_PATH" >> $MN_CONF_FILE

cd $CURDIR