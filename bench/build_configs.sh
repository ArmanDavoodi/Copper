ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT=$(dirname $ROOT)
CURDIR=$(pwd)
cd $ROOT

CONF_FILE="bench/run.conf"

VAR_LOG_OUTPUT_PATH=$ROOT/bench/out/logs/

VAR_CLUSTERING="KMeans"
VAR_LEAF_SIZE=(128 1024)
VAR_INTERNAL_SIZE=(128 1024)
# VAR_LEAF_SIZE=(1 8)
# VAR_INTERNAL_SIZE=(1 4)
VAR_LEAF_SPLIT=2
VAR_INTERNAL_SPLIT=2

VAR_USE_BLOCK_BYTES=0
VAR_LEAF_BLOCK_BYTES=$(( 1024 * 16 ))
VAR_INTERNAL_BLOCK_BYTES=$(( 1024 * 16 ))
VAR_LEAF_BLOCK_SIZE=$((${VAR_LEAF_SIZE[1]}))
VAR_INTERNAL_BLOCK_SIZE=$((${VAR_INTERNAL_SIZE[1]}))

VAR_MEMORY_POOL_SIZE_GB=$(( 100 )) #size of memory pool for data nodes in GB

# VAR_DEF_LEAF_SEARCH_SPAN=8
# VAR_DEF_INTERNAL_SEARCH_SPAN=4
# VAR_DEF_K=2
VAR_DEF_LEAF_SEARCH_SPAN=10
VAR_DEF_INTERNAL_SEARCH_SPAN=10
VAR_DEF_K=10

VAR_MIGRATION_CHECK_TRIGGER_RATE=18
VAR_MIGRATION_CHECK_TRIGGER_SINGLE_RATE=8
VAR_RANDOM_BASE=80

# VAR_NUM_QUERY_THREADS=4
# VAR_NUM_SEARCHER_THREADS=8
# VAR_NUM_BG_MIGRATOR_THREADS=2
# VAR_NUM_BG_MERGER_THREADS=1
# VAR_NUM_BG_COMPACTOR_THREADS=2
VAR_NUM_QUERY_THREADS=40
VAR_NUM_SEARCHER_THREADS=80
VAR_NUM_BG_MIGRATOR_THREADS=18
VAR_NUM_BG_MERGER_THREADS=4
VAR_NUM_BG_COMPACTOR_THREADS=18
#todo: add numa ctl and bind threads

VAR_WRITE_RATIO=10 #10% of queries are updates
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

VAR_COLLECT_BUILD_STATS=1 #1 to collect and print stats after build phase
VAR_COLLECT_WARMUP_STATS=1 #1 to collect and print stats after warmup phase
VAR_COLLECT_RUN_STATS=1 #1 to collect and print stats after run phase

VAR_COLLECT_AVG_DISTANCES=1 #1 to collect and print average distances during searches

#create the config file if it does not exists and clean it if it does
echo > $CONF_FILE

echo "log-path:$VAR_LOG_OUTPUT_PATH" >> $CONF_FILE

echo "clustering:$VAR_CLUSTERING" >> $CONF_FILE
echo "leaf-size:[${VAR_LEAF_SIZE[0]},${VAR_LEAF_SIZE[1]}]" >> $CONF_FILE
echo "internal-size:[${VAR_INTERNAL_SIZE[0]},${VAR_INTERNAL_SIZE[1]}]" >> $CONF_FILE
echo "leaf-split:$VAR_LEAF_SPLIT" >> $CONF_FILE
echo "internal-split:$VAR_INTERNAL_SPLIT" >> $CONF_FILE
echo "use-block-bytes:$VAR_USE_BLOCK_BYTES" >> $CONF_FILE

if [ $VAR_USE_BLOCK_BYTES -eq 0 ]; then
    echo "leaf-block-size:$VAR_LEAF_BLOCK_SIZE" >> $CONF_FILE
    echo "internal-block-size:$VAR_INTERNAL_BLOCK_SIZE" >> $CONF_FILE
else
    echo "leaf-block-bytes:$VAR_LEAF_BLOCK_BYTES" >> $CONF_FILE
    echo "internal-block-bytes:$VAR_INTERNAL_BLOCK_BYTES" >> $CONF_FILE
fi
echo "memory-pool-size-gb:$VAR_MEMORY_POOL_SIZE_GB" >> $CONF_FILE

echo "leaf-block-bytes:$VAR_LEAF_BLOCK_BYTES" >> $CONF_FILE
echo "internal-block-bytes:$VAR_INTERNAL_BLOCK_BYTES" >> $CONF_FILE
echo "default-leaf-search-span:$VAR_DEF_LEAF_SEARCH_SPAN" >> $CONF_FILE
echo "default-internal-search-span:$VAR_DEF_INTERNAL_SEARCH_SPAN" >> $CONF_FILE
echo "default-k:$VAR_DEF_K" >> $CONF_FILE

echo "migration-check-trigger-rate:$VAR_MIGRATION_CHECK_TRIGGER_RATE" >> $CONF_FILE
echo "migration-check-trigger-single-rate:$VAR_MIGRATION_CHECK_TRIGGER_SINGLE_RATE" >> $CONF_FILE
echo "random-rate-base:$VAR_RANDOM_BASE" >> $CONF_FILE


echo "num-client-threads:$VAR_NUM_QUERY_THREADS" >> $CONF_FILE

echo "num-searcher-threads:$VAR_NUM_SEARCHER_THREADS" >> $CONF_FILE
echo "num-bg-migrator-threads:$VAR_NUM_BG_MIGRATOR_THREADS" >> $CONF_FILE
echo "num-bg-merger-threads:$VAR_NUM_BG_MERGER_THREADS" >> $CONF_FILE
echo "num-bg-compactor-threads:$VAR_NUM_BG_COMPACTOR_THREADS" >> $CONF_FILE


echo "write-ratio:$VAR_WRITE_RATIO" >> $CONF_FILE
echo "delete-ratio:$VAR_DELETE_RATIO" >> $CONF_FILE

echo "bench-batch-size:$VAR_BATCH_SIZE" >> $CONF_FILE
echo "build-size:$VAR_BUILT_SIZE" >> $CONF_FILE
echo "warmup-time:$VAR_WARMUP_TIME_SEC" >> $CONF_FILE
echo "run-time:$VAR_RUN_TIME_SEC" >> $CONF_FILE

echo "throughput-report-time:$VAR_RUNTIME_THROUGHPUT_REPORT_SEC" >> $CONF_FILE
echo "show-runtime-report-for-build-and-warmup:$VAR_SHOW_RUNTIME_REPORT_FOR_BUILD_AND_WARMUP" >> $CONF_FILE

echo "collect-build-stats:$VAR_COLLECT_BUILD_STATS" >> $CONF_FILE
echo "collect-warmup-stats:$VAR_COLLECT_WARMUP_STATS" >> $CONF_FILE
echo "collect-run-stats:$VAR_COLLECT_RUN_STATS" >> $CONF_FILE

echo "collect-avg-distances:$VAR_COLLECT_AVG_DISTANCES" >> $CONF_FILE

cd $CURDIR