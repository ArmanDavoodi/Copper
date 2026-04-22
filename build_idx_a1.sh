ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
CURDIR=$(pwd)
cd $ROOT

# std::cerr << "Usage: " << argv[0] << " <input_file> <output_file> <num_threads> <num_points> "
#                  "<max_iter> <algorithm> (<num_clusters> | <num_clusters> <sample_size> | (<cluster_cap> | <leaf_cap> <internal_cap> <weighted_kmeans>) <clustering_alg> [<max_num_clusters> | <coefficent>])\n";
VAR_INPUT_FILE=$ROOT/bench/datasets/bigann/unique_identified/BIGANN1B.u8bin
VAR_OUTPUT_FILE_DIR=$ROOT/bench/datasets/bigann/index/ctypeu8/capped
VAR_OUTPUT_FILE_NAME_LIST=(
    bigann1B_1024C_MAX8192_MI24
    bigann1B_2048C_MAX8192_MI24
    bigann1B_4096C_MAX8192_MI24
    bigann1B_8192C_MAX8192_MI24
    bigann1B_16384C_MAX8192_MI24
    bigann1B_32768C_MAX8192_MI24
    bigann1B_65536C_MAX8192_MI24
)

VAR_CLUSTER_CAP_LISTS=(
    1024
    2048
    4096
    8192
    16384
    32768
    65536
)

VAR_NUM_COMMANDS=${#VAR_OUTPUT_FILE_NAME_LIST[@]}


for ((i=0; i<$VAR_NUM_COMMANDS; i++)); do
    VAR_OUTPUT_FILE_NAME=${VAR_OUTPUT_FILE_NAME_LIST[i]}
    VAR_CLUSTER_CAP=${VAR_CLUSTER_CAP_LISTS[i]}
    echo ============================================================================================================================================================================================================================================================================================================ >> $ROOT/a1_clustering_capped.log
    echo "building ivf-capped with cluster_cap=$VAR_CLUSTER_CAP, output_file=$VAR_OUTPUT_FILE_NAME"
    echo "building ivf-capped with cluster_cap=$VAR_CLUSTER_CAP, output_file=$VAR_OUTPUT_FILE_NAME" >> $ROOT/a1_clustering_capped.log
    ./bench/datasets/bin/clustering $VAR_INPUT_FILE $VAR_OUTPUT_FILE_DIR/$VAR_OUTPUT_FILE_NAME 0 0 24 kmeans_capped $VAR_CLUSTER_CAP max 8192 >> $ROOT/a1_clustering_capped.log 2>&1
    echo "done building ivf-capped with cluster_cap=$VAR_CLUSTER_CAP, output_file=$VAR_OUTPUT_FILE_NAME"
    echo "done building ivf-capped with cluster_cap=$VAR_CLUSTER_CAP, output_file=$VAR_OUTPUT_FILE_NAME" >> $ROOT/a1_clustering_capped.log
    echo ============================================================================================================================================================================================================================================================================================================ >> $ROOT/a1_clustering_capped.log
done

cd $CURDIR