ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
CURDIR=$(pwd)
cd $ROOT

# std::cerr << "Usage: " << argv[0] << " <input_file> <output_file> <num_threads> <num_points> "
#                  "<max_iter> <algorithm> (<num_clusters> | <num_clusters> <sample_size> | (<cluster_cap> | <leaf_cap> <internal_cap> <weighted_kmeans>) <clustering_alg> [<max_num_clusters> | <coefficent>])\n";
VAR_INPUT_FILE=$ROOT/bench/datasets/bigann/unique_identified/BIGANN1B.u8bin
VAR_OUTPUT_FILE_DIR=$ROOT/bench/datasets/bigann/index/ctypeu8/flat
VAR_OUTPUT_FILE_NAME_LIST=(
    bigann1B_sample32M_16384C_MI24
    bigann1B_sample32M_32768C_MI24
    bigann1B_sample32M_65536C_MI24
)

VAR_NUM_CLUSTER_LISTS=(
    16384
    32768
    65536
)

VAR_NUM_COMMANDS=${#VAR_OUTPUT_FILE_NAME_LIST[@]}


for ((i=0; i<$VAR_NUM_COMMANDS; i++)); do
    VAR_OUTPUT_FILE_NAME=${VAR_OUTPUT_FILE_NAME_LIST[i]}
    VAR_NUM_CLUSTER=${VAR_NUM_CLUSTER_LISTS[i]}
    echo ============================================================================================================================================================================================================================================================================================================ >> $ROOT/efes_clustering_flat.log
    echo "building ivf-flat with num_clusters=$VAR_NUM_CLUSTER, output_file=$VAR_OUTPUT_FILE_NAME"
    echo "building ivf-flat with num_clusters=$VAR_NUM_CLUSTER, output_file=$VAR_OUTPUT_FILE_NAME" >> $ROOT/efes_clustering_flat.log
    ./bench/datasets/bin/clustering $VAR_INPUT_FILE $VAR_OUTPUT_FILE_DIR/$VAR_OUTPUT_FILE_NAME 0 0 24 kmeans_sampled $VAR_NUM_CLUSTER 33554432 >> $ROOT/efes_clustering_flat.log 2>&1
    echo "done building ivf-flat with num_clusters=$VAR_NUM_CLUSTER, output_file=$VAR_OUTPUT_FILE_NAME"
    echo "done building ivf-flat with num_clusters=$VAR_NUM_CLUSTER, output_file=$VAR_OUTPUT_FILE_NAME" >> $ROOT/efes_clustering_flat.log
    echo ============================================================================================================================================================================================================================================================================================================ >> $ROOT/efes_clustering_flat.log
done

cd $CURDIR