ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
CURDIR=$(pwd)
cd $ROOT

# std::cerr << "Usage: " << argv[0] << " <input_file> <output_file> <num_threads> <num_points> "
#                  "<max_iter> <algorithm> (<num_clusters> | <num_clusters> <sample_size> | (<cluster_cap> | <leaf_cap> <internal_cap> <weighted_kmeans>) <clustering_alg> [<max_num_clusters> | <coefficent>])\n";
VAR_INPUT_FILE=$ROOT/bench/datasets/bigann/unique_identified/BIGANN1B.u8bin
VAR_OUTPUT_FILE_DIR=$ROOT/bench/datasets/bigann/index/ctypeu8/tree/sen/
VAR_OUTPUT_FILE_NAME_LIST=(
    bigann32M_8LC_8IC_MAX4096_MI24

    bigann32M_32LC_8IC_MAX4096_MI24
    bigann32M_32LC_16IC_MAX4096_MI24
    bigann32M_32LC_32IC_MAX4096_MI24

    bigann32M_64LC_8IC_MAX4096_MI24
    bigann32M_64LC_16IC_MAX4096_MI24
    bigann32M_64LC_32IC_MAX4096_MI24
    bigann32M_64LC_64IC_MAX4096_MI24
    bigann32M_64LC_128IC_MAX4096_MI24

    bigann32M_128LC_8IC_MAX4096_MI24
    bigann32M_128LC_16IC_MAX4096_MI24
    bigann32M_128LC_32IC_MAX4096_MI24
    bigann32M_128LC_64IC_MAX4096_MI24
    bigann32M_128LC_128IC_MAX4096_MI24
    bigann32M_128LC_256IC_MAX4096_MI24
    bigann32M_128LC_512IC_MAX4096_MI24

    bigann32M_256LC_8IC_MAX4096_MI24
    bigann32M_256LC_16IC_MAX4096_MI24
    bigann32M_256LC_32IC_MAX4096_MI24
    bigann32M_256LC_64IC_MAX4096_MI24
    bigann32M_256LC_128IC_MAX4096_MI24
    bigann32M_256LC_256IC_MAX4096_MI24
    bigann32M_256LC_512IC_MAX4096_MI24
    bigann32M_256LC_1024IC_MAX4096_MI24

    bigann32M_512LC_8IC_MAX4096_MI24
    bigann32M_512LC_16IC_MAX4096_MI24
    bigann32M_512LC_32IC_MAX4096_MI24
    bigann32M_512LC_64IC_MAX4096_MI24
    bigann32M_512LC_128IC_MAX4096_MI24
    bigann32M_512LC_256IC_MAX4096_MI24
    bigann32M_512LC_512IC_MAX4096_MI24
    bigann32M_512LC_1024IC_MAX4096_MI24
    bigann32M_512LC_2048IC_MAX4096_MI24

    bigann32M_1024LC_8IC_MAX4096_MI24
    bigann32M_1024LC_16IC_MAX4096_MI24
    bigann32M_1024LC_32IC_MAX4096_MI24
    bigann32M_1024LC_64IC_MAX4096_MI24
    bigann32M_1024LC_128IC_MAX4096_MI24
    bigann32M_1024LC_256IC_MAX4096_MI24
    bigann32M_1024LC_512IC_MAX4096_MI24
    bigann32M_1024LC_1024IC_MAX4096_MI24
    bigann32M_1024LC_2048IC_MAX4096_MI24

    bigann32M_2048LC_8IC_MAX4096_MI24
    bigann32M_2048LC_16IC_MAX4096_MI24
    bigann32M_2048LC_32IC_MAX4096_MI24
    bigann32M_2048LC_64IC_MAX4096_MI24
    bigann32M_2048LC_128IC_MAX4096_MI24
    bigann32M_2048LC_256IC_MAX4096_MI24
    bigann32M_2048LC_512IC_MAX4096_MI24
    bigann32M_2048LC_1024IC_MAX4096_MI24
    bigann32M_2048LC_2048IC_MAX4096_MI24
)

VAR_INTERNAL_CLUSTER_CAP_LISTS=(
    8
    8
    16
    32
    8
    16
    32
    64
    128
    8
    16
    32
    64
    128
    256
    512
    8
    16
    32
    64
    128
    256
    512
    1024
    8
    16
    32
    64
    128
    256
    512
    1024
    2048
    8
    16
    32
    64
    128
    256
    512
    1024
    2048
    8
    16
    32
    64
    128
    256
    512
    1024
    2048
)

VAR_LEAF_CLUSTER_CAP_LISTS=(
    8
    32
    32
    32
    64
    64
    64
    64
    64
    128
    128
    128
    128
    128
    128
    128
    256
    256
    256
    256
    256
    256
    256
    256
    512
    512
    512
    512
    512
    512
    512
    512
    512
    1024
    1024
    1024
    1024
    1024
    1024
    1024
    1024
    1024
    2048
    2048
    2048
    2048
    2048
    2048
    2048
    2048
    2048
)

VAR_NUM_COMMANDS=${#VAR_OUTPUT_FILE_NAME_LIST[@]}


for ((i=0; i<$VAR_NUM_COMMANDS; i++)); do
    VAR_OUTPUT_FILE_NAME=${VAR_OUTPUT_FILE_NAME_LIST[i]}
    VAR_LEAF_CLUSTER_CAP=${VAR_LEAF_CLUSTER_CAP_LISTS[i]}
    VAR_INTERNAL_CLUSTER_CAP=${VAR_INTERNAL_CLUSTER_CAP_LISTS[i]}
    echo ============================================================================================================================================================================================================================================================================================================ >> $ROOT/a2_clustering_tree_sen.log
    echo "building ivf-tree with leaf_cluster_cap=$VAR_LEAF_CLUSTER_CAP, internal_cluster_cap=$VAR_INTERNAL_CLUSTER_CAP, output_file=$VAR_OUTPUT_FILE_NAME"
    echo "building ivf-tree with leaf_cluster_cap=$VAR_LEAF_CLUSTER_CAP, internal_cluster_cap=$VAR_INTERNAL_CLUSTER_CAP, output_file=$VAR_OUTPUT_FILE_NAME" >> $ROOT/a2_clustering_tree_sen.log
    ./bench/datasets/bin/clustering $VAR_INPUT_FILE $VAR_OUTPUT_FILE_DIR/$VAR_OUTPUT_FILE_NAME 0 33554432 24 kmeans_hierarchical $VAR_LEAF_CLUSTER_CAP $VAR_INTERNAL_CLUSTER_CAP 0 max 4096 >> $ROOT/a2_clustering_tree_sen.log 2>&1
    echo "done building ivf-tree with leaf_cluster_cap=$VAR_LEAF_CLUSTER_CAP, internal_cluster_cap=$VAR_INTERNAL_CLUSTER_CAP, output_file=$VAR_OUTPUT_FILE_NAME"
    echo "done building ivf-tree with leaf_cluster_cap=$VAR_LEAF_CLUSTER_CAP, internal_cluster_cap=$VAR_INTERNAL_CLUSTER_CAP, output_file=$VAR_OUTPUT_FILE_NAME" >> $ROOT/a2_clustering_tree_sen.log
    echo ============================================================================================================================================================================================================================================================================================================ >> $ROOT/a2_clustering_tree_sen.log
done

cd $CURDIR