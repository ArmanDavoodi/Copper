ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
CURDIR=$(pwd)
cd $ROOT

# std::cerr << "Usage: " << argv[0] << " <input_file> <output_file> <num_threads> <num_points> "
#                  "<max_iter> <algorithm> (<num_clusters> | <num_clusters> <sample_size> | (<cluster_cap> | <leaf_cap> <internal_cap> <weighted_kmeans>) <clustering_alg> [<max_num_clusters> | <coefficent>])\n";
VAR_INPUT_FILE=$ROOT/bench/datasets/bigann/unique_identified/BIGANN1B.u8bin
VAR_OUTPUT_FILE_DIR=$ROOT/bench/datasets/bigann/index/ctypeu8/capped
VAR_OUTPUT_FILE_NAME_LIST=(
    # bigann1B_1024C_MAX512_MI24
    # bigann1B_2048C_MAX512_MI24
    bigann1B_4096C_MAX512_MI24
    bigann1B_8192C_MAX512_MI24
    bigann1B_16384C_MAX512_MI24
    # bigann1B_32768C_MAX512_MI24
    # bigann1B_65536C_MAX512_MI24
)

VAR_CLUSTER_CAP_LISTS=(
    # 1024
    # 2048
    4096
    8192
    16384
    # 32768
    # 65536
)

VAR_NUM_COMMANDS=${#VAR_OUTPUT_FILE_NAME_LIST[@]}


for ((i=0; i<$VAR_NUM_COMMANDS; i++)); do
    VAR_OUTPUT_FILE_NAME=${VAR_OUTPUT_FILE_NAME_LIST[i]}
    VAR_CLUSTER_CAP=${VAR_CLUSTER_CAP_LISTS[i]}
    echo ============================================================================================================================================================================================================================================================================================================ >> $ROOT/a2_clustering_capped.log
    echo "building ivf-capped with cluster_cap=$VAR_CLUSTER_CAP, output_file=$VAR_OUTPUT_FILE_NAME"
    echo "building ivf-capped with cluster_cap=$VAR_CLUSTER_CAP, output_file=$VAR_OUTPUT_FILE_NAME" >> $ROOT/a2_clustering_capped.log
    ./bench/datasets/bin/clustering $VAR_INPUT_FILE $VAR_OUTPUT_FILE_DIR/$VAR_OUTPUT_FILE_NAME 100 0 24 kmeans_capped $VAR_CLUSTER_CAP max 512 >> $ROOT/a2_clustering_capped.log 2>&1
    echo "done building ivf-capped with cluster_cap=$VAR_CLUSTER_CAP, output_file=$VAR_OUTPUT_FILE_NAME"
    echo "done building ivf-capped with cluster_cap=$VAR_CLUSTER_CAP, output_file=$VAR_OUTPUT_FILE_NAME" >> $ROOT/a2_clustering_capped.log
    echo ============================================================================================================================================================================================================================================================================================================ >> $ROOT/a2_clustering_capped.log
done

cd $CURDIR

# ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# CURDIR=$(pwd)
# cd $ROOT

# # std::cerr << "Usage: " << argv[0] << " <input_file> <output_file> <num_threads> <num_points> "
# #                  "<max_iter> <algorithm> (<num_clusters> | <num_clusters> <sample_size> | (<cluster_cap> | <leaf_cap> <internal_cap> <weighted_kmeans>) <clustering_alg> [<max_num_clusters> | <coefficent>])\n";
# VAR_INPUT_FILE=$ROOT/bench/datasets/bigann/unique_identified/BIGANN1B.u8bin
# VAR_OUTPUT_FILE_DIR=$ROOT/bench/datasets/bigann/index/ctypeu8/tree/sen/
# VAR_OUTPUT_FILE_NAME_LIST=(
#     # bigann128M_2048LCAP_8ICAP_MAX512_MI24
#     # bigann128M_2048LCAP_128ICAP_MAX512_MI24
#     # bigann128M_2048LCAP_512ICAP_MAX512_MI24

#     # bigann128M_4096LCAP_8ICAP_MAX512_MI24
#     bigann128M_4096LCAP_128ICAP_MAX512_MI24
#     bigann128M_4096LCAP_512ICAP_MAX512_MI24

#     bigann128M_8192LCAP_8ICAP_MAX512_MI24
#     bigann128M_8192LCAP_32ICAP_MAX512_MI24
#     # bigann128M_8192LCAP_128ICAP_MAX512_MI24

#     # bigann128M_16384LCAP_8ICAP_MAX512_MI24
#     # bigann128M_16384LCAP_32ICAP_MAX512_MI24
#     # bigann128M_16384LCAP_64ICAP_MAX512_MI24
# )

# VAR_INTERNAL_CLUSTER_CAP_LISTS=(
#     # 8
#     # 128
#     # 512
#     # 8
#     128
#     512
#     8
#     32
#     # 128
#     # 8
#     # 32
#     # 64
# )

# VAR_LEAF_CLUSTER_CAP_LISTS=(
#     # 2048
#     # 2048
#     # 2048
#     # 4096
#     4096
#     4096
#     8192
#     8192
#     # 8192
#     # 16384
#     # 16384
#     # 16384
# )

# VAR_NUM_COMMANDS=${#VAR_OUTPUT_FILE_NAME_LIST[@]}


# for ((i=0; i<$VAR_NUM_COMMANDS; i++)); do
#     VAR_OUTPUT_FILE_NAME=${VAR_OUTPUT_FILE_NAME_LIST[i]}
#     VAR_LEAF_CLUSTER_CAP=${VAR_LEAF_CLUSTER_CAP_LISTS[i]}
#     VAR_INTERNAL_CLUSTER_CAP=${VAR_INTERNAL_CLUSTER_CAP_LISTS[i]}
#     echo ============================================================================================================================================================================================================================================================================================================ >> $ROOT/a1128_clustering_tree_sen.log
#     echo "building ivf-tree with leaf_cluster_cap=$VAR_LEAF_CLUSTER_CAP, internal_cluster_cap=$VAR_INTERNAL_CLUSTER_CAP, output_file=$VAR_OUTPUT_FILE_NAME"
#     echo "building ivf-tree with leaf_cluster_cap=$VAR_LEAF_CLUSTER_CAP, internal_cluster_cap=$VAR_INTERNAL_CLUSTER_CAP, output_file=$VAR_OUTPUT_FILE_NAME" >> $ROOT/a1128_clustering_tree_sen.log
#     ./bench/datasets/bin/clustering $VAR_INPUT_FILE $VAR_OUTPUT_FILE_DIR/$VAR_OUTPUT_FILE_NAME 120 134217728 24 kmeans_hierarchical $VAR_LEAF_CLUSTER_CAP $VAR_INTERNAL_CLUSTER_CAP 0 max 512 >> $ROOT/a1128_clustering_tree_sen.log 2>&1
#     echo "done building ivf-tree with leaf_cluster_cap=$VAR_LEAF_CLUSTER_CAP, internal_cluster_cap=$VAR_INTERNAL_CLUSTER_CAP, output_file=$VAR_OUTPUT_FILE_NAME"
#     echo "done building ivf-tree with leaf_cluster_cap=$VAR_LEAF_CLUSTER_CAP, internal_cluster_cap=$VAR_INTERNAL_CLUSTER_CAP, output_file=$VAR_OUTPUT_FILE_NAME" >> $ROOT/a1128_clustering_tree_sen.log
#     echo ============================================================================================================================================================================================================================================================================================================ >> $ROOT/a1128_clustering_tree_sen.log
# done

# cd $CURDIR