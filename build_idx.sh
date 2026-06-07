# ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# CURDIR=$(pwd)
# cd $ROOT

# # std::cerr << "Usage: " << argv[0] << " <input_file> <output_file> <num_threads> <num_points> "
# #                  "<max_iter> <algorithm> (<num_clusters> | <num_clusters> <sample_size> | (<cluster_cap> | <leaf_cap> <internal_cap> <weighted_kmeans>) <clustering_alg> [<max_num_clusters> | <coefficent>])\n";
# VAR_INPUT_FILE=$ROOT/bench/datasets/bigann/unique_identified/BIGANN1B.u8bin
# VAR_OUTPUT_FILE_DIR=$ROOT/bench/datasets/bigann/index/ctypeu8/flat
# VAR_OUTPUT_FILE_NAME_LIST=(
#     bigann1B_sample32M_16384C_MI24
#     bigann1B_sample32M_32768C_MI24
#     bigann1B_sample32M_65536C_MI24
# )

# VAR_NUM_CLUSTER_LISTS=(
#     16384
#     32768
#     65536
# )

# VAR_NUM_COMMANDS=${#VAR_OUTPUT_FILE_NAME_LIST[@]}


# for ((i=0; i<$VAR_NUM_COMMANDS; i++)); do
#     VAR_OUTPUT_FILE_NAME=${VAR_OUTPUT_FILE_NAME_LIST[i]}
#     VAR_NUM_CLUSTER=${VAR_NUM_CLUSTER_LISTS[i]}
#     echo ============================================================================================================================================================================================================================================================================================================ >> $ROOT/efes_clustering_flat.log
#     echo "building ivf-flat with num_clusters=$VAR_NUM_CLUSTER, output_file=$VAR_OUTPUT_FILE_NAME"
#     echo "building ivf-flat with num_clusters=$VAR_NUM_CLUSTER, output_file=$VAR_OUTPUT_FILE_NAME" >> $ROOT/efes_clustering_flat.log
#     ./bench/datasets/bin/clustering $VAR_INPUT_FILE $VAR_OUTPUT_FILE_DIR/$VAR_OUTPUT_FILE_NAME 0 0 24 kmeans_sampled $VAR_NUM_CLUSTER 33554432 >> $ROOT/efes_clustering_flat.log 2>&1
#     echo "done building ivf-flat with num_clusters=$VAR_NUM_CLUSTER, output_file=$VAR_OUTPUT_FILE_NAME"
#     echo "done building ivf-flat with num_clusters=$VAR_NUM_CLUSTER, output_file=$VAR_OUTPUT_FILE_NAME" >> $ROOT/efes_clustering_flat.log
#     echo ============================================================================================================================================================================================================================================================================================================ >> $ROOT/efes_clustering_flat.log
# done

# cd $CURDIR

ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
CURDIR=$(pwd)
cd $ROOT

# std::cerr << "Usage: " << argv[0] << " <input_file> <output_file> <num_threads> <num_points> "
#                  "<max_iter> <algorithm> (<num_clusters> | <num_clusters> <sample_size> | (<cluster_cap> | <leaf_cap> <internal_cap> <weighted_kmeans>) <clustering_alg> [<max_num_clusters> | <coefficent>])\n";
VAR_INPUT_FILE=$ROOT/bench/datasets/bigann/unique_identified/BIGANN1B.u8bin
VAR_OUTPUT_FILE_DIR=$ROOT/bench/datasets/bigann/index/ctypeu8/tree/sen/lcap
# VAR_OUTPUT_FILE_NAME_LIST=(
#     # bigann32M_32LCAP_8ICAP_MAX4096_MI24
#     bigann32M_64LCAP_8ICAP_MAX4096_MI24

#     # ------------------
#     # bigann128M_2048LCAP_8ICAP_MAX512_MI24
#     # bigann128M_2048LCAP_128ICAP_MAX512_MI24
#     # bigann128M_2048LCAP_512ICAP_MAX512_MI24

#     # bigann128M_4096LCAP_8ICAP_MAX512_MI24
#     # bigann128M_4096LCAP_128ICAP_MAX512_MI24
#     # bigann128M_4096LCAP_512ICAP_MAX512_MI24

#     # bigann128M_8192LCAP_8ICAP_MAX512_MI24
#     # bigann128M_8192LCAP_32ICAP_MAX512_MI24
#     # bigann128M_8192LCAP_128ICAP_MAX512_MI24

#     # bigann128M_16384LCAP_8ICAP_MAX512_MI24
#     # bigann128M_16384LCAP_32ICAP_MAX512_MI24
#     # bigann128M_16384LCAP_64ICAP_MAX512_MI24
# )

# VAR_INTERNAL_CLUSTER_CAP_LISTS=(
#     # 8
#     8
#     # ---------------------
#     # 8
#     # 128
#     # 512
#     # 8
#     # 128
#     # 512
#     # 8
#     # 32
#     # 128
#     # 8
#     # 32
#     # 64
# )

# VAR_LEAF_CLUSTER_CAP_LISTS=(
#     # 32
#     64
#     # ---------------------
#     # 2048
#     # 2048
#     # 2048
#     # 4096
#     # 4096
#     # 4096
#     # 8192
#     # 8192
#     # 8192
#     # 16384
#     # 16384
#     # 16384
# )

VAR_OUTPUT_FILE_NAME_LIST=(
    # bigann32M_32LCAP_128ICAP_MAX4096_MI24
    bigann32M_64LCAP_128ICAP_MAX4096_MI24
    bigann32M_128LCAP_128ICAP_MAX4096_MI24
    bigann32M_256LCAP_128ICAP_MAX4096_MI24
    bigann32M_512LCAP_128ICAP_MAX4096_MI24
    bigann32M_8192LCAP_128ICAP_MAX4096_MI24
)

VAR_INTERNAL_CLUSTER_CAP_LISTS=(
    # 128
    128
    128
    128
    128
    128
)

VAR_LEAF_CLUSTER_CAP_LISTS=(
    # 32
    64
    128
    256
    512
    8192
)

VAR_NUM_COMMANDS=${#VAR_OUTPUT_FILE_NAME_LIST[@]}

> $ROOT/efes128_clustering_tree_sen.log

for ((i=0; i<$VAR_NUM_COMMANDS; i++)); do
    VAR_OUTPUT_FILE_NAME=${VAR_OUTPUT_FILE_NAME_LIST[i]}
    VAR_LEAF_CLUSTER_CAP=${VAR_LEAF_CLUSTER_CAP_LISTS[i]}
    VAR_INTERNAL_CLUSTER_CAP=${VAR_INTERNAL_CLUSTER_CAP_LISTS[i]}
    echo ============================================================================================================================================================================================================================================================================================================ >> $ROOT/efes128_clustering_tree_sen.log
    echo "building ivf-tree with leaf_cluster_cap=$VAR_LEAF_CLUSTER_CAP, internal_cluster_cap=$VAR_INTERNAL_CLUSTER_CAP, output_file=$VAR_OUTPUT_FILE_NAME"
    echo "building ivf-tree with leaf_cluster_cap=$VAR_LEAF_CLUSTER_CAP, internal_cluster_cap=$VAR_INTERNAL_CLUSTER_CAP, output_file=$VAR_OUTPUT_FILE_NAME" >> $ROOT/efes128_clustering_tree_sen.log
    OMP_NUM_THREADS=120 OMP_PROC_BIND=spread OMP_PLACES=cores nice -n 10 ./bench/datasets/bin/clustering $VAR_INPUT_FILE $VAR_OUTPUT_FILE_DIR/$VAR_OUTPUT_FILE_NAME 120 33554432 24 kmeans_hierarchical $VAR_LEAF_CLUSTER_CAP $VAR_INTERNAL_CLUSTER_CAP 0 max 4096 >> $ROOT/efes128_clustering_tree_sen.log 2>&1
    echo "done building ivf-tree with leaf_cluster_cap=$VAR_LEAF_CLUSTER_CAP, internal_cluster_cap=$VAR_INTERNAL_CLUSTER_CAP, output_file=$VAR_OUTPUT_FILE_NAME"
    echo "done building ivf-tree with leaf_cluster_cap=$VAR_LEAF_CLUSTER_CAP, internal_cluster_cap=$VAR_INTERNAL_CLUSTER_CAP, output_file=$VAR_OUTPUT_FILE_NAME" >> $ROOT/efes128_clustering_tree_sen.log
    echo ============================================================================================================================================================================================================================================================================================================ >> $ROOT/efes128_clustering_tree_sen.log
done

cd $CURDIR