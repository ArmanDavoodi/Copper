#ifndef COPPER_BUFFER_H_
#define COPPER_BUFFER_H_

#include "common.h"
#include "vector_utils.h"

namespace copper {

template <typename T, uint16_t min, uint16_t max> class Copper_Node;

// todo implement buffer
template <typename T, uint16_t ki_min, uint16_t ki_max, uint16_t kl_min, uint16_t kl_max>
class Buffer_Manager {
public:
    inline Copper_Node<T, ki_min, ki_max>* Get_Node(VectorID& node_id);
    inline Copper_Node<T, kl_min, kl_max>* Get_Leaf(VectorID& leaf_id);
    inline Copper_Node<T, kl_min, kl_max>* Get_Container_Leaf(VectorID& vec_id);
    inline Vector Get_Vector(VectorID id);

};

};

#endif