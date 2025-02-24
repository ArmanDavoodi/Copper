#include "debug.h"
#include "vector_utils.h"

int main() {
    CLOG(LOG_LEVEL_LOG, LOG_TAG_TEST, "Test Vector_Utils START.");

    uint16_t dim = 8;
    uint16_t size = 3;

    uint16_t _data16[dim * size] = {1, 2, 3, 4, 5, 6, 7, 8, 
                                    9, 10, 11, 12, 13, 14, 15, 16, 
                                    17, 18, 19, 20, 21, 22, 23, 24};
    uint64_t _ids16[dim * size] = {1ul, 2ul, 3ul};
    copper::VectorID* ids16 = static_cast<copper::VectorID*>((void*)_ids16);

    float _dataf[dim * size] = {0.2f, 25.6f, -12.2f, 1.112f, 36.0f, 7.5f, -3.3f, 8.8f, 
                                9.1f, -4.6f, 5.5f, 2.2f, 3.3f, -1.1f, 6.6f, 7.7f, 
                                8.8f, 9.9f, -10.1f, 11.2f, 12.3f, -13.4f, 14.5f, 15.6f};
    uint64_t _idsf[dim * size] = {4ul, 5ul, 6ul};
    copper::VectorID* idsf = static_cast<copper::VectorID*>((void*)_idsf);

    copper::Vector vec = _data16;
    CLOG(LOG_LEVEL_LOG, LOG_TAG_TEST, "Uint Vector: %s", copper::to_string<uint16_t>(vec, dim).c_str());

    vec = _dataf;
    CLOG(LOG_LEVEL_LOG, LOG_TAG_TEST, "Float Vector: %s", copper::to_string<float>(vec, dim).c_str());

    
    copper::VectorSet uset{_data16, ids16, size};
    CLOG(LOG_LEVEL_LOG, LOG_TAG_TEST, "Uint VectorSet: %s", uset.to_string<uint16_t>(dim).c_str());

    copper::VectorSet fset{_dataf, idsf, size};
    CLOG(LOG_LEVEL_LOG, LOG_TAG_TEST, "Float VectorSet: %s", fset.to_string<float>(dim).c_str());
    
    copper::Distance<uint16_t> *_du = new copper::L2_Distance<uint16_t>;
    copper::Distance<float> *_df = new copper::L2_Distance<float>;
    
    double udist[size][size], fdist[size][size];

    double b_udist[size][size] = {{0.0, 512.0, 2048.0}, {512.0, 0.0, 512.0}, {2048.0, 512.0, 0.0}};
    // I used the python dist to calculate these then I powered each by 2 so they may not be the correct values
    double b_fdist[size][size] = {{0.0, 2548.193744, 1788.207744}, {2548.193744, 0.0, 891.81}, {1788.207744, 891.81, 0.0}};

    for (uint16_t i = 0; i < size; ++i) {
        for (uint16_t j = 0; j < size; ++j) {
            if (j > i)
                continue;

            udist[i][j] = ((*_du)(uset.Get_Vector<uint16_t>(i, dim), uset.Get_Vector<uint16_t>(j, dim), dim));
            AssertError(udist[i][j] == b_udist[i][j], LOG_TAG_TEST, 
                        "uint16 distance calculation err: calc=%f, ground_truth=%f", udist[i][j], b_udist[i][j]);
            fdist[i][j] = ((*_df)(fset.Get_Vector<float>(i, dim), fset.Get_Vector<float>(j, dim), dim));
            AssertError(fdist[i][j] == b_fdist[i][j], LOG_TAG_TEST, 
                "float distance calculation err: calc=%f, ground_truth=%f", fdist[i][j], b_fdist[i][j]);
        }
    }
    
    //  todo check _dist operator for distances
    //  todo check effect of similarity for heap
    //  todo check VectorSet functions
    
    //  todo check VectorIndex operations
    
    CLOG(LOG_LEVEL_LOG, LOG_TAG_TEST, "Test Vector_Utils END.");
}