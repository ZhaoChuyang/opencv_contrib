#include "precomp.hpp"

namespace cv{

    uint64 state = 0;
    RNG rng(state);

    void setSeed(uint64 seed){
        rng.state = seed;
    }

}