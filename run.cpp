#include <immintrin.h>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <numeric>
#include <vector>

#include "ntt.hpp"

int32_t main(int argc, char** argv) {
    assert(argc >= 3);
    int lg = std::atoi(argv[1]);
    int cnt = std::atoi(argv[2]);

    NTT ntt;

    u32* a = (u32*)_mm_malloc(4LL << lg, 64);
    u32* b = (u32*)_mm_malloc(4LL << lg, 64);
    std::fill(a, a + (1LL << lg), 1);
    std::iota(b, b + (1LL << lg), 0);

    ntt.convolve_cyclic(lg, a, b);

    clock_t beg = clock();
    for (int i = 0; i < cnt; i++) {
        ntt.convolve_cyclic(lg, a, b);
    }
    double tm = (clock() - beg) * 1.0 / CLOCKS_PER_SEC;

    std::cout.precision(20);
    std::cout << std::fixed;
    std::cout << tm << std::endl;

    _mm_free(a), _mm_free(b);
}
