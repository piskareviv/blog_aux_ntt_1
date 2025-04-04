#include <array>
#include <cassert>
#include <iostream>
#include <random>
#include <vector>

#include "ntt.hpp"

int32_t main() {
    constexpr auto prime_list = std::array{
        3,
        5,
        17,
        257,
        65537,
        int(1e9 + 7),
        int(1e9 + 9),
        int(1e9 + 1329),
        998'244'353,
    };

    for (const uint32_t mod : prime_list) {
        std::cerr << "testing mod: " << mod << std::endl;

        std::mt19937 rnd;
        NTT ntt(mod);
        auto gen_vec = [&](int n) {
            std::vector<uint32_t> vec(n);
            for (int i = 0; i < n; i++) {
                vec[i] = rnd() % mod;
            }
            return vec;
        };
        for (int lg = 0; (mod - 1) % (1 << lg) == 0; lg++) {
            clock_t beg = clock();
            for (int k = 0; k < std::max<int>(1, 1e6 / (1 << lg)); k++) {
                auto a = gen_vec(1 << lg), b = gen_vec(1 << lg);
                auto a2 = a, b2 = b;
                ntt.convolve_cyclic(lg, a2.data(), b2.data());

                std::vector<int> all(1 << lg);
                std::iota(all.begin(), all.end(), 0);
                std::swap(all.back(), all[std::min<int>(all.size() - 1, 1)]);
                std::shuffle(all.begin() + std::min<int>(all.size(), 2), all.end(), rnd);
                all.resize(std::min<int>(30, all.size()));
                for (int i : all) {
                    uint32_t val = 0;
                    for (int j = 0; j < (1 << lg); j++) {
                        val += a[j] * 1ULL * b[((1 << lg) + i - j) % (1 << lg)] % mod;
                        val %= mod;
                    }
                    assert(a2[i] == val);
                }
            }

            std::cerr.precision(5);
            std::cerr << std::fixed;
            std::cerr << lg << (lg < 10 ? "  " : " ") << (clock() - beg) * 1.0 / CLOCKS_PER_SEC << "s  OK" << std::endl;
        }
        std::cerr << std::endl;
    }
}
