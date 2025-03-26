#include <algorithm>
#include <cstdint>
#include <vector>

using u32 = uint32_t;
using u64 = uint64_t;

struct NTT {
    static constexpr u32 mod = 998'244'353;
    static constexpr u32 pr_root = 3;

    static u32 mul(u32 a, u32 b) {
        return u64(a) * b % mod;
    }
    static u32 add(u32 a, u32 b) {
        return a + b - mod * (a + b >= mod);
    }
    static u32 power(u32 b, u32 e) {
        u32 r = 1;
        for (; e > 0; e >>= 1) {
            if (e & 1)
                r = mul(r, b);
            b = mul(b, b);
        }
        return r;
    }

    std::vector<std::vector<u32>> w;
    std::vector<std::vector<int>> bit_rev;

    void expand(int k) {
        while (w.size() < k) {
            int t = w.size();
            w.emplace_back(1 << t);
            u32 f = power(pr_root, mod - 1 >> t + 1);
            w[t][0] = 1;
            for (int i = 1; i < (1 << t); i++) {
                w[t][i] = mul(w[t][i - 1], f);
            }
        }
        while (bit_rev.size() <= k) {
            int t = bit_rev.size();
            bit_rev.emplace_back(1 << t, 0);
            for (int i = 1; i < (1 << t); i++) {
                bit_rev[t][i] = (bit_rev[t][i >> 1] >> 1) | ((i & 1) << t - 1);
            }
        }
    }

    void butterfly_x2(u32& a, u32& b, u32 w) {
        u32 a1 = a, b1 = mul(b, w);
        a = add(a1, b1), b = add(a1, mod - b1);
    }

    void transform(int lg, u32* data) {
        expand(lg);
        for (int i = 0; i < (1 << lg); i++) {
            if (bit_rev[lg][i] < i) {
                std::swap(data[i], data[bit_rev[lg][i]]);
            }
        }
#ifndef ONLY_BIT_REVERSE
        for (int k = 0; k < lg; k++) {
            for (int i = 0; i < (1 << lg); i += (1 << k + 1)) {
                for (int j = 0; j < (1 << k); j++) {
                    butterfly_x2(data[i + j], data[i + (1 << k) + j], w[k][j]);
                }
            }
        }
#endif
    }

    void convolve_cyclic(int lg, u32* a, u32* b) {
#ifndef DO_NOTHING
        transform(lg, a);
        transform(lg, b);
#ifndef ONLY_BIT_REVERSE
        for (int i = 0; i < (1 << lg); i++) {
            a[i] = mul(a[i], b[i]);
        }
#endif
        transform(lg, a);
#ifndef ONLY_BIT_REVERSE
        std::reverse(a + 1, a + (1 << lg));
        u32 inv = power(mod + 1 >> 1, lg);
        for (int i = 0; i < (1 << lg); i++) {
            a[i] = mul(a[i], inv);
        }
#endif
#endif
    }
};
