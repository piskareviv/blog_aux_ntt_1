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

    std::vector<u32> w, wr;

    void expand(int k) {
        while (w.size() * 2 < (1 << k)) {
            if (w.size() == 0) {
                w = {1}, wr = {1};
                continue;
            }
            int lg = __builtin_ctz(w.size());
            w.resize(1 << lg + 1), wr.resize(1 << lg + 1);
            u32 f = power(pr_root, mod - 1 >> lg + 2), fr = power(f, mod - 2);
            for (int i = 0; i < (1 << lg); i++) {
                w[i + (1 << lg)] = mul(w[i], f);
                wr[i + (1 << lg)] = mul(wr[i], fr);
            }
        }
    }

    template <bool transposed>
    void butterfly_x2(u32& a, u32& b, u32 w) {
        if (!transposed) {
            u32 a1 = a, b1 = mul(b, w);
            a = add(a1, b1), b = add(a1, mod - b1);
        } else {
            u32 a2 = add(a, b), b2 = mul(add(a, mod - b), w);
            a = a2, b = b2;
        }
    }

    void transform_forward(int lg, u32* data) {
        expand(lg);
        for (int k = lg - 1; k >= 0; k--) {
            for (int i = 0; i < (1 << lg); i += (1 << k + 1)) {
                u32 wi = w[i >> k + 1];
                for (int j = 0; j < (1 << k); j++) {
                    butterfly_x2<false>(data[i + j], data[i + (1 << k) + j], wi);
                }
            }
        }
    }

    void transform_inverse(int lg, u32* data) {
        expand(lg);
        for (int k = 0; k < lg; k++) {
            for (int i = 0; i < (1 << lg); i += (1 << k + 1)) {
                u32 wi = wr[i >> k + 1];
                for (int j = 0; j < (1 << k); j++) {
                    butterfly_x2<true>(data[i + j], data[i + (1 << k) + j], wi);
                }
            }
        }
        u32 inv = power(mod + 1 >> 1, lg);
        for (int i = 0; i < (1 << lg); i++) {
            data[i] = mul(data[i], inv);
        }
    }

    void convolve_cyclic(int lg, u32* a, u32* b) {
        transform_forward(lg, a);
        transform_forward(lg, b);
        for (int i = 0; i < (1 << lg); i++) {
            a[i] = mul(a[i], b[i]);
        }
        transform_inverse(lg, a);
    }
};
