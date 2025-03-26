#include <algorithm>
#include <cstdint>
#include <vector>

#pragma GCC target("bmi")

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

    std::vector<u32> wd, wrd;

    NTT() {
        int lg = __builtin_ctz(mod - 1);
        wd.assign(lg, 0), wrd.assign(lg, 0);
        for (int i = 0; i < lg - 1; i++) {
            u32 a = power(pr_root, mod - 1 >> i + 2);
            u32 b = power(pr_root, (mod - 1 >> i + 1) * ((1 << i) - 1));
            u32 f = mul(a, power(b, mod - 2));
            wd[i] = f, wrd[i] = power(f, mod - 2);
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
        for (int k = lg - 1; k >= 0; k--) {
            u32 wi = 1;
            for (int i = 0; i < (1 << lg); i += (1 << k + 1)) {
                for (int j = 0; j < (1 << k); j++) {
                    butterfly_x2<false>(data[i + j], data[i + (1 << k) + j], wi);
                }
                wi = mul(wi, wd[__builtin_ctz(~i >> k + 1)]);
            }
        }
    }

    void transform_inverse(int lg, u32* data) {
        for (int k = 0; k < lg; k++) {
            u32 wi = 1;
            for (int i = 0; i < (1 << lg); i += (1 << k + 1)) {
                for (int j = 0; j < (1 << k); j++) {
                    butterfly_x2<true>(data[i + j], data[i + (1 << k) + j], wi);
                }
                wi = mul(wi, wrd[__builtin_ctz(~i >> k + 1)]);
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
