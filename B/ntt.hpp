#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <vector>

#pragma GCC target("bmi")

using u32 = uint32_t;
using u64 = uint64_t;

struct Montgomery {
    u32 mod;    // mod
    u32 mod2;   // 2 * mod
    u32 n_inv;  // n_inv * mod == -1 (mod 2^32)
    u32 r;      // 2^32 % mod
    u32 r2;     // (2^32)^2 % mod

    Montgomery() = default;
    Montgomery(u32 mod) : mod(mod) {
        assert(mod % 2 == 1);
        assert(mod < (1 << 30));
        mod2 = 2 * mod;
        n_inv = 1;
        for (int i = 0; i < 5; i++) {
            n_inv *= 2 + n_inv * mod;
        }
        r = (u64(1) << 32) % mod;
        r2 = u64(r) * r % mod;
    }

    u32 shrink(u32 val) const {
        return std::min(val, val - mod);
    }
    u32 shrink2(u32 val) const {
        return std::min(val, val - mod2);
    }

    template <bool strict = true>
    u32 reduce(u64 val) const {
        u32 res = val + u32(val) * n_inv * u64(mod) >> 32;
        if constexpr (strict)
            res = shrink(res);
        return res;
    }

    template <bool strict = true>
    u32 mul(u32 a, u32 b) const {
        return reduce<strict>(u64(a) * b);
    }

    template <bool input_in_space = false, bool output_in_space = false>
    u32 power(u32 b, u32 e) const {
        if (!input_in_space)
            b = mul<false>(b, r2);
        u32 r = output_in_space ? this->r : 1;
        for (; e > 0; e >>= 1) {
            if (e & 1)
                r = mul<false>(r, b);
            b = mul<false>(b, b);
        }
        return shrink(r);
    }
};

struct NTT {
    u32 mod, pr_root;
    Montgomery mt;
    std::vector<u32> wd, wrd;

    static u32 find_pr_root(u32 mod, const Montgomery& mt) {
        std::vector<u32> factors;
        u32 n = mod - 1;
        for (u32 i = 2; u64(i) * i <= n; i++) {
            if (n % i == 0) {
                factors.push_back(i);
                do {
                    n /= i;
                } while (n % i == 0);
            }
        }
        if (n > 1) {
            factors.push_back(n);
        }
        for (u32 i = 2; i < mod; i++) {
            if (std::all_of(factors.begin(), factors.end(), [&](u32 f) { return mt.power<false, false>(i, (mod - 1) / f) != 1; })) {
                return i;
            }
        }
        assert(false && "primitive root not found");
    }

    NTT() = default;
    NTT(u32 mod) : mod(mod), mt(mod) {
        const Montgomery mt = this->mt;
        pr_root = find_pr_root(mod, mt);

        int lg = __builtin_ctz(mod - 1);
        wd.assign(lg, 0), wrd.assign(lg, 0);
        for (int i = 0; i < lg - 1; i++) {
            u32 a = mt.power<false, true>(pr_root, mod - 1 >> i + 2);
            u32 b = mt.power<false, true>(pr_root, (mod - 1 >> i + 1) * ((1 << i) - 1));
            u32 f = mt.mul<true>(a, mt.power<true, true>(b, mod - 2));
            wd[i] = f, wrd[i] = mt.power<true, true>(f, mod - 2);
        }
    }

    template <bool transposed, bool trivial = false>
    static void butterfly_x2(u32& a, u32& b, u32 w, const Montgomery& mt) {
        if (!transposed) {
            u32 a1 = mt.shrink2(a), b1 = trivial ? mt.shrink2(b) : mt.mul<false>(b, w);
            a = a1 + b1, b = a1 + mt.mod2 - b1;
        } else {
            u32 a2 = mt.shrink2(a + b), b2 = trivial ? mt.shrink2(a + mt.mod2 - b) : mt.mul<false>(a + mt.mod2 - b, w);
            a = a2, b = b2;
        }
    }

    template <bool inverse, bool trivial = false>
    void transform_aux(int k, int i, u32* data, u32& wi, const Montgomery& mt) const {
        for (int j = 0; j < (1 << k); j++) {
            butterfly_x2<inverse, trivial>(data[i + j], data[i + (1 << k) + j], wi, mt);
        }
        wi = mt.mul<true>(wi, (inverse ? wrd : wd)[__builtin_ctz(~i >> k + 1)]);
    }

    // input in [0, 4 * mod)
    // output in [0, 4 * mod)
    void transform_forward(int lg, u32* data) const {
        const Montgomery mt = this->mt;
        for (int k = lg - 1; k >= 0; k--) {
            u32 wi = mt.r;
            transform_aux<false, true>(k, 0, data, wi, mt);
            for (int i = (1 << k + 1); i < (1 << lg); i += (1 << k + 1)) {
                transform_aux<false>(k, i, data, wi, mt);
            }
        }
    }

    // input in [0, 2 * mod)
    // output in [0, mod)
    template <bool mul_by_sc = false>
    void transform_inverse(int lg, u32* data, /* as normal number */ u32 sc = u32()) const {
        const Montgomery mt = this->mt;
        for (int k = 0; k < lg; k++) {
            u32 wi = mt.r;
            transform_aux<true, true>(k, 0, data, wi, mt);
            for (int i = (1 << k + 1); i < (1 << lg); i += (1 << k + 1)) {
                transform_aux<true>(k, i, data, wi, mt);
            }
        }

        u32 f = mt.power<false, true>(mod + 1 >> 1, lg);
        if constexpr (mul_by_sc)
            f = mt.mul<true>(f, mt.mul<false>(mt.r2, sc));
        for (int i = 0; i < (1 << lg); i++) {
            data[i] = mt.mul<true>(data[i], f);
        }
    }

    void convolve_cyclic(int lg, u32* a, u32* b) const {
        transform_forward(lg, a);
        transform_forward(lg, b);
        const Montgomery mt = this->mt;
        for (int i = 0; i < (1 << lg); i++) {
            a[i] = mt.mul<false>(mt.shrink2(a[i]), mt.shrink2(b[i]));
        }
        transform_inverse<true>(lg, a, mt.r);
    }
};
