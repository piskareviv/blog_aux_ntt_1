#include <immintrin.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>

#pragma GCC target("avx2,bmi")

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

using i256 = __m256i;
using u32x8 = u32 __attribute__((vector_size(32)));
using u64x4 = u64 __attribute__((vector_size(32)));

u32x8 load_u32x8(const u32* ptr) {
    return (u32x8)_mm256_load_si256((const i256*)ptr);
}
void store_u32x8(u32* ptr, u32x8 vec) {
    _mm256_store_si256((i256*)ptr, (i256)vec);
}

struct Montgomery_simd {
    u32x8 mod;    // mod
    u32x8 mod2;   // 2 * mod
    u32x8 n_inv;  // n_inv * mod == -1 (mod 2^32)
    u32x8 r;      // 2^32 % mod
    u32x8 r2;     // (2^32)^2 % mod

    Montgomery_simd() = default;
    Montgomery_simd(u32 mod) {
        Montgomery mt(mod);
        this->mod = (u32x8)_mm256_set1_epi32(mt.mod);
        this->mod2 = (u32x8)_mm256_set1_epi32(mt.mod2);
        this->n_inv = (u32x8)_mm256_set1_epi32(mt.n_inv);
        this->r = (u32x8)_mm256_set1_epi32(mt.r);
        this->r2 = (u32x8)_mm256_set1_epi32(mt.r2);
    }

    u32x8 shrink(u32x8 vec) const {
        return (u32x8)_mm256_min_epu32((i256)vec, _mm256_sub_epi32((i256)vec, (i256)mod));
    }
    u32x8 shrink2(u32x8 vec) const {
        return (u32x8)_mm256_min_epu32((i256)vec, _mm256_sub_epi32((i256)vec, (i256)mod2));
    }
    u32x8 shrink_n(u32x8 vec) const {
        return (u32x8)_mm256_min_epu32((i256)vec, _mm256_add_epi32((i256)vec, (i256)mod));
    }
    u32x8 shrink2_n(u32x8 vec) const {
        return (u32x8)_mm256_min_epu32((i256)vec, _mm256_add_epi32((i256)vec, (i256)mod2));
    }

    template <bool strict = true>
    u32x8 reduce(u64x4 x0246, u64x4 x1357) const {
        u64x4 x0246_ninv = (u64x4)_mm256_mul_epu32((i256)x0246, (i256)n_inv);
        u64x4 x1357_ninv = (u64x4)_mm256_mul_epu32((i256)x1357, (i256)n_inv);
        u64x4 x0246_res = (u64x4)_mm256_add_epi64((i256)x0246, _mm256_mul_epu32((i256)x0246_ninv, (i256)mod));
        u64x4 x1357_res = (u64x4)_mm256_add_epi64((i256)x1357, _mm256_mul_epu32((i256)x1357_ninv, (i256)mod));
        u32x8 res = (u32x8)_mm256_or_si256(_mm256_bsrli_epi128((i256)x0246_res, 4), (i256)x1357_res);
        if (strict)
            res = shrink(res);
        return res;
    }

    template <bool strict = true, bool b_use_only_even = false>
    u32x8 mul_u32x8(u32x8 a, u32x8 b) const {
        u32x8 a_sh = (u32x8)_mm256_bsrli_epi128((i256)a, 4);
        u32x8 b_sh = b_use_only_even ? b : (u32x8)_mm256_bsrli_epi128((i256)b, 4);
        u64x4 x0246 = (u64x4)_mm256_mul_epu32((i256)a, (i256)b);
        u64x4 x1357 = (u64x4)_mm256_mul_epu32((i256)a_sh, (i256)b_sh);
        return reduce<strict>(x0246, x1357);
    }
};

struct NTT {
    static constexpr int LG = 32;  // more than enough for u32

    u32 mod, pr_root;
    Montgomery mt;
    Montgomery_simd mts;
    u32 w[4], wr[4];
    u32 wd[LG], wrd[LG];

    u32x8 wl_init, wrl_init;
    u32x8 wld_x8[LG], wrld_x8[LG];

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
    NTT(u32 mod) : mod(mod), mt(mod), mts(mod) {
        const Montgomery mt = this->mt;
        const Montgomery_simd mts = this->mts;

        pr_root = find_pr_root(mod, mt);

        int lg = __builtin_ctz(mod - 1);
        assert(lg <= LG);

        memset(w, 0, sizeof(w));
        memset(wr, 0, sizeof(wr));
        memset(wd, 0, sizeof(wd));
        memset(wrd, 0, sizeof(wrd));
        memset(wld_x8, 0, sizeof(wld_x8));
        memset(wrld_x8, 0, sizeof(wrld_x8));

        std::vector<u32> vec(lg + 1), vecr(lg + 1);
        vec[lg] = mt.power<false, true>(pr_root, mod - 1 >> lg);
        vecr[lg] = mt.power<true, true>(vec[lg], mod - 2);
        for (int i = lg - 1; i >= 0; i--) {
            vec[i] = mt.mul<true>(vec[i + 1], vec[i + 1]);
            vecr[i] = mt.mul<true>(vecr[i + 1], vecr[i + 1]);
        }

        w[0] = wr[0] = mt.r;
        if (lg >= 2) {
            w[1] = vec[2], wr[1] = vecr[2];
            if (lg >= 3) {
                w[2] = vec[3], wr[2] = vecr[3];
                w[3] = mt.mul<true>(w[1], w[2]);
                wr[3] = mt.mul<true>(wr[1], wr[2]);
            }
        }
        wl_init = (u32x8)_mm256_setr_epi32(mt.r, w[1], w[2], w[3], mt.r, w[1], mt.r, mt.r);
        wrl_init = (u32x8)_mm256_setr_epi32(mt.r, wr[1], wr[2], wr[3], mt.r, wr[1], mt.r, mt.r);

        u32 prf = mt.r, prf_r = mt.r;
        for (int i = 0; i < lg - 1; i++) {
            // u32 f = mt.power<false, true>(pr_root, (mod - 1 >> i + 2) * ((1 << i + 2) + 1 - ((2 << i) - 2)));
            // u32 fr = mt.power<1, 1>(f, mod - 2);
            u32 f = mt.mul<true>(prf, vec[i + 2]), fr = mt.mul<true>(prf_r, vecr[i + 2]);
            prf = mt.mul<true>(prf, vecr[i + 2]), prf_r = mt.mul<true>(prf_r, vec[i + 2]);

            wd[i] = f, wrd[i] = fr;
        }

        prf = mt.r, prf_r = mt.r;
        for (int i = 0; i < lg - 3; i++) {
            // u32 f1 = mt.power<false, true>(pr_root, (mod - 1 >> i + 4) * ((1 << i + 4) + 1 - ((2 << i) - 2)));
            // u32 f1r = mt.power<1, 1>(f1, mod - 2);
            u32 f1 = mt.mul<true>(prf, vec[i + 4]), f1r = mt.mul<true>(prf_r, vecr[i + 4]);
            prf = mt.mul<true>(prf, vecr[i + 4]), prf_r = mt.mul<true>(prf_r, vec[i + 4]);

            u32 f2 = mt.mul<true>(f1, f1);
            u32 f3 = mt.mul<true>(f2, f2);
            wld_x8[i] = (u32x8)_mm256_setr_epi32(f1, f1, f1, f1, f2, f2, f3, mt.r);

            u32 f2r = mt.mul<true>(f1r, f1r);
            u32 f3r = mt.mul<true>(f2r, f2r);
            wrld_x8[i] = (u32x8)_mm256_setr_epi32(f1r, f1r, f1r, f1r, f2r, f2r, f3r, mt.r);
        }
    }

    template <bool transposed, bool trivial = false>
    static void butterfly_x2(u32* ptr_a, u32* ptr_b, u32x8 w, const Montgomery_simd& mts) {
        u32x8 a = load_u32x8(ptr_a), b = load_u32x8(ptr_b);
        u32x8 a2, b2;
        if (!transposed) {
            a = mts.shrink2(a), b = trivial ? mts.shrink2(b) : mts.mul_u32x8<false, true>(b, w);
            a2 = a + b, b2 = a + mts.mod2 - b;
        } else {
            a2 = mts.shrink2(a + b), b2 = trivial ? mts.shrink2_n(a - b) : mts.mul_u32x8<false, true>(a + mts.mod2 - b, w);
        }
        store_u32x8(ptr_a, a2), store_u32x8(ptr_b, b2);
    }

    template <bool inverse, bool trivial = false>
    void transform_aux(int k, int i, u32* data, u32& wi, const Montgomery& mt, const Montgomery_simd& mts) const {
        u32x8 wi_x8 = (u32x8)_mm256_set1_epi32(wi);
        for (int j = 0; j < (1 << k); j += 8) {
            butterfly_x2<inverse, trivial>(data + i + j, data + i + (1 << k) + j, wi_x8, mts);
        }
        wi = mt.mul<true>(wi, (inverse ? wrd : wd)[__builtin_ctz(~i >> k + 1)]);
    }

    // input in [0, 4 * mod)
    // output in [0, 4 * mod)
    // data must be 32-byte aligned
    void transform_forward(int lg, u32* data) const {
        const Montgomery mt = this->mt;
        const Montgomery_simd mts = this->mts;

        for (int k = lg - 1; k >= 3; k--) {
            u32 wi = mt.r;
            transform_aux<false, true>(k, 0, data, wi, mt, mts);
            for (int i = (1 << k + 1); i < (1 << lg); i += (1 << k + 1)) {
                transform_aux<false>(k, i, data, wi, mt, mts);
            }
        }

        u32x8 wi = wl_init;
        for (int i = 0; i < (1 << lg); i += 8) {
            u32x8 w0 = (u32x8)_mm256_permutevar8x32_epi32((i256)wi, _mm256_setr_epi32(7, 7, 7, 7, 6, 6, 6, 6));
            u32x8 w1 = (u32x8)_mm256_permutevar8x32_epi32((i256)wi, _mm256_setr_epi32(7, 7, 4, 4, 7, 7, 5, 5));
            u32x8 w2 = (u32x8)_mm256_permutevar8x32_epi32((i256)wi, _mm256_setr_epi32(7, 0, 7, 1, 7, 2, 7, 3));

            u32x8 vec = load_u32x8(data + i);
            vec = mts.mul_u32x8<false>(vec, w0);
            vec = mts.mul_u32x8<false>((u32x8)_mm256_blend_epi32((i256)vec, (i256)(mts.mod2 - vec), 0b1111'0000) + (u32x8)_mm256_permute2x128_si256((i256)vec, (i256)vec, 1), w1);
            vec = mts.mul_u32x8<false>((u32x8)_mm256_blend_epi32((i256)vec, (i256)(mts.mod2 - vec), 0b1100'1100) + (u32x8)_mm256_shuffle_epi32((i256)vec, 0b01'00'11'10), w2);
            vec = mts.shrink2((u32x8)_mm256_blend_epi32((i256)vec, (i256)(mts.mod2 - vec), 0b1010'1010) + (u32x8)_mm256_shuffle_epi32((i256)vec, 0b10'11'00'01));
            store_u32x8(data + i, vec);

            wi = mts.mul_u32x8<true>(wi, wld_x8[__builtin_ctz(~i >> 3)]);
        }
    }

    // input in [0, 2 * mod)
    // output in [0, mod)
    // data must be 32-byte aligned
    template <bool mul_by_sc = false>
    void transform_inverse(int lg, u32* data, /* as normal number */ u32 sc = u32()) const {
        const Montgomery mt = this->mt;
        const Montgomery_simd mts = this->mts;

        u32x8 wi = wrl_init;
        for (int i = 0; i < (1 << lg); i += 8) {
            u32x8 w0 = (u32x8)_mm256_permutevar8x32_epi32((i256)wi, _mm256_setr_epi32(7, 7, 7, 7, 6, 6, 6, 6));
            u32x8 w1 = (u32x8)_mm256_permutevar8x32_epi32((i256)wi, _mm256_setr_epi32(7, 7, 4, 4, 7, 7, 5, 5));
            u32x8 w2 = (u32x8)_mm256_permutevar8x32_epi32((i256)wi, _mm256_setr_epi32(7, 0, 7, 1, 7, 2, 7, 3));

            u32x8 vec = load_u32x8(data + i);
            vec = mts.mul_u32x8<false>((u32x8)_mm256_blend_epi32((i256)vec, (i256)(mts.mod2 - vec), 0b1010'1010) + (u32x8)_mm256_shuffle_epi32((i256)vec, 0b10'11'00'01), w2);
            vec = mts.mul_u32x8<false>((u32x8)_mm256_blend_epi32((i256)vec, (i256)(mts.mod2 - vec), 0b1100'1100) + (u32x8)_mm256_shuffle_epi32((i256)vec, 0b01'00'11'10), w1);
            vec = mts.mul_u32x8<false>((u32x8)_mm256_blend_epi32((i256)vec, (i256)(mts.mod2 - vec), 0b1111'0000) + (u32x8)_mm256_permute2x128_si256((i256)vec, (i256)vec, 1), w0);
            store_u32x8(data + i, vec);

            wi = mts.mul_u32x8<true>(wi, wrld_x8[__builtin_ctz(~i >> 3)]);
        }

        for (int k = 3; k < lg; k++) {
            u32 wi = mt.r;
            transform_aux<true, true>(k, 0, data, wi, mt, mts);
            for (int i = (1 << k + 1); i < (1 << lg); i += (1 << k + 1)) {
                transform_aux<true>(k, i, data, wi, mt, mts);
            }
        }

        u32 f = mt.power<false, true>(mod + 1 >> 1, lg);
        if constexpr (mul_by_sc)
            f = mt.mul<true>(f, mt.mul<false>(mt.r2, sc));

        u32x8 f_x8 = (u32x8)_mm256_set1_epi32(f);
        for (int i = 0; i < (1 << lg); i += 8) {
            store_u32x8(data + i, mts.mul_u32x8<true, true>(load_u32x8(data + i), f_x8));
        }
    }

    // a, b must be 32-byte aligned
    void convolve_cyclic(int lg, u32* a, u32* b) const {
        assert(lg >= 3);
        transform_forward(lg, a);
        transform_forward(lg, b);
        const Montgomery_simd mts = this->mts;
        for (int i = 0; i < (1 << lg); i += 8) {
            store_u32x8(a + i, mts.mul_u32x8<false>(mts.shrink2(load_u32x8(a + i)), mts.shrink2(load_u32x8(b + i))));
        }
        transform_inverse<true>(lg, a, mt.r);
    }
};
