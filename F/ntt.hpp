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

    template <bool strict = true>
    u64x4 mul_u64x4(u64x4 a, u64x4 b) const {
        u64x4 pr = (u64x4)_mm256_mul_epu32((i256)a, (i256)b);
        u64x4 pr2 = (u64x4)_mm256_mul_epu32(_mm256_mul_epu32((i256)pr, (i256)n_inv), (i256)mod);
        u64x4 res = (u64x4)_mm256_bsrli_epi128(_mm256_add_epi64((i256)pr, (i256)pr2), 4);
        if (strict)
            res = (u64x4)shrink((u32x8)res);
        return res;
    }
};

class NTT {
   public:
    u32 mod, pr_root;

   private:
    static constexpr int LG = 32;  // more than enough for u32

    Montgomery mt;
    Montgomery_simd mts;

    u32 w[4], wr[4];
    u32 wd[LG], wrd[LG];

    u64x4 wt_init, wrt_init;
    u64x4 wd_x4[LG], wrd_x4[LG];

    u64x4 wl_init;
    u64x4 wld_x4[LG];

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

   public:
    NTT() = default;
    NTT(u32 mod) : mod(mod), mt(mod), mts(mod) {
        const Montgomery mt = this->mt;
        const Montgomery_simd mts = this->mts;

        pr_root = find_pr_root(mod, mt);

        int lg = __builtin_ctz(mod - 1);
        assert(lg <= LG);

        memset(w, 0, sizeof(w));
        memset(wr, 0, sizeof(wr));
        memset(wd_x4, 0, sizeof(wd_x4));
        memset(wrd_x4, 0, sizeof(wrd_x4));
        memset(wld_x4, 0, sizeof(wld_x4));

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
        wt_init = (u64x4)_mm256_setr_epi64x(w[0], w[0], w[0], w[1]);
        wrt_init = (u64x4)_mm256_setr_epi64x(wr[0], wr[0], wr[0], wr[1]);

        wl_init = (u64x4)_mm256_setr_epi64x(w[0], w[1], w[2], w[3]);

        u32 prf = mt.r, prf_r = mt.r;
        for (int i = 0; i < lg - 2; i++) {
            u32 f = mt.mul<true>(prf, vec[i + 3]), fr = mt.mul<true>(prf_r, vecr[i + 3]);
            prf = mt.mul<true>(prf, vecr[i + 3]), prf_r = mt.mul<true>(prf_r, vec[i + 3]);
            u32 f2 = mt.mul<true>(f, f), f2r = mt.mul<true>(fr, fr);

            wd_x4[i] = (u64x4)_mm256_setr_epi64x(f2, f, f2, f);
            wrd_x4[i] = (u64x4)_mm256_setr_epi64x(f2r, fr, f2r, fr);
        }

        prf = mt.r;
        for (int i = 0; i < lg - 3; i++) {
            u32 f = mt.mul<true>(prf, vec[i + 4]);
            prf = mt.mul<true>(prf, vecr[i + 4]);
            wld_x4[i] = (u64x4)_mm256_set1_epi64x(f);
        }
    }

   private:
    static constexpr int L0 = 3;
    int get_low_lg(int lg) const {
        return lg % 2 == L0 % 2 ? L0 : L0 + 1;
    }

    //    public:
    //     bool lg_available(int lg) {
    //         return L0 <= lg && lg <= __builtin_ctz(mod - 1) + get_low_lg(lg);
    //     }

   private:
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

    template <bool transposed, bool trivial = false>
    static void butterfly_x4(u32* ptr_a, u32* ptr_b, u32* ptr_c, u32* ptr_d, u32x8 w1, u32x8 w2, u32x8 w3, const Montgomery_simd& mts) {
        u32x8 a = load_u32x8(ptr_a), b = load_u32x8(ptr_b), c = load_u32x8(ptr_c), d = load_u32x8(ptr_d);
        if (!transposed) {
            butterfly_x2<false, trivial>((u32*)&a, (u32*)&c, w1, mts);
            butterfly_x2<false, trivial>((u32*)&b, (u32*)&d, w1, mts);
            butterfly_x2<false, trivial>((u32*)&a, (u32*)&b, w2, mts);
            butterfly_x2<false, false>((u32*)&c, (u32*)&d, w3, mts);
        } else {
            butterfly_x2<true, trivial>((u32*)&a, (u32*)&b, w2, mts);
            butterfly_x2<true, false>((u32*)&c, (u32*)&d, w3, mts);
            butterfly_x2<true, trivial>((u32*)&a, (u32*)&c, w1, mts);
            butterfly_x2<true, trivial>((u32*)&b, (u32*)&d, w1, mts);
        }
        store_u32x8(ptr_a, a), store_u32x8(ptr_b, b), store_u32x8(ptr_c, c), store_u32x8(ptr_d, d);
    }

    template <bool inverse, bool trivial = false>
    void transform_aux(int k, int i, u32* data, u64x4& wi, const Montgomery_simd& mts) const {
        u32x8 w1 = (u32x8)_mm256_shuffle_epi32((i256)wi, 0b00'00'00'00);
        u32x8 w2 = (u32x8)_mm256_permute4x64_epi64((i256)wi, 0b01'01'01'01);  // only even indices will be used
        u32x8 w3 = (u32x8)_mm256_permute4x64_epi64((i256)wi, 0b11'11'11'11);  // only even indices will be used
        for (int j = 0; j < (1 << k); j += 8) {
            butterfly_x4<inverse, trivial>(data + i + (1 << k) * 0 + j, data + i + (1 << k) * 1 + j,
                                           data + i + (1 << k) * 2 + j, data + i + (1 << k) * 3 + j,
                                           w1, w2, w3, mts);
        }
        wi = mts.mul_u64x4<true>(wi, (inverse ? wrd_x4 : wd_x4)[__builtin_ctz(~i >> k + 2)]);
    }

   public:
    // input in [0, 4 * mod)
    // output in [0, 4 * mod)
    // data must be 32-byte aligned
    void transform_forward(int lg, u32* data) const {
        const Montgomery_simd mts = this->mts;
        const int L = get_low_lg(lg);

        // for (int k = lg - 2; k >= L; k -= 2) {
        //     u64x4 wi = wt_init;
        //     transform_aux<false, true>(k, 0, data, wi, mts);
        //     for (int i = (1 << k + 2); i < (1 << lg); i += (1 << k + 2)) {
        //         transform_aux<false>(k, i, data, wi, mts);
        //     }
        // }

        if (L < lg) {
            const int lc = (lg - L) / 2;
            u64x4 wi_data[LG / 2];
            std::fill(wi_data, wi_data + lc, wt_init);

            for (int k = lg - 2; k >= L; k -= 2) {
                transform_aux<false, true>(k, 0, data, wi_data[k - L >> 1], mts);
            }
            for (int i = 1; i < (1 << lc * 2 - 2); i++) {
                int s = __builtin_ctz(i) >> 1;
                for (int k = s; k >= 0; k--) {
                    transform_aux<false>(2 * k + L, i * (1 << L + 2), data, wi_data[k], mts);
                }
            }
        }
    }

    // input in [0, 2 * mod)
    // output in [0, mod)
    // data must be 32-byte aligned
    template <bool mul_by_sc = false>
    void transform_inverse(int lg, u32* data, /* as normal number */ u32 sc = u32()) const {
        const Montgomery_simd mts = this->mts;
        const int L = get_low_lg(lg);

        // for (int k = L; k + 2 <= lg; k += 2) {
        //     u64x4 wi = wrt_init;
        //     transform_aux<true, true>(k, 0, data, wi, mts);
        //     for (int i = (1 << k + 2); i < (1 << lg); i += (1 << k + 2)) {
        //         transform_aux<true>(k, i, data, wi, mts);
        //     }
        // }

        if (L < lg) {
            const int lc = (lg - L) / 2;
            u64x4 wi_data[LG / 2];
            std::fill(wi_data, wi_data + lc, wrt_init);

            for (int i = 0; i < (1 << lc * 2 - 2); i++) {
                int s = __builtin_ctz(~i) >> 1;
                if (i + 1 == (1 << 2 * s)) {
                    s--;
                }
                for (int k = 0; k <= s; k++) {
                    transform_aux<true>(2 * k + L, (i + 1 - (1 << 2 * k)) * (1 << L + 2), data, wi_data[k], mts);
                }
                if (i + 1 == (1 << 2 * (s + 1))) {
                    s++;
                    transform_aux<true, true>(2 * s + L, (i + 1 - (1 << 2 * s)) * (1 << L + 2), data, wi_data[s], mts);
                }
            }
        }

        const Montgomery mt = this->mt;
        u32 f = mt.power<false, true>(mod + 1 >> 1, lg - L);
        if constexpr (mul_by_sc)
            f = mt.mul<true>(f, mt.mul<false>(mt.r2, sc));
        u32x8 f_x8 = (u32x8)_mm256_set1_epi32(f);
        for (int i = 0; i < (1 << lg); i += 8) {
            store_u32x8(data + i, mts.mul_u32x8<true, true>(load_u32x8(data + i), f_x8));
        }
    }

   private:
    // input in [0, 4 * mod)
    // output in [0, 2 * mod)
    // multiplies mod (x^2^L - w)
    template <int L, int K, bool remove_montgomery_reduction_factor = true>
    /* !!! O3 is crucial here !!! */ __attribute__((optimize("O3"))) static void aux_mul_mod_x2L(const u32* a, const u32* b, u32* c, const std::array<u32x8, K>& ar_w, const Montgomery_simd& mts) {
        static_assert(L >= 3);
        // static_assert(L == L0 || L == L0 + 1);

        constexpr int n = 1 << L;
        alignas(64) u32 aux_a[K][n];
        alignas(64) u64 aux_b[K][n * 2];
        for (int k = 0; k < K; k++) {
            for (int i = 0; i < n; i += 8) {
                u32x8 ai = load_u32x8(a + n * k + i);
                if constexpr (remove_montgomery_reduction_factor) {
                    ai = mts.mul_u32x8<true, true>(ai, mts.r2);
                } else {
                    ai = mts.shrink(mts.shrink2(ai));
                }
                store_u32x8(aux_a[k] + i, ai);

                u32x8 bi = load_u32x8(b + n * k + i);
                u32x8 bi_0 = mts.shrink(mts.shrink2(bi));
                u32x8 bi_w = mts.mul_u32x8<true, true>(bi, ar_w[k]);

                store_u32x8((u32*)(aux_b[k] + i + 0), (u32x8)_mm256_permutevar8x32_epi32((i256)bi_w, _mm256_setr_epi64x(0, 1, 2, 3)));
                store_u32x8((u32*)(aux_b[k] + i + 4), (u32x8)_mm256_permutevar8x32_epi32((i256)bi_w, _mm256_setr_epi64x(4, 5, 6, 7)));
                store_u32x8((u32*)(aux_b[k] + n + i + 0), (u32x8)_mm256_permutevar8x32_epi32((i256)bi_0, _mm256_setr_epi64x(0, 1, 2, 3)));
                store_u32x8((u32*)(aux_b[k] + n + i + 4), (u32x8)_mm256_permutevar8x32_epi32((i256)bi_0, _mm256_setr_epi64x(4, 5, 6, 7)));
            }
        }

        u64x4 aux_ans[K][n / 4];
        memset(aux_ans, 0, sizeof(aux_ans));
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < K; k++) {
                u64x4 ai = (u64x4)_mm256_set1_epi32(aux_a[k][i]);
                for (int j = 0; j < n; j += 4) {
                    u64x4 bi = (u64x4)_mm256_loadu_si256((i256*)(aux_b[k] + n - i + j));
                    aux_ans[k][j / 4] += /* 64-bit addition */ (u64x4)_mm256_mul_epu32((i256)ai, (i256)bi);
                }
            }
            if (i >= 8 && (i & 7) == 7) {
                for (int k = 0; k < K; k++) {
                    for (int j = 0; j < n; j += 4) {
                        aux_ans[k][j / 4] = (u64x4)mts.shrink2((u32x8)aux_ans[k][j / 4]);
                    }
                }
            }
        }

        for (int k = 0; k < K; k++) {
            for (int i = 0; i < n; i += 8) {
                u64x4 c0 = aux_ans[k][i / 4], c1 = aux_ans[k][i / 4 + 1];
                u32x8 res = (u32x8)_mm256_permutevar8x32_epi32((i256)mts.reduce<false>(c0, c1), _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7));
                store_u32x8(c + k * n + i, mts.shrink2(res));
            }
        }
    }

    template <int L, bool remove_montgomery_reduction_factor = true>
    void aux_mul_mod_full(int lg, const u32* a, const u32* b, u32* c) const {
        constexpr int sz = 1 << L;
        const Montgomery_simd mts = this->mts;
        int cnt = 1 << lg - L;
        if (cnt == 1) {
            aux_mul_mod_x2L<L, 1, remove_montgomery_reduction_factor>(a, b, c, {mts.r}, mts);
            return;
        }
        if (cnt <= 8) {
            for (int i = 0; i < cnt; i += 2) {
                u32x8 wi = (u32x8)_mm256_set1_epi32(w[i / 2]);
                aux_mul_mod_x2L<L, 2, remove_montgomery_reduction_factor>(a + i * sz, b + i * sz, c + i * sz, {wi, (mts.mod - wi)}, mts);
            }
            return;
        }
        u64x4 wi = wl_init;
        for (int i = 0; i < cnt; i += 8) {
            u32x8 w_ar[4] = {
                (u32x8)_mm256_permute4x64_epi64((i256)wi, 0b00'00'00'00),
                (u32x8)_mm256_permute4x64_epi64((i256)wi, 0b01'01'01'01),
                (u32x8)_mm256_permute4x64_epi64((i256)wi, 0b10'10'10'10),
                (u32x8)_mm256_permute4x64_epi64((i256)wi, 0b11'11'11'11),
            };
            if constexpr (L == L0) {
                for (int j = 0; j < 8; j += 4) {
                    aux_mul_mod_x2L<L, 4, remove_montgomery_reduction_factor>(a + (i + j) * sz, b + (i + j) * sz, c + (i + j) * sz,
                                                                              {w_ar[j / 2], mts.mod - w_ar[j / 2], w_ar[j / 2 + 1], mts.mod - w_ar[j / 2 + 1]}, mts);
                }
            } else {
                for (int j = 0; j < 8; j += 2) {
                    aux_mul_mod_x2L<L, 2, remove_montgomery_reduction_factor>(a + (i + j) * sz, b + (i + j) * sz, c + (i + j) * sz,
                                                                              {w_ar[j / 2], mts.mod - w_ar[j / 2]}, mts);
                }
            }
            wi = mts.mul_u64x4<true>(wi, wld_x4[__builtin_ctz(~i >> 3)]);
        }
    }

   public:
    template <bool remove_montgomery_reduction_factor = true>
    void aux_dot_mod(int lg, const u32* a, const u32* b, u32* c) const {
        int L = get_low_lg(lg);
        if (L == L0) {
            aux_mul_mod_full<L0, remove_montgomery_reduction_factor>(lg, a, b, c);
        } else {
            aux_mul_mod_full<L0 + 1, remove_montgomery_reduction_factor>(lg, a, b, c);
        }
    }

    // lg must be greater than or equal to 3
    // a, b must be 32-byte aligned
    void convolve_cyclic(int lg, u32* a, u32* b) const {
        transform_forward(lg, a);
        transform_forward(lg, b);
        aux_dot_mod(lg, a, b, a);
        transform_inverse(lg, a);
    }
};
