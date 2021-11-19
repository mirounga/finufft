// Defines templates for the tiled interpolation code
// And optimized specializations.

#ifndef PIPEDINTERP_H
#define PIPEDINTERP_H

#include <defs.h>
#include <tiledeval.h>
#include <utils_precindep.h>

#include <cassert>
#include <utility>
#include <immintrin.h>

using namespace std;

// declarations of purely internal functions...
void interp_line(FLT *out,FLT *du, FLT *ker,BIGINT i1,BIGINT N1,int ns);
void interp_square(FLT *out,FLT *du, FLT *ker1, FLT *ker2, BIGINT i1,BIGINT i2,BIGINT N1,BIGINT N2,int ns);
void interp_cube(FLT *out,FLT *du, FLT *ker1, FLT *ker2, FLT *ker3,
		 BIGINT i1,BIGINT i2,BIGINT i3,BIGINT N1,BIGINT N2,BIGINT N3,int ns);

template<int ns>
void tiled_interp_line(FLT* target, FLT* du, FLT** ker, BIGINT* i1, BIGINT N1)
// 1D interpolate complex values from du array to out, using real weights
// ker[0] through ker[ns-1]. out must be size 2 (real,imag), and du
// of size 2*N1 (alternating real,imag). i1 is the left-most index in [0,N1)
// Periodic wrapping in the du array is applied, assuming N1>=ns.
// dx is index into ker array, j index in complex du (data_uniform) array.
// Barnett 6/15/17
{
    for (int n = 0; n < npipes; n++) {
        FLT out[] = { 0.0, 0.0 };
        BIGINT j = i1[n];
        if (i1[n] < 0) {                               // wraps at left
            j += N1;
            for (int dx = 0; dx < -i1[n]; ++dx) {
                out[0] += du[2 * j] * ker[n][dx];
                out[1] += du[2 * j + 1] * ker[n][dx];
                ++j;
            }
            j -= N1;
            for (int dx = -i1[n]; dx < ns; ++dx) {
                out[0] += du[2 * j] * ker[n][dx];
                out[1] += du[2 * j + 1] * ker[n][dx];
                ++j;
            }
        }
        else if (i1[n] + ns >= N1) {                    // wraps at right
            for (int dx = 0; dx < N1 - i1[n]; ++dx) {
                out[0] += du[2 * j] * ker[n][dx];
                out[1] += du[2 * j + 1] * ker[n][dx];
                ++j;
            }
            j -= N1;
            for (int dx = N1 - i1[n]; dx < ns; ++dx) {
                out[0] += du[2 * j] * ker[n][dx];
                out[1] += du[2 * j + 1] * ker[n][dx];
                ++j;
            }
        }
        else {                                     // doesn't wrap
            for (int dx = 0; dx < ns; ++dx) {
                out[0] += du[2 * j] * ker[n][dx];
                out[1] += du[2 * j + 1] * ker[n][dx];
                ++j;
            }
        }
        target[2 * n + 0] = out[0];
        target[2 * n + 1] = out[1];
    }
}

template<int ns>
void tiled_interp_cube(FLT* target, FLT* du, FLT** ker1, FLT** ker2, FLT** ker3,
    BIGINT* i1, BIGINT* i2, BIGINT* i3, BIGINT N1, BIGINT N2, BIGINT N3)
    // 3D interpolate complex values from du (uniform grid data) array to out value,
    // using ns*ns*ns cube of real weights
    // in ker. out must be size 2 (real,imag), and du
    // of size 2*N1*N2*N3 (alternating real,imag). i1 is the left-most index in
    // [0,N1), i2 the bottom index in [0,N2), i3 lowest in [0,N3).
    // Periodic wrapping in the du array is applied, assuming N1,N2,N3>=ns.
    // dx,dy,dz indices into ker array, j index in complex du array.
    // Barnett 6/16/17
{
    // no wrapping: avoid ptrs
    for (int n = 0; n < npipes; n++) {
        FLT out[] = { 0.0, 0.0 };
        if (i1[n] >= 0 && i1[n] + ns <= N1 && i2[n] >= 0 && i2[n] + ns <= N2 && i3[n] >= 0 && i3[n] + ns <= N3) {
            for (int dz = 0; dz < ns; dz++) {
                BIGINT oz = N1 * N2 * (i3[n] + dz);        // offset due to z
                for (int dy = 0; dy < ns; dy++) {
                    BIGINT j = oz + N1 * (i2[n] + dy) + i1[n];
                    FLT ker23 = ker2[n][dy] * ker3[n][dz];
                    for (int dx = 0; dx < ns; dx++) {
                        FLT k = ker1[n][dx] * ker23;
                        out[0] += du[2 * j] * k;
                        out[1] += du[2 * j + 1] * k;
                        ++j;
                    }
                }
            }
        }
        else {                         // wraps somewhere: use ptr list (slower)
            BIGINT j1[MAX_NSPREAD], j2[MAX_NSPREAD], j3[MAX_NSPREAD];   // 1d ptr lists
            BIGINT x = i1[n], y = i2[n], z = i3[n];         // initialize coords
            for (int d = 0; d < ns; d++) {          // set up ptr lists
                if (x < 0) x += N1;
                if (x >= N1) x -= N1;
                j1[d] = x++;
                if (y < 0) y += N2;
                if (y >= N2) y -= N2;
                j2[d] = y++;
                if (z < 0) z += N3;
                if (z >= N3) z -= N3;
                j3[d] = z++;
            }
            for (int dz = 0; dz < ns; dz++) {             // use the pts lists
                BIGINT oz = N1 * N2 * j3[dz];               // offset due to z
                for (int dy = 0; dy < ns; dy++) {
                    BIGINT oy = oz + N1 * j2[dy];           // offset due to y & z
                    FLT ker23 = ker2[n][dy] * ker3[n][dz];
                    for (int dx = 0; dx < ns; dx++) {
                        FLT k = ker1[n][dx] * ker23;
                        BIGINT j = oy + j1[dx];
                        out[0] += du[2 * j] * k;
                        out[1] += du[2 * j + 1] * k;
                    }
                }
            }
        }
        target[2 * n + 0] = out[0];
        target[2 * n + 1] = out[1];
    }
#ifdef _DEBUG
    for (int n = 0; n < npipes; n++) {
        FLT test[] = { 0.0, 0.0 };
        interp_cube(test, du, ker1[n], ker2[n], ker3[n], i1[n], i2[n], i3[n], N1, N2, N3, ns);
        assert((test[0] == 0.0) ? (target[2 * n + 0] == 0.0) : (((target[2 * n + 0] - test[0]) / test[0]) < 1e-5));
        assert((test[1] == 0.0) ? (target[2 * n + 1] == 0.0) : (((target[2 * n + 1] - test[1]) / test[1]) < 1e-5));
    }
#endif
}
#ifdef __AVX512F__
#ifdef SINGLE
template<>
void tiled_interp_line<7>(FLT* target, FLT* du, FLT** ker, BIGINT* i1, BIGINT N1)
// 1D interpolate complex values from du array to out, using real weights
// ker[0] through ker[ns-1]. out must be size 2 (real,imag), and du
// of size 2*N1 (alternating real,imag). i1 is the left-most index in [0,N1)
// Periodic wrapping in the du array is applied, assuming N1>=ns.
// dx is index into ker array, j index in complex du (data_uniform) array.
// Barnett 6/15/17
{
    constexpr BIGINT ns = 7;

    __m512i _zero = _mm512_setzero_si512();
    __m512i _ns = _mm512_set1_epi64(ns);
    __m512i _N1 = _mm512_set1_epi64(N1);
    __m512i _i1 = _mm512_load_epi64(i1);

    __mmask8 _wrap = _kor_mask8(
        _mm512_cmp_epi64_mask(_i1, _zero, _MM_CMPINT_LT),
        _mm512_cmp_epi64_mask(_N1,
            _mm512_add_epi64(_i1, _ns),
            _MM_CMPINT_LT));

    if (_wrap == 0)
    {
        const __mmask16 _ns_mask = 0x007f;
        const __mmask16 _n2_mask = 0x3fff;
        const __m512i _perm0 = _mm512_set_epi32(7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0);

        __m512 _ker1a = _mm512_permutexvar_ps(_perm0, _mm512_maskz_loadu_ps(_ns_mask, ker[0]));
        __m512 _acca = _mm512_mul_ps(_ker1a, _mm512_maskz_loadu_ps(_n2_mask, du + 2 * i1[0]));

        __m512 _ker1b = _mm512_permutexvar_ps(_perm0, _mm512_maskz_loadu_ps(_ns_mask, ker[1]));
        __m512 _accb = _mm512_mul_ps(_ker1b, _mm512_maskz_loadu_ps(_n2_mask, du + 2 * i1[1]));

        __m512 _ker1c = _mm512_permutexvar_ps(_perm0, _mm512_maskz_loadu_ps(_ns_mask, ker[2]));
        __m512 _accc = _mm512_mul_ps(_ker1c, _mm512_maskz_loadu_ps(_n2_mask, du + 2 * i1[2]));

        __m512 _ker1d = _mm512_permutexvar_ps(_perm0, _mm512_maskz_loadu_ps(_ns_mask, ker[3]));
        __m512 _accd = _mm512_mul_ps(_ker1d, _mm512_maskz_loadu_ps(_n2_mask, du + 2 * i1[3]));

        __m512 _ker1e = _mm512_permutexvar_ps(_perm0, _mm512_maskz_loadu_ps(_ns_mask, ker[4]));
        __m512 _acce = _mm512_mul_ps(_ker1e, _mm512_maskz_loadu_ps(_n2_mask, du + 2 * i1[4]));

        __m512 _ker1f = _mm512_permutexvar_ps(_perm0, _mm512_maskz_loadu_ps(_ns_mask, ker[5]));
        __m512 _accf = _mm512_mul_ps(_ker1f, _mm512_maskz_loadu_ps(_n2_mask, du + 2 * i1[5]));

        __m512 _ker1g = _mm512_permutexvar_ps(_perm0, _mm512_maskz_loadu_ps(_ns_mask, ker[6]));
        __m512 _accg = _mm512_mul_ps(_ker1g, _mm512_maskz_loadu_ps(_n2_mask, du + 2 * i1[6]));

        __m512 _acch = _mm512_setzero_ps();

        // Transpose
        __m512 _t0 = _mm512_shuffle_ps(_acca, _accb, 0x44);
        __m512 _t1 = _mm512_shuffle_ps(_acca, _accb, 0xee);
        __m512 _t2 = _mm512_shuffle_ps(_accc, _accd, 0x44);
        __m512 _t3 = _mm512_shuffle_ps(_accc, _accd, 0xee);
        __m512 _t4 = _mm512_shuffle_ps(_acce, _accf, 0x44);
        __m512 _t5 = _mm512_shuffle_ps(_acce, _accf, 0xee);
        __m512 _t6 = _mm512_shuffle_ps(_accg, _acch, 0x44);
        __m512 _t7 = _mm512_shuffle_ps(_accg, _acch, 0xee);

        __m512 _tt0 = _mm512_shuffle_f32x4(_t0, _t2, 0x88);
        __m512 _tt1 = _mm512_shuffle_f32x4(_t1, _t3, 0x88);
        __m512 _tt2 = _mm512_shuffle_f32x4(_t4, _t6, 0x88);
        __m512 _tt3 = _mm512_shuffle_f32x4(_t5, _t7, 0x88);
        __m512 _tt4 = _mm512_shuffle_f32x4(_t0, _t2, 0xdd);
        __m512 _tt5 = _mm512_shuffle_f32x4(_t1, _t3, 0xdd);
        __m512 _tt6 = _mm512_shuffle_f32x4(_t4, _t6, 0xdd);
        __m512 _tt7 = _mm512_shuffle_f32x4(_t5, _t7, 0xdd);

        __m512 _outa = _mm512_shuffle_f32x4(_tt0, _tt2, 0x88);
        __m512 _outb = _mm512_shuffle_f32x4(_tt1, _tt3, 0x88);
        __m512 _outc = _mm512_shuffle_f32x4(_tt4, _tt6, 0x88);
        __m512 _outd = _mm512_shuffle_f32x4(_tt5, _tt7, 0x88);
        __m512 _oute = _mm512_shuffle_f32x4(_tt0, _tt2, 0xdd);
        __m512 _outf = _mm512_shuffle_f32x4(_tt1, _tt3, 0xdd);
        __m512 _outg = _mm512_shuffle_f32x4(_tt4, _tt6, 0xdd);
        __m512 _outh = _mm512_shuffle_f32x4(_tt5, _tt7, 0xdd);

        // Reduce
        __m512 _out = _mm512_add_ps(
            _mm512_add_ps(
                _mm512_add_ps(_outa, _outb),
                _mm512_add_ps(_outc, _outd)),
            _mm512_add_ps(
                _mm512_add_ps(_oute, _outf),
                _mm512_add_ps(_outg, _outh)));

        // Store
        _mm512_storeu_ps(target, _out);

#ifdef _DEBUG
        for (int n = 0; n < npipes; n++) {
            FLT test[] = { 0.0, 0.0 };
            interp_line(test, du, ker[n], i1[n], N1, ns);
            assert((test[0] == 0.0) ? (target[2 * n + 0] == 0.0) : (((target[2 * n + 0] - test[0]) / test[0]) < 1e-2));
            assert((test[1] == 0.0) ? (target[2 * n + 1] == 0.0) : (((target[2 * n + 1] - test[1]) / test[1]) < 1e-2));
    }
#endif

    }
    else {
        for (int n = 0; n < npipes; n++) {
            FLT out[] = { 0.0, 0.0 };
            BIGINT j = i1[n];
            if (i1[n] < 0) {                               // wraps at left
                j += N1;
                for (int dx = 0; dx < -i1[n]; ++dx) {
                    out[0] += du[2 * j] * ker[n][dx];
                    out[1] += du[2 * j + 1] * ker[n][dx];
                    ++j;
                }
                j -= N1;
                for (int dx = -i1[n]; dx < ns; ++dx) {
                    out[0] += du[2 * j] * ker[n][dx];
                    out[1] += du[2 * j + 1] * ker[n][dx];
                    ++j;
                }
            }
            else if (i1[n] + ns >= N1) {                    // wraps at right
                for (int dx = 0; dx < N1 - i1[n]; ++dx) {
                    out[0] += du[2 * j] * ker[n][dx];
                    out[1] += du[2 * j + 1] * ker[n][dx];
                    ++j;
                }
                j -= N1;
                for (int dx = N1 - i1[n]; dx < ns; ++dx) {
                    out[0] += du[2 * j] * ker[n][dx];
                    out[1] += du[2 * j + 1] * ker[n][dx];
                    ++j;
                }
            }
            else {                                     // doesn't wrap
                for (int dx = 0; dx < ns; ++dx) {
                    out[0] += du[2 * j] * ker[n][dx];
                    out[1] += du[2 * j + 1] * ker[n][dx];
                    ++j;
                }
            }
            target[2 * n + 0] = out[0];
            target[2 * n + 1] = out[1];
        }
    }
}

template<>
void tiled_interp_cube<7>(float* target, float* du, float** ker1, float** ker2, float** ker3,
    BIGINT* i1, BIGINT* i2, BIGINT* i3, BIGINT N1, BIGINT N2, BIGINT N3)
    // 3D interpolate complex values from du (uniform grid data) array to out value,
    // using ns*ns*ns cube of real weights
    // in ker. out must be size 2 (real,imag), and du
    // of size 2*N1*N2*N3 (alternating real,imag). i1 is the left-most index in
    // [0,N1), i2 the bottom index in [0,N2), i3 lowest in [0,N3).
    // Periodic wrapping in the du array is applied, assuming N1,N2,N3>=ns.
    // dx,dy,dz indices into ker array, j index in complex du array.
    // Barnett 6/16/17
{
    constexpr BIGINT ns = 7;
    const __mmask16 _ns_mask = 0x007f;
    const __mmask16 _n2_mask = 0x3fff;
    const __m512i _perm0 = _mm512_set_epi32(7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0);
    // no wrapping: avoid ptrs
    for (int n = 0; n < npipes; n++) {
        if (i1[n] >= 0 && i1[n] + ns <= N1 && i2[n] >= 0 && i2[n] + ns <= N2 && i3[n] >= 0 && i3[n] + ns <= N3) {
            const __m512 _ker1x = _mm512_maskz_load_ps(_ns_mask, ker1[n]);
            const __m512 _ker1 = _mm512_permutexvar_ps(_perm0, _ker1x);
            __m512 _acc0a = _mm512_setzero_ps();
            __m512 _acc0b = _mm512_setzero_ps();
            __m512 _acc0c = _mm512_setzero_ps();
            __m512 _acc0d = _mm512_setzero_ps();
            __m512 _acc0e = _mm512_setzero_ps();
            __m512 _acc0f = _mm512_setzero_ps();
            __m512 _acc0g = _mm512_setzero_ps();
            for (int dz = 0; dz < ns; dz++) {
                const BIGINT oz = N1 * N2 * (i3[n] + dz);        // offset due to z
                const BIGINT oy = oz + N1 * i2[n] + i1[n];

                float* pdua = du + 2 * oy;
                __m512 _ker23a = _mm512_set1_ps(ker2[n][0] * ker3[n][dz]);
                __m512 _k0a = _mm512_mul_ps(_ker1, _ker23a);
                _acc0a = _mm512_fmadd_ps(_k0a, _mm512_maskz_loadu_ps(_n2_mask, pdua), _acc0a);

                //{
                //    float lng[2 * MAX_NSPREAD];
                //    BIGINT j = oz + N1 * (i2[n] + 0) + i1[n];
                //    float ker23 = ker2[n][0] * ker3[n][dz];
                //    for (int dx = 0; dx < ns; dx++) {
                //        FLT k = ker1[n][dx] * ker23;
                //        lng[2 * dx] = du[2 * j] * k;
                //        lng[2 * dx + 1] = du[2 * j + 1] * k;
                //        ++j;
                //    }
                //    for (int di = 0; di < 2 * ns; di++) {
                //        assert((lng[di] == 0.0) ? (_acc0a.m512_f32[di] == 0.0) : (((_acc0a.m512_f32[di] - lng[di]) / lng[di]) < 1));
                //    }
                //}

                    
                float* pdub = pdua + 2 * N1;
                __m512 _ker23b = _mm512_set1_ps(ker2[n][1] * ker3[n][dz]);
                __m512 _k0b = _mm512_mul_ps(_ker1, _ker23b);
                _acc0b = _mm512_fmadd_ps(_k0b, _mm512_maskz_loadu_ps(_n2_mask, pdub), _acc0b);

                float* pduc = pdub + 2 * N1;
                __m512 _ker23c = _mm512_set1_ps(ker2[n][2] * ker3[n][dz]);
                __m512 _k0c = _mm512_mul_ps(_ker1, _ker23c);
                _acc0c = _mm512_fmadd_ps(_k0c, _mm512_maskz_loadu_ps(_n2_mask, pduc), _acc0c);

                float* pdud = pduc + 2 * N1;
                __m512 _ker23d = _mm512_set1_ps(ker2[n][3] * ker3[n][dz]);
                __m512 _k0d = _mm512_mul_ps(_ker1, _ker23d);
                _acc0d = _mm512_fmadd_ps(_k0d, _mm512_maskz_loadu_ps(_n2_mask, pdud), _acc0d);

                float* pdue = pdud + 2 * N1;
                __m512 _ker23e = _mm512_set1_ps(ker2[n][4] * ker3[n][dz]);
                __m512 _k0e = _mm512_mul_ps(_ker1, _ker23e);
                _acc0e = _mm512_fmadd_ps(_k0e, _mm512_maskz_loadu_ps(_n2_mask, pdue), _acc0e);

                float* pduf = pdue + 2 * N1;
                __m512 _ker23f = _mm512_set1_ps(ker2[n][5] * ker3[n][dz]);
                __m512 _k0f = _mm512_mul_ps(_ker1, _ker23f);
                _acc0f = _mm512_fmadd_ps(_k0f, _mm512_maskz_loadu_ps(_n2_mask, pduf), _acc0f);

                float* pdug = pduf + 2 * N1;
                __m512 _ker23g = _mm512_set1_ps(ker2[n][6] * ker3[n][dz]);
                __m512 _k0g = _mm512_mul_ps(_ker1, _ker23g);
                _acc0g = _mm512_fmadd_ps(_k0g, _mm512_maskz_loadu_ps(_n2_mask, pdug), _acc0g);
            }

            __m512 _acc = _mm512_add_ps(
                _mm512_add_ps(
                    _mm512_add_ps(_acc0a, _acc0b),
                    _mm512_add_ps(_acc0c, _acc0d)),
                _mm512_add_ps(
                    _mm512_add_ps(_acc0e, _acc0f),
                    _acc0g));

            _acc = _mm512_add_ps(_acc, _mm512_permute_ps(_acc, 0x8e));
            __m128 _out = _mm_add_ps(
                _mm_add_ps(
                    _mm512_castps512_ps128(_acc),
                    _mm512_extractf32x4_ps(_acc, 1)),
                _mm_add_ps(
                    _mm512_extractf32x4_ps(_acc, 2),
                    _mm512_extractf32x4_ps(_acc, 3)));

            _mm_mask_storeu_ps(target + 2 * n, 0x3, _out);
#ifdef _DEBUG
            FLT test[] = { 0.0, 0.0 };
            interp_cube(test, du, ker1[n], ker2[n], ker3[n], i1[n], i2[n], i3[n], N1, N2, N3, ns);
            assert((test[0] == 0.0) ? (target[2 * n + 0] == 0.0) : (((target[2 * n + 0] - test[0]) / test[0]) < 1.e-2));
        //        assert((test[1] == 0.0) ? (target[2 * n + 1] == 0.0) : (((target[2 * n + 1] - test[1]) / test[1]) < 1));
#endif
        }
        else {                         // wraps somewhere: use ptr list (slower)
            FLT out[] = { 0.0, 0.0 };
            BIGINT j1[MAX_NSPREAD], j2[MAX_NSPREAD], j3[MAX_NSPREAD];   // 1d ptr lists
            BIGINT x = i1[n], y = i2[n], z = i3[n];         // initialize coords
            for (int d = 0; d < ns; d++) {          // set up ptr lists
                if (x < 0) x += N1;
                if (x >= N1) x -= N1;
                j1[d] = x++;
                if (y < 0) y += N2;
                if (y >= N2) y -= N2;
                j2[d] = y++;
                if (z < 0) z += N3;
                if (z >= N3) z -= N3;
                j3[d] = z++;
            }
            for (int dz = 0; dz < ns; dz++) {             // use the pts lists
                BIGINT oz = N1 * N2 * j3[dz];               // offset due to z
                for (int dy = 0; dy < ns; dy++) {
                    BIGINT oy = oz + N1 * j2[dy];           // offset due to y & z
                    FLT ker23 = ker2[n][dy] * ker3[n][dz];
                    for (int dx = 0; dx < ns; dx++) {
                        FLT k = ker1[n][dx] * ker23;
                        BIGINT j = oy + j1[dx];
                        out[0] += du[2 * j] * k;
                        out[1] += du[2 * j + 1] * k;
                    }
                }
            }

            target[2 * n + 0] = out[0];
            target[2 * n + 1] = out[1];
#ifdef _DEBUG
            FLT test[] = { 0.0, 0.0 };
            interp_cube(test, du, ker1[n], ker2[n], ker3[n], i1[n], i2[n], i3[n], N1, N2, N3, ns);
            assert((test[0] == 0.0) ? (target[2 * n + 0] == 0.0) : (((target[2 * n + 0] - test[0]) / test[0]) < 1));
        //        assert((test[1] == 0.0) ? (target[2 * n + 1] == 0.0) : (((target[2 * n + 1] - test[1]) / test[1]) < 1));
#endif
        }
    }
}
#else
template<>
void tiled_interp_cube<7>(double* target, double* du, double** ker1, double** ker2, double** ker3,
    BIGINT* i1, BIGINT* i2, BIGINT* i3, BIGINT N1, BIGINT N2, BIGINT N3)
{
    constexpr BIGINT ns = 7;
    const __m512i _perm0 = _mm512_set_epi64(7, 7, 2, 2, 1, 1, 0, 0);
    const __m512i _perm1 = _mm512_set_epi64(6, 6, 5, 5, 4, 4, 3, 3);
    // no wrapping: avoid ptrs
    for (int n = 0; n < npipes; n++) {
        if (i1[n] >= 0 && i1[n] + ns <= N1 && i2[n] >= 0 && i2[n] + ns <= N2 && i3[n] >= 0 && i3[n] + ns <= N3) {
            const __m512d _ker1x = _mm512_maskz_loadu_pd(0x7f, ker1[n]);
            const __m512d _ker0 = _mm512_permutexvar_pd(_perm0, _ker1x );
            const __m512d _ker1 = _mm512_permutexvar_pd(_perm1, _ker1x);
            __m512d _acc0a = _mm512_setzero_pd();
            __m512d _acc0b = _mm512_setzero_pd();
            __m512d _acc0c = _mm512_setzero_pd();
            __m512d _acc0d = _mm512_setzero_pd();
            __m512d _acc0e = _mm512_setzero_pd();
            __m512d _acc0f = _mm512_setzero_pd();
            __m512d _acc0g = _mm512_setzero_pd();
            __m512d _acc1a = _mm512_setzero_pd();
            __m512d _acc1b = _mm512_setzero_pd();
            __m512d _acc1c = _mm512_setzero_pd();
            __m512d _acc1d = _mm512_setzero_pd();
            __m512d _acc1e = _mm512_setzero_pd();
            __m512d _acc1f = _mm512_setzero_pd();
            __m512d _acc1g = _mm512_setzero_pd();
            for (int dz = 0; dz < ns; dz++) {
                const BIGINT oz = N1 * N2 * (i3[n] + dz);        // offset due to z
                const BIGINT oy = oz + N1 * i2[n] + i1[n];

                double* pdua = du + 2 * oy;
                __m512d _ker23a = _mm512_set1_pd(ker2[n][0] * ker3[n][dz]);
                __m512d _k0a = _mm512_mul_pd(_ker0, _ker23a);
                __m512d _k1a = _mm512_mul_pd(_ker1, _ker23a);
                _acc0a = _mm512_fmadd_pd(_k0a, _mm512_loadu_pd(pdua + 0), _acc0a);
                _acc1a = _mm512_fmadd_pd(_k1a, _mm512_loadu_pd(pdua + 6), _acc1a);

                double* pdub = pdua + 2 * N1;
                __m512d _ker23b = _mm512_set1_pd(ker2[n][1] * ker3[n][dz]);
                __m512d _k0b = _mm512_mul_pd(_ker0, _ker23b);
                __m512d _k1b = _mm512_mul_pd(_ker1, _ker23b);
                _acc0b = _mm512_fmadd_pd(_k0b, _mm512_loadu_pd(pdub + 0), _acc0b);
                _acc1b = _mm512_fmadd_pd(_k1b, _mm512_loadu_pd(pdub + 6), _acc1b);

                double* pduc = pdub + 2 * N1;
                __m512d _ker23c = _mm512_set1_pd(ker2[n][2] * ker3[n][dz]);
                __m512d _k0c = _mm512_mul_pd(_ker0, _ker23c);
                __m512d _k1c = _mm512_mul_pd(_ker1, _ker23c);
                _acc0c = _mm512_fmadd_pd(_k0c, _mm512_loadu_pd(pduc + 0), _acc0c);
                _acc1c = _mm512_fmadd_pd(_k1c, _mm512_loadu_pd(pduc + 6), _acc1c);

                double* pdud = pduc + 2 * N1;
                __m512d _ker23d = _mm512_set1_pd(ker2[n][3] * ker3[n][dz]);
                __m512d _k0d = _mm512_mul_pd(_ker0, _ker23d);
                __m512d _k1d = _mm512_mul_pd(_ker1, _ker23d);
                _acc0d = _mm512_fmadd_pd(_k0d, _mm512_loadu_pd(pdud + 0), _acc0d);
                _acc1d = _mm512_fmadd_pd(_k1d, _mm512_loadu_pd(pdud + 6), _acc1d);

                double* pdue = pdud + 2 * N1;
                __m512d _ker23e = _mm512_set1_pd(ker2[n][4] * ker3[n][dz]);
                __m512d _k0e = _mm512_mul_pd(_ker0, _ker23e);
                __m512d _k1e = _mm512_mul_pd(_ker1, _ker23e);
                _acc0e = _mm512_fmadd_pd(_k0e, _mm512_loadu_pd(pdue + 0), _acc0e);
                _acc1e = _mm512_fmadd_pd(_k1e, _mm512_loadu_pd(pdue + 6), _acc1e);

                double* pduf = pdue + 2 * N1;
                __m512d _ker23f = _mm512_set1_pd(ker2[n][5] * ker3[n][dz]);
                __m512d _k0f = _mm512_mul_pd(_ker0, _ker23f);
                __m512d _k1f = _mm512_mul_pd(_ker1, _ker23f);
                _acc0f = _mm512_fmadd_pd(_k0f, _mm512_loadu_pd(pduf + 0), _acc0f);
                _acc1f = _mm512_fmadd_pd(_k1f, _mm512_loadu_pd(pduf + 6), _acc1f);

                double* pdug = pduf + 2 * N1;
                __m512d _ker23g = _mm512_set1_pd(ker2[n][6] * ker3[n][dz]);
                __m512d _k0g = _mm512_mul_pd(_ker0, _ker23g);
                __m512d _k1g = _mm512_mul_pd(_ker1, _ker23g);
                _acc0g = _mm512_fmadd_pd(_k0g, _mm512_loadu_pd(pdug + 0), _acc0g);
                _acc1g = _mm512_fmadd_pd(_k1g, _mm512_loadu_pd(pdug + 6), _acc1g);
            }

            __m512d _acc = _mm512_add_pd(
                _mm512_add_pd(
                    _mm512_add_pd(
                        _mm512_add_pd(_acc0a, _acc1a),
                        _mm512_add_pd(_acc0b, _acc1b)),
                    _mm512_add_pd(
                        _mm512_add_pd(_acc0c, _acc1c),
                        _mm512_add_pd(_acc0d, _acc1d))),
                _mm512_add_pd(
                    _mm512_add_pd(
                        _mm512_add_pd(_acc0e, _acc1e),
                        _mm512_add_pd(_acc0f, _acc1f)),
                    _mm512_add_pd(_acc0g, _acc1g)));

            __m128d _out = _mm_add_pd(
                _mm_add_pd(
                    _mm512_castpd512_pd128(_acc),
                    _mm512_extractf64x2_pd(_acc, 1)),
                _mm_add_pd(
                    _mm512_extractf64x2_pd(_acc, 2),
                    _mm512_extractf64x2_pd(_acc, 3)));

            _mm_storeu_pd(target + 2 * n, _out);
//#ifdef _DEBUG
//    {
//        FLT test[] = { 0.0, 0.0 };
//        interp_cube(test, du, ker1[n], ker2[n], ker3[n], i1[n], i2[n], i3[n], N1, N2, N3, ns);
//        assert((test[0] == 0.0) ? (target[2 * n + 0] == 0.0) : (((target[2 * n + 0] - test[0]) / test[0]) < 1e-2));
//        assert((test[1] == 0.0) ? (target[2 * n + 1] == 0.0) : (((target[2 * n + 1] - test[1]) / test[1]) < 1e-2));
//    }
//#endif

        }
        else {                         // wraps somewhere: use ptr list (slower)
            double out[] = { 0.0, 0.0 };
            BIGINT j1[MAX_NSPREAD], j2[MAX_NSPREAD], j3[MAX_NSPREAD];   // 1d ptr lists
            BIGINT x = i1[n], y = i2[n], z = i3[n];         // initialize coords
            for (int d = 0; d < ns; d++) {          // set up ptr lists
                if (x < 0) x += N1;
                if (x >= N1) x -= N1;
                j1[d] = x++;
                if (y < 0) y += N2;
                if (y >= N2) y -= N2;
                j2[d] = y++;
                if (z < 0) z += N3;
                if (z >= N3) z -= N3;
                j3[d] = z++;
            }
            for (int dz = 0; dz < ns; dz++) {             // use the pts lists
                BIGINT oz = N1 * N2 * j3[dz];               // offset due to z
                for (int dy = 0; dy < ns; dy++) {
                    BIGINT oy = oz + N1 * j2[dy];           // offset due to y & z
                    FLT ker23 = ker2[n][dy] * ker3[n][dz];
                    for (int dx = 0; dx < ns; dx++) {
                        FLT k = ker1[n][dx] * ker23;
                        BIGINT j = oy + j1[dx];
                        out[0] += du[2 * j] * k;
                        out[1] += du[2 * j + 1] * k;
                    }
                }
            }
            target[2 * n + 0] = out[0];
            target[2 * n + 1] = out[1];
        }
    }
}
#endif
#else
#ifdef __AVX2__
#ifdef SINGLE
template<>
void tiled_interp_cube<7>(float* target, float* du, float** ker1, float** ker2, float** ker3,
    BIGINT* i1, BIGINT* i2, BIGINT* i3, BIGINT N1, BIGINT N2, BIGINT N3)
    // 3D interpolate complex values from du (uniform grid data) array to out value,
    // using ns*ns*ns cube of real weights
    // in ker. out must be size 2 (real,imag), and du
    // of size 2*N1*N2*N3 (alternating real,imag). i1 is the left-most index in
    // [0,N1), i2 the bottom index in [0,N2), i3 lowest in [0,N3).
    // Periodic wrapping in the du array is applied, assuming N1,N2,N3>=ns.
    // dx,dy,dz indices into ker array, j index in complex du array.
    // Barnett 6/16/17
{
    constexpr BIGINT ns = 7;
    const __m256i _ns_mask = _mm256_set_epi32(0, -1, -1, -1, -1, -1, -1, -1);
    const __m256i _perm0 = _mm256_set_epi32(7, 7, 2, 2, 1, 1, 0, 0);
    const __m256i _perm1 = _mm256_set_epi32(6, 6, 5, 5, 4, 4, 3, 3);
    // no wrapping: avoid ptrs
    for (int n = 0; n < npipes; n++) {
        if (i1[n] >= 0 && i1[n] + ns <= N1 && i2[n] >= 0 && i2[n] + ns <= N2 && i3[n] >= 0 && i3[n] + ns <= N3) {
            const __m256 _ker1x = _mm256_maskload_ps(ker1[n], _ns_mask);
            const __m256 _ker0 = _mm256_permutevar8x32_ps(_ker1x, _perm0);
            const __m256 _ker1 = _mm256_permutevar8x32_ps(_ker1x, _perm1);
            __m256 _acc0a = _mm256_setzero_ps();
            __m256 _acc1a = _mm256_setzero_ps();
            __m256 _acc0b = _mm256_setzero_ps();
            __m256 _acc1b = _mm256_setzero_ps();
            __m256 _ker23a, _ker23b;
            __m256 _k0a, _k1a, _k0b,_k1b;
            float *pdua, *pdub;
            for (int dz = 0; dz < ns; dz++) {
                const BIGINT oz = N1 * N2 * (i3[n] + dz);        // offset due to z
                const BIGINT oy = oz + N1 * i2[n] + i1[n];

                pdub = du + 2 * oy;
                _ker23b = _mm256_set1_ps(ker2[n][0] * ker3[n][dz]);
                _k0b = _mm256_mul_ps(_ker0, _ker23b);
                _k1b = _mm256_mul_ps(_ker1, _ker23b);
                _acc0b = _mm256_fmadd_ps(_k0b, _mm256_loadu_ps(pdub + 0), _acc0b);
                _acc1b = _mm256_fmadd_ps(_k1b, _mm256_loadu_ps(pdub + 6), _acc1b);

                pdua = pdub + 2 * N1;
                _ker23a = _mm256_set1_ps(ker2[n][1] * ker3[n][dz]);
                _k0a = _mm256_mul_ps(_ker0, _ker23a);
                _k1a = _mm256_mul_ps(_ker1, _ker23a);
                _acc0a = _mm256_fmadd_ps(_k0a, _mm256_loadu_ps(pdua + 0), _acc0a);
                _acc1a = _mm256_fmadd_ps(_k1a, _mm256_loadu_ps(pdua + 6), _acc1a);

                pdub = pdua + 2 * N1;
                _ker23b = _mm256_set1_ps(ker2[n][2] * ker3[n][dz]);
                _k0b = _mm256_mul_ps(_ker0, _ker23b);
                _k1b = _mm256_mul_ps(_ker1, _ker23b);
                _acc0b = _mm256_fmadd_ps(_k0b, _mm256_loadu_ps(pdub + 0), _acc0b);
                _acc1b = _mm256_fmadd_ps(_k1b, _mm256_loadu_ps(pdub + 6), _acc1b);

                pdua = pdub + 2 * N1;
                _ker23a = _mm256_set1_ps(ker2[n][3] * ker3[n][dz]);
                _k0a = _mm256_mul_ps(_ker0, _ker23a);
                _k1a = _mm256_mul_ps(_ker1, _ker23a);
                _acc0a = _mm256_fmadd_ps(_k0a, _mm256_loadu_ps(pdua + 0), _acc0a);
                _acc1a = _mm256_fmadd_ps(_k1a, _mm256_loadu_ps(pdua + 6), _acc1a);

                pdub = pdua + 2 * N1;
                _ker23b = _mm256_set1_ps(ker2[n][4] * ker3[n][dz]);
                _k0b = _mm256_mul_ps(_ker0, _ker23b);
                _k1b = _mm256_mul_ps(_ker1, _ker23b);
                _acc0b = _mm256_fmadd_ps(_k0b, _mm256_loadu_ps(pdub + 0), _acc0b);
                _acc1b = _mm256_fmadd_ps(_k1b, _mm256_loadu_ps(pdub + 6), _acc1b);

                pdua = pdub + 2 * N1;
                _ker23a = _mm256_set1_ps(ker2[n][5] * ker3[n][dz]);
                _k0a = _mm256_mul_ps(_ker0, _ker23a);
                _k1a = _mm256_mul_ps(_ker1, _ker23a);
                _acc0a = _mm256_fmadd_ps(_k0a, _mm256_loadu_ps(pdua + 0), _acc0a);
                _acc1a = _mm256_fmadd_ps(_k1a, _mm256_loadu_ps(pdua + 6), _acc1a);

                pdub = pdua + 2 * N1;
                _ker23b = _mm256_set1_ps(ker2[n][6] * ker3[n][dz]);
                _k0b = _mm256_mul_ps(_ker0, _ker23b);
                _k1b = _mm256_mul_ps(_ker1, _ker23b);
                _acc0b = _mm256_fmadd_ps(_k0b, _mm256_loadu_ps(pdub + 0), _acc0b);
                _acc1b = _mm256_fmadd_ps(_k1b, _mm256_loadu_ps(pdub + 6), _acc1b);
            }

            __m256 _acc = _mm256_add_ps(
                _mm256_add_ps(_acc0a, _acc1a),
                _mm256_add_ps(_acc0b, _acc1b));
            _acc = _mm256_add_ps(_acc, _mm256_permute_ps(_acc, 0x8e));
            __m128 _out = _mm_add_ps(
                _mm256_castps256_ps128(_acc),
                _mm256_extractf128_ps(_acc, 1));

            _mm_maskstore_ps(target + 2 * n, _mm_set_epi32(0, 0, -1, -1), _out);
        }
        else {                         // wraps somewhere: use ptr list (slower)
            FLT out[] = { 0.0, 0.0 };
            BIGINT j1[MAX_NSPREAD], j2[MAX_NSPREAD], j3[MAX_NSPREAD];   // 1d ptr lists
            BIGINT x = i1[n], y = i2[n], z = i3[n];         // initialize coords
            for (int d = 0; d < ns; d++) {          // set up ptr lists
                if (x < 0) x += N1;
                if (x >= N1) x -= N1;
                j1[d] = x++;
                if (y < 0) y += N2;
                if (y >= N2) y -= N2;
                j2[d] = y++;
                if (z < 0) z += N3;
                if (z >= N3) z -= N3;
                j3[d] = z++;
            }
            for (int dz = 0; dz < ns; dz++) {             // use the pts lists
                BIGINT oz = N1 * N2 * j3[dz];               // offset due to z
                for (int dy = 0; dy < ns; dy++) {
                    BIGINT oy = oz + N1 * j2[dy];           // offset due to y & z
                    FLT ker23 = ker2[n][dy] * ker3[n][dz];
                    for (int dx = 0; dx < ns; dx++) {
                        FLT k = ker1[n][dx] * ker23;
                        BIGINT j = oy + j1[dx];
                        out[0] += du[2 * j] * k;
                        out[1] += du[2 * j + 1] * k;
                    }
                }
            }

            target[2 * n + 0] = out[0];
            target[2 * n + 1] = out[1];
        }
    }
//#ifdef _DEBUG
//    for (int n = 0; n < npipes; n++) {
//        FLT test[] = { 0.0, 0.0 };
//        interp_cube(test, du, ker1[n], ker2[n], ker3[n], i1[n], i2[n], i3[n], N1, N2, N3, ns);
//        assert((test[0] == 0.0) ? (target[2 * n + 0] == 0.0) : (((target[2 * n + 0] - test[0]) / test[0]) < 1));
//        assert((test[1] == 0.0) ? (target[2 * n + 1] == 0.0) : (((target[2 * n + 1] - test[1]) / test[1]) < 1));
//    }
//#endif
}
#else
template<>  
void tiled_interp_cube<7>(double* target, double* du, double** ker1, double** ker2, double** ker3,
    BIGINT* i1, BIGINT* i2, BIGINT* i3, BIGINT N1, BIGINT N2, BIGINT N3)
{
    constexpr BIGINT ns = 7;
    for (int n = 0; n < npipes; n++) {
        if (i1[n] >= 0 && i1[n] + ns <= N1 && i2[n] >= 0 && i2[n] + ns <= N2 && i3[n] >= 0 && i3[n] + ns <= N3) {
            FLT out[] = { 0.0, 0.0,  0.0, 0.0,  0.0, 0.0 , 0.0, 0.0,  0.0, 0.0 , 0.0, 0.0 , 0.0, 0.0 };
            __m128d _out = _mm_setzero_pd();
            __m256d _acc1 = _mm256_setzero_pd();
            __m256d _acc2 = _mm256_setzero_pd();
            __m256d _acc3 = _mm256_setzero_pd();
            for (int dz = 0; dz < ns; dz++) {
                BIGINT oz = N1 * N2 * (i3[n] + dz);        // offset due to z
                // Main column loop
                for (int dy = 0; dy < ns; dy++) {
                    BIGINT j0 = oz + N1 * (i2[n] + dy) + i1[n];
                    double* pdu = du + 2 * j0;

                    __m256d _ker23a = _mm256_set1_pd(ker2[n][dy] * ker3[n][dz]);

                    __m256d _ker01 = _mm256_load_pd(ker1[n]);
                    __m256d _k01 = _mm256_mul_pd(_ker01, _ker23a);
                    __m256d _ker23 = _mm256_loadu_pd(ker1[n] + 3);
                    __m256d _k23 = _mm256_mul_pd(_ker23, _ker23a);

                    __m128d _kk0 = _mm256_castpd256_pd128(
                        _mm256_permute4x64_pd(_k01, 0x50));
                    __m128d _du0 = _mm_loadu_pd(pdu + 0);
                    _out = _mm_fmadd_pd(_kk0, _du0, _out);

                    __m256d _kk1 = _mm256_permute4x64_pd(_k01, 0xa5);
                    __m256d _du1 = _mm256_loadu_pd(pdu + 2);
                    _acc1 = _mm256_fmadd_pd(_kk1, _du1, _acc1);

                    __m256d _kk2 = _mm256_permute4x64_pd(_k23, 0x50);
                    __m256d _du2 = _mm256_loadu_pd(pdu + 6);
                    _acc2 = _mm256_fmadd_pd(_kk2, _du2, _acc2);

                    __m256d _kk3 = _mm256_permute4x64_pd(_k23,0xfa);
                    __m256d _du3 = _mm256_loadu_pd(pdu + 10);
                    _acc3 = _mm256_fmadd_pd(_kk3, _du3, _acc3);
                }
            }

            __m256d _acc = _mm256_add_pd(_mm256_add_pd(_acc1, _acc2), _acc3);
            _out = _mm_add_pd(
                _out,
                _mm256_castpd256_pd128(_acc));
            _out = _mm_add_pd(
                _out,
                _mm256_extractf128_pd(_acc, 1));

            _mm_storeu_pd(target + 2 * n, _out);
        }
        else {                         // wraps somewhere: use ptr list (slower)
            double out[] = { 0.0, 0.0 };
            BIGINT j1[MAX_NSPREAD], j2[MAX_NSPREAD], j3[MAX_NSPREAD];   // 1d ptr lists
            BIGINT x = i1[n], y = i2[n], z = i3[n];         // initialize coords
            for (int d = 0; d < ns; d++) {          // set up ptr lists
                if (x < 0) x += N1;
                if (x >= N1) x -= N1;
                j1[d] = x++;
                if (y < 0) y += N2;
                if (y >= N2) y -= N2;
                j2[d] = y++;
                if (z < 0) z += N3;
                if (z >= N3) z -= N3;
                j3[d] = z++;
            }
            for (int dz = 0; dz < ns; dz++) {             // use the pts lists
                BIGINT oz = N1 * N2 * j3[dz];               // offset due to z
                for (int dy = 0; dy < ns; dy++) {
                    BIGINT oy = oz + N1 * j2[dy];           // offset due to y & z
                    FLT ker23 = ker2[n][dy] * ker3[n][dz];
                    for (int dx = 0; dx < ns; dx++) {
                        FLT k = ker1[n][dx] * ker23;
                        BIGINT j = oy + j1[dx];
                        out[0] += du[2 * j] * k;
                        out[1] += du[2 * j + 1] * k;
                    }
                }
            }
            target[2 * n + 0] = out[0];
            target[2 * n + 1] = out[1];
        }
#ifdef _DEBUG
    {
        FLT test[] = { 0.0, 0.0 };
        interp_cube(test, du, ker1[n], ker2[n], ker3[n], i1[n], i2[n], i3[n], N1, N2, N3, ns);
        assert((test[0] == 0.0) ? (target[2 * n + 0] == 0.0) : (((target[2 * n + 0] - test[0]) / test[0]) < 1e-2));
        assert((test[1] == 0.0) ? (target[2 * n + 1] == 0.0) : (((target[2 * n + 1] - test[1]) / test[1]) < 1e-2));
    }
#endif
    }
}
#endif
#endif
#endif

// --------------------------------------------------------------------------
template<int ndims, int ns, bool isUpsampfacHigh>  
int tiled_interpSorted(BIGINT* sort_indices,BIGINT N1, BIGINT N2, BIGINT N3,
		      FLT *data_uniform,BIGINT M, FLT *kx, FLT *ky, FLT *kz,
		      FLT *data_nonuniform, const spread_opts& opts, int did_sort)
// Interpolate to NU pts in sorted order from a uniform grid.
// See spreadinterp() for doc.
{
    CNTime timer;
  FLT ns2 = (FLT)ns/2;          // half spread width, used as stencil shift
  int nthr = MY_OMP_GET_MAX_THREADS();   // # threads to use to interp
  if (opts.nthreads>0)
    nthr = min(nthr,opts.nthreads);      // user override up to max avail
  if (opts.debug)
    printf("\tinterp %dD (M=%lld; N1=%lld,N2=%lld,N3=%lld; pir=%d), nthr=%d\n",ndims,(long long)M,(long long)N1,(long long)N2,(long long)N3,opts.pirange,nthr);

  timer.start();  
#pragma omp parallel num_threads(nthr)
  {
#define CHUNKSIZE 16     // Chunks of Type 2 targets (Ludvig found by expt)
    BIGINT jlist[CHUNKSIZE];
    FLT xjlist[CHUNKSIZE], yjlist[CHUNKSIZE], zjlist[CHUNKSIZE];
    FLT outbuf[2*CHUNKSIZE];
    // Kernels: static alloc is faster, so we do it for up to 3D...
    alignas(64) FLT kernel_args[3 * MAX_NSPREAD];
    alignas(64) FLT kernel_values[3 * npipes * MAX_NSPREAD];

    FLT* ker1[npipes];
    FLT* ker2[npipes];
    FLT* ker3[npipes];
    FLT* ker = kernel_values;
    for (int n = 0; n < npipes; n++) {
        ker1[n] = ker; ker += MAX_NSPREAD;
        ker2[n] = ker; ker += MAX_NSPREAD;
        ker3[n] = ker; ker += MAX_NSPREAD;
    }

    // Loop over interpolation chunks
#pragma omp for schedule (dynamic,1000)  // assign threads to NU targ pts:
    for (BIGINT i=0; i<M; i+=CHUNKSIZE)  // main loop over NU targs, interp each from U
      {
        // Setup buffers for this chunk
        int bufsize = (i+CHUNKSIZE > M) ? M-i : CHUNKSIZE;
        for (int ibuf=0; ibuf<bufsize; ibuf++) {
          BIGINT j = sort_indices[i+ibuf];
          jlist[ibuf] = j;
	  xjlist[ibuf] = FOLDRESCALE(kx[j],N1,opts.pirange);
	  if(ndims >=2)
	    yjlist[ibuf] = FOLDRESCALE(ky[j],N2,opts.pirange);
	  if(ndims == 3)
	    zjlist[ibuf] = FOLDRESCALE(kz[j],N3,opts.pirange);             
	}
      
    int b0 = bufsize % npipes;

    // Prologue
    for (int ibuf = 0; ibuf < b0; ibuf++) {
        FLT xj = xjlist[ibuf];
        FLT yj = (ndims > 1) ? yjlist[ibuf] : 0;
        FLT zj = (ndims > 2) ? zjlist[ibuf] : 0;

        FLT* target = outbuf + 2 * ibuf;

        // coords (x,y,z), spread block corner index (i1,i2,i3) of current NU targ
        BIGINT i1 = (BIGINT)std::ceil(xj - ns2); // leftmost grid index
        BIGINT i2 = (ndims > 1) ? (BIGINT)std::ceil(yj - ns2) : 0; // min y grid index
        BIGINT i3 = (ndims > 1) ? (BIGINT)std::ceil(zj - ns2) : 0; // min z grid index

        FLT x1 = (FLT)i1 - xj;           // shift of ker center, in [-w/2,-w/2+1]
        FLT x2 = (ndims > 1) ? (FLT)i2 - yj : 0;
        FLT x3 = (ndims > 2) ? (FLT)i3 - zj : 0;

        // eval kernel values patch and use to interpolate from uniform data...
        if (!(opts.flags & TF_OMIT_SPREADING)) {

            if (opts.kerevalmeth == 0) {               // choose eval method
                set_kernel_args(kernel_args, x1, opts);
                if (ndims > 1)  set_kernel_args(kernel_args + ns, x2, opts);
                if (ndims > 2)  set_kernel_args(kernel_args + 2 * ns, x3, opts);

                evaluate_kernel_vector(ker1[0], kernel_args, opts, ndims * ns);
            }

            else {
                eval_kernel_vec_Horner(ker1[0], x1, ns, opts);
                if (ndims > 1) eval_kernel_vec_Horner(ker2[0], x2, ns, opts);
                if (ndims > 2) eval_kernel_vec_Horner(ker3[0], x3, ns, opts);
            }

            switch (ndims) {
            case 1:
                interp_line(target, data_uniform, ker1[0], i1, N1, ns);
                break;
            case 2:
                interp_square(target, data_uniform, ker1[0], ker2[0], i1, i2, N1, N2, ns);
                break;
            case 3:
                interp_cube(target, data_uniform, ker1[0], ker2[0], ker3[0], i1, i2, i3, N1, N2, N3, ns);
                break;
            default: //can't get here
                break;

            }
        }
    } // end loop over targets in chunk

    // Main loop over targets in chunk
    for (int b=b0; b<bufsize; b+=npipes) {
        alignas(64) BIGINT i1[npipes];   // fine grid start indices
        alignas(64) BIGINT i2[npipes];
        alignas(64) BIGINT i3[npipes];
        alignas(64) FLT x1[npipes];
        alignas(64) FLT x2[npipes];
        alignas(64) FLT x3[npipes];

        for (int n = 0; n < npipes; n++) {
            int ibuf = b + n;

            FLT xj = xjlist[ibuf];
            FLT yj = (ndims > 1) ? yjlist[ibuf] : 0;
            FLT zj = (ndims > 2) ? zjlist[ibuf] : 0;

            // coords (x,y,z), spread block corner index (i1,i2,i3) of current NU targ
            i1[n] = (BIGINT)std::ceil(xj - ns2); // leftmost grid index
            i2[n] = (ndims > 1) ? (BIGINT)std::ceil(yj - ns2) : 0; // min y grid index
            i3[n] = (ndims > 1) ? (BIGINT)std::ceil(zj - ns2) : 0; // min z grid index

            x1[n] = (FLT)i1[n] - xj;           // shift of ker center, in [-w/2,-w/2+1]
            x2[n] = (ndims > 1) ? (FLT)i2[n] - yj : 0;
            x3[n] = (ndims > 2) ? (FLT)i3[n] - zj : 0;
        }
      // eval kernel values patch and use to interpolate from uniform data...
        if (!(opts.flags & TF_OMIT_SPREADING)) {

            if (opts.kerevalmeth == 0) {               // choose eval method
                for (int n = 0; n < npipes; n++) {
                    set_kernel_args(kernel_args, x1[n], opts);
                    if (ndims > 1)  set_kernel_args(kernel_args + ns, x2[n], opts);
                    if (ndims > 2)  set_kernel_args(kernel_args + 2 * ns, x3[n], opts);

                    evaluate_kernel_vector(ker1[n], kernel_args, opts, ndims * ns);
                }
            }

            else {
                tiled_eval_kernel_vec_Horner<ns, isUpsampfacHigh>(ker1, x1, opts);
                if (ndims > 1) tiled_eval_kernel_vec_Horner<ns, isUpsampfacHigh>(ker2, x2, opts);
                if (ndims > 2) tiled_eval_kernel_vec_Horner<ns, isUpsampfacHigh>(ker3, x3, opts);
            }

            switch (ndims) {
            case 1:
                for (int n = 0; n < npipes; n++) {
                    FLT* target = outbuf + 2 * b;

                    tiled_interp_line<ns>(target, data_uniform, ker1, i1, N1);
                }
                break;
            case 2:
                for (int n = 0; n < npipes; n++) {
                    int ibuf = b + n;

                    FLT* target = outbuf + 2 * ibuf;

                    interp_square(target, data_uniform, ker1[n], ker2[n], i1[n], i2[n], N1, N2, ns);
                }
                break;
            case 3:
                {
                    FLT* target = outbuf + 2 * b;

                    tiled_interp_cube<ns>(target, data_uniform, ker1, ker2, ker3, i1, i2, i3, N1, N2, N3);
                }
                break;
            default: //can't get here
                break;

            }
        }
    } // end loop over targets in chunk
        
    // Copy result buffer to output array
    for (int ibuf=0; ibuf<bufsize; ibuf++) {
      BIGINT j = jlist[ibuf];
      data_nonuniform[2*j] = outbuf[2*ibuf];
      data_nonuniform[2*j+1] = outbuf[2*ibuf+1];              
    }         
        
      } // end NU targ loop
  } // end parallel section
  if (opts.debug) printf("\tt2 spreading loop: \t%.3g s\n",timer.elapsedsec());
  return 0;
};


#endif // PIPEDINTERP_H
