// Defines templates for the tiled polynomial evaluation code
// And optimized specializations.

#ifndef PIPEDEVAL_H
#define PIPEDEVAL_H

#include <defs.h>
#include <immintrin.h>

#include <cassert>

static inline void set_kernel_args(FLT* args, FLT x, const spread_opts& opts);
static inline void evaluate_kernel_vector(FLT* ker, FLT* args, const spread_opts& opts, const int N);
static inline void eval_kernel_vec_Horner(FLT* ker, const FLT z, const int w, const spread_opts& opts);

template<int w, bool isUpsampfacHigh>
inline void tiled_eval_kernel_vec_Horner(FLT** kers, const FLT* x, const spread_opts& opts) {
    for (int n = 0; n < npipes; n++) {
        eval_kernel_vec_Horner(kers[n], x[n], w, opts);
    }
}

alignas(64) static FLT c70[] = { 3.9948351830487481E+03, 5.4715865608590771E+05, 5.0196413492771760E+06, 9.8206709220713247E+06, 5.0196413492771825E+06, 5.4715865608590783E+05, 3.9948351830642519E+03, 0.0000000000000000E+00 };
alignas(64) static FLT c71[] = { 1.5290160332974696E+04, 8.7628248584320408E+05, 3.4421061790934438E+06, -2.6908159596373561E-10, -3.4421061790934461E+06, -8.7628248584320408E+05, -1.5290160332958067E+04, 0.0000000000000000E+00 };
alignas(64) static FLT c72[] = { 2.4458227486779251E+04, 5.3904618484139396E+05, 2.4315566181017534E+05, -1.6133959371974322E+06, 2.4315566181017453E+05, 5.3904618484139396E+05, 2.4458227486795113E+04, 0.0000000000000000E+00 };
alignas(64) static FLT c73[] = { 2.1166189345881645E+04, 1.3382732160223130E+05, -3.3113450969689694E+05, 6.9013724510092140E-10, 3.3113450969689724E+05, -1.3382732160223136E+05, -2.1166189345866893E+04, 0.0000000000000000E+00 };
alignas(64) static FLT c74[] = { 1.0542795672344864E+04, -7.0739172265098678E+03, -6.5563293056049893E+04, 1.2429734005960064E+05, -6.5563293056049602E+04, -7.0739172265098332E+03, 1.0542795672361213E+04, 0.0000000000000000E+00 };
alignas(64) static FLT c75[] = { 2.7903491906228419E+03, -1.0975382873973093E+04, 1.3656979541144799E+04, 7.7346408577822045E-10, -1.3656979541143772E+04, 1.0975382873973256E+04, -2.7903491906078298E+03, 0.0000000000000000E+00 };
alignas(64) static FLT c76[] = { 1.6069721418053300E+02, -1.5518707872251393E+03, 4.3634273936642621E+03, -5.9891976420595174E+03, 4.3634273936642730E+03, -1.5518707872251064E+03, 1.6069721419533221E+02, 0.0000000000000000E+00 };
alignas(64) static FLT c77[] = { -1.2289277373867256E+02, 2.8583630927743314E+02, -2.8318194617327981E+02, 6.9043515551118249E-10, 2.8318194617392436E+02, -2.8583630927760140E+02, 1.2289277375319763E+02, 0.0000000000000000E+00 };
alignas(64) static FLT c78[] = { -3.2270164914249058E+01, 9.1892112257581346E+01, -1.6710678096334209E+02, 2.0317049305432383E+02, -1.6710678096383771E+02, 9.1892112257416159E+01, -3.2270164900224913E+01, 0.0000000000000000E+00 };
alignas(64) static FLT c79[] = { -1.4761409685186277E-01, -9.1862771280377487E-01, 1.2845147741777752E+00, 5.6547359492808854E-10, -1.2845147728310689E+00, 9.1862771293147971E-01, 1.4761410890866353E-01, 0.0000000000000000E+00 };
    
#ifdef __AVX512F__
#ifdef SINGLE
template<>
inline void tiled_eval_kernel_vec_Horner<7, true>(FLT** kers, const FLT* x, const spread_opts& opts)
{
    if (!(opts.flags & TF_OMIT_EVALUATE_KERNEL)) {
        constexpr FLT z_shift = FLT(6.0);
        __m512 _zab, _zcd, _zef, _zgh;
        __m512 _kerab, _kercd, _keref, _kergh;

        {
            __m256 _x = _mm256_load_ps(x + 0);
            __m256 _z = _mm256_fmadd_ps(
                _x,
                _mm256_set1_ps(2.0),
                _mm256_set1_ps(z_shift));

            __m128 _z_lo = _mm256_castps256_ps128(_z);
            _zab = _mm512_insertf32x8(
                _mm512_broadcastss_ps(_z_lo),
                _mm256_broadcastss_ps(_mm_permute_ps(_z_lo, 0x39)), 1);

            _zcd = _mm512_insertf32x8(
                _mm512_broadcastss_ps(_mm_permute_ps(_z_lo, 0x8e)),
                _mm256_broadcastss_ps(_mm_permute_ps(_z_lo, 0x93)), 1);

            __m128 _z_hi = _mm256_extractf128_ps(_z, 1);
            _zef = _mm512_insertf32x8(
                _mm512_broadcastss_ps(_z_hi),
                _mm256_broadcastss_ps(_mm_permute_ps(_z_hi, 0x39)), 1);

            _zgh = _mm512_insertf32x8(
                _mm512_broadcastss_ps(_mm_permute_ps(_z_hi, 0x8e)),
                _mm256_broadcastss_ps(_mm_permute_ps(_z_hi, 0x93)), 1);
        }

        {
            __m512 _c9 = _mm512_broadcast_f32x8(_mm256_load_ps(c79 + 0));
            __m512 _c8 = _mm512_broadcast_f32x8(_mm256_load_ps(c78 + 0));
            _kerab = _mm512_fmadd_ps(_zab, _c9, _c8);
            _kercd = _mm512_fmadd_ps(_zcd, _c9, _c8);
            _keref = _mm512_fmadd_ps(_zef, _c9, _c8);
            _kergh = _mm512_fmadd_ps(_zgh, _c9, _c8);
        }

        {
            __m512 _c7 = _mm512_broadcast_f32x8(_mm256_load_ps(c77 + 0));
            _kerab = _mm512_fmadd_ps(_zab, _kerab, _c7);
            _kercd = _mm512_fmadd_ps(_zcd, _kercd, _c7);
            _keref = _mm512_fmadd_ps(_zef, _keref, _c7);
            _kergh = _mm512_fmadd_ps(_zgh, _kergh, _c7);
        }

        {
            __m512 _c6 = _mm512_broadcast_f32x8(_mm256_load_ps(c76 + 0));
            _kerab = _mm512_fmadd_ps(_zab, _kerab, _c6);
            _kercd = _mm512_fmadd_ps(_zcd, _kercd, _c6);
            _keref = _mm512_fmadd_ps(_zef, _keref, _c6);
            _kergh = _mm512_fmadd_ps(_zgh, _kergh, _c6);
        }

        {
            __m512 _c5 = _mm512_broadcast_f32x8(_mm256_load_ps(c75 + 0));
            _kerab = _mm512_fmadd_ps(_zab, _kerab, _c5);
            _kercd = _mm512_fmadd_ps(_zcd, _kercd, _c5);
            _keref = _mm512_fmadd_ps(_zef, _keref, _c5);
            _kergh = _mm512_fmadd_ps(_zgh, _kergh, _c5);
        }

        {
            __m512 _c4 = _mm512_broadcast_f32x8(_mm256_load_ps(c74 + 0));
            _kerab = _mm512_fmadd_ps(_zab, _kerab, _c4);
            _kercd = _mm512_fmadd_ps(_zcd, _kercd, _c4);
            _keref = _mm512_fmadd_ps(_zef, _keref, _c4);
            _kergh = _mm512_fmadd_ps(_zgh, _kergh, _c4);
        }

        {
            __m512 _c3 = _mm512_broadcast_f32x8(_mm256_load_ps(c73 + 0));
            _kerab = _mm512_fmadd_ps(_zab, _kerab, _c3);
            _kercd = _mm512_fmadd_ps(_zcd, _kercd, _c3);
            _keref = _mm512_fmadd_ps(_zef, _keref, _c3);
            _kergh = _mm512_fmadd_ps(_zgh, _kergh, _c3);
        }

        {
            __m512 _c2 = _mm512_broadcast_f32x8(_mm256_load_ps(c72 + 0));
            _kerab = _mm512_fmadd_ps(_zab, _kerab, _c2);
            _kercd = _mm512_fmadd_ps(_zcd, _kercd, _c2);
            _keref = _mm512_fmadd_ps(_zef, _keref, _c2);
            _kergh = _mm512_fmadd_ps(_zgh, _kergh, _c2);
        }

        {
            __m512 _c1 = _mm512_broadcast_f32x8(_mm256_load_ps(c71 + 0));
            _kerab = _mm512_fmadd_ps(_zab, _kerab, _c1);
            _kercd = _mm512_fmadd_ps(_zcd, _kercd, _c1);
            _keref = _mm512_fmadd_ps(_zef, _keref, _c1);
            _kergh = _mm512_fmadd_ps(_zgh, _kergh, _c1);
        }

        {
            __m512 _c0 = _mm512_broadcast_f32x8(_mm256_load_ps(c70 + 0));
            _kerab = _mm512_fmadd_ps(_zab, _kerab, _c0);
            _kercd = _mm512_fmadd_ps(_zcd, _kercd, _c0);
            _keref = _mm512_fmadd_ps(_zef, _keref, _c0);
            _kergh = _mm512_fmadd_ps(_zgh, _kergh, _c0);
        }

        _mm256_store_ps(kers[0] + 0, _mm512_castps512_ps256(_kerab));
        _mm256_store_ps(kers[1] + 0, _mm512_extractf32x8_ps(_kerab, 1));
        _mm256_store_ps(kers[2] + 0, _mm512_castps512_ps256(_kercd));
        _mm256_store_ps(kers[3] + 0, _mm512_extractf32x8_ps(_kercd, 1));
        _mm256_store_ps(kers[4] + 0, _mm512_castps512_ps256(_keref));
        _mm256_store_ps(kers[5] + 0, _mm512_extractf32x8_ps(_keref, 1));
        _mm256_store_ps(kers[6] + 0, _mm512_castps512_ps256(_kergh));
        _mm256_store_ps(kers[7] + 0, _mm512_extractf32x8_ps(_kergh, 1));

#ifdef _DEBUG
        for (int n = 0; n < npipes; n++) {
            constexpr int w = 7;
            FLT* ker = kers[n];
            FLT z = 2 * x[n] + w - 1.0;         // scale so local grid offset z in [-1,1]
            FLT test[8];
            for (int i = 0; i < 8; i++) {
                test[i] = c70[i] + z * (c71[i] + z * (c72[i] + z * (c73[i] + z * (c74[i] + z * (c75[i] + z * (c76[i] + z * (c77[i] + z * (c78[i] + z * (c79[i])))))))));
                assert(test[i] == 0.0 ? kers[n][i] == 0.0 : (abs(test[i] - kers[n][i]) / test[i]) < 1e-2);
            }
        }
#endif
    }
}
#else
template<>
inline void tiled_eval_kernel_vec_Horner<7, true>(FLT** kers, const FLT* x, const spread_opts& opts)
{
    if (!(opts.flags & TF_OMIT_EVALUATE_KERNEL)) {
        constexpr FLT z_shift = FLT(6.0);
        __m512d _za, _zb, _zc, _zd, _ze, _zf, _zg, _zh;
        __m512d _kera, _kerb, _kerc, _kerd, _kere, _kerf, _kerg, _kerh;

        {
            __m512d _x = _mm512_load_pd(x + 0);
            __m512d _z = _mm512_fmadd_pd(
                _x,
                _mm512_set1_pd(2.0),
                _mm512_set1_pd(z_shift));

            __m128d _z0 = _mm512_castpd512_pd128(_z);
            _za = _mm512_broadcastsd_pd(_z0);
            _zb = _mm512_broadcastsd_pd(_mm_permute_pd(_z0, 0x01));

            __m128d _z1 = _mm512_extractf64x2_pd(_z, 1);
            _zc = _mm512_broadcastsd_pd(_z1);
            _zd = _mm512_broadcastsd_pd(_mm_permute_pd(_z1, 0x01));

            __m128d _z2 = _mm512_extractf64x2_pd(_z, 2);
            _ze = _mm512_broadcastsd_pd(_z2);
            _zf = _mm512_broadcastsd_pd(_mm_permute_pd(_z2, 0x01));

            __m128d _z3 = _mm512_extractf64x2_pd(_z, 3);
            _zg = _mm512_broadcastsd_pd(_z3);
            _zh = _mm512_broadcastsd_pd(_mm_permute_pd(_z3, 0x01));
        }

        {
            __m512d _c9 = _mm512_load_pd(c79 + 0);
            __m512d _c8 = _mm512_load_pd(c78 + 0);
            _kera = _mm512_fmadd_pd(_za, _c9, _c8);
            _kerb = _mm512_fmadd_pd(_zb, _c9, _c8);
            _kerc = _mm512_fmadd_pd(_zc, _c9, _c8);
            _kerd = _mm512_fmadd_pd(_zd, _c9, _c8);
            _kere = _mm512_fmadd_pd(_ze, _c9, _c8);
            _kerf = _mm512_fmadd_pd(_zf, _c9, _c8);
            _kerg = _mm512_fmadd_pd(_zg, _c9, _c8);
            _kerh = _mm512_fmadd_pd(_zh, _c9, _c8);
        }

        {
            __m512d _c7 = _mm512_load_pd(c77 + 0);
            _kera = _mm512_fmadd_pd(_za, _kera, _c7);
            _kerb = _mm512_fmadd_pd(_zb, _kerb, _c7);
            _kerc = _mm512_fmadd_pd(_zc, _kerc, _c7);
            _kerd = _mm512_fmadd_pd(_zd, _kerd, _c7);
            _kere = _mm512_fmadd_pd(_ze, _kere, _c7);
            _kerf = _mm512_fmadd_pd(_zf, _kerf, _c7);
            _kerg = _mm512_fmadd_pd(_zg, _kerg, _c7);
            _kerh = _mm512_fmadd_pd(_zh, _kerh, _c7);
        }

        {
            __m512d _c6 = _mm512_load_pd(c76 + 0);
            _kera = _mm512_fmadd_pd(_za, _kera, _c6);
            _kerb = _mm512_fmadd_pd(_zb, _kerb, _c6);
            _kerc = _mm512_fmadd_pd(_zc, _kerc, _c6);
            _kerd = _mm512_fmadd_pd(_zd, _kerd, _c6);
            _kere = _mm512_fmadd_pd(_ze, _kere, _c6);
            _kerf = _mm512_fmadd_pd(_zf, _kerf, _c6);
            _kerg = _mm512_fmadd_pd(_zg, _kerg, _c6);
            _kerh = _mm512_fmadd_pd(_zh, _kerh, _c6);
        }

        {
            __m512d _c5 = _mm512_load_pd(c75 + 0);
            _kera = _mm512_fmadd_pd(_za, _kera, _c5);
            _kerb = _mm512_fmadd_pd(_zb, _kerb, _c5);
            _kerc = _mm512_fmadd_pd(_zc, _kerc, _c5);
            _kerd = _mm512_fmadd_pd(_zd, _kerd, _c5);
            _kere = _mm512_fmadd_pd(_ze, _kere, _c5);
            _kerf = _mm512_fmadd_pd(_zf, _kerf, _c5);
            _kerg = _mm512_fmadd_pd(_zg, _kerg, _c5);
            _kerh = _mm512_fmadd_pd(_zh, _kerh, _c5);
        }

        {
            __m512d _c4 = _mm512_load_pd(c74 + 0);
            _kera = _mm512_fmadd_pd(_za, _kera, _c4);
            _kerb = _mm512_fmadd_pd(_zb, _kerb, _c4);
            _kerc = _mm512_fmadd_pd(_zc, _kerc, _c4);
            _kerd = _mm512_fmadd_pd(_zd, _kerd, _c4);
            _kere = _mm512_fmadd_pd(_ze, _kere, _c4);
            _kerf = _mm512_fmadd_pd(_zf, _kerf, _c4);
            _kerg = _mm512_fmadd_pd(_zg, _kerg, _c4);
            _kerh = _mm512_fmadd_pd(_zh, _kerh, _c4);
        }

        {
            __m512d _c3 = _mm512_load_pd(c73 + 0);
            _kera = _mm512_fmadd_pd(_za, _kera, _c3);
            _kerb = _mm512_fmadd_pd(_zb, _kerb, _c3);
            _kerc = _mm512_fmadd_pd(_zc, _kerc, _c3);
            _kerd = _mm512_fmadd_pd(_zd, _kerd, _c3);
            _kere = _mm512_fmadd_pd(_ze, _kere, _c3);
            _kerf = _mm512_fmadd_pd(_zf, _kerf, _c3);
            _kerg = _mm512_fmadd_pd(_zg, _kerg, _c3);
            _kerh = _mm512_fmadd_pd(_zh, _kerh, _c3);
        }

        {
            __m512d _c2 = _mm512_load_pd(c72 + 0);
            _kera = _mm512_fmadd_pd(_za, _kera, _c2);
            _kerb = _mm512_fmadd_pd(_zb, _kerb, _c2);
            _kerc = _mm512_fmadd_pd(_zc, _kerc, _c2);
            _kerd = _mm512_fmadd_pd(_zd, _kerd, _c2);
            _kere = _mm512_fmadd_pd(_ze, _kere, _c2);
            _kerf = _mm512_fmadd_pd(_zf, _kerf, _c2);
            _kerg = _mm512_fmadd_pd(_zg, _kerg, _c2);
            _kerh = _mm512_fmadd_pd(_zh, _kerh, _c2);
        }

        {
            __m512d _c1 = _mm512_load_pd(c71 + 0);
            _kera = _mm512_fmadd_pd(_za, _kera, _c1);
            _kerb = _mm512_fmadd_pd(_zb, _kerb, _c1);
            _kerc = _mm512_fmadd_pd(_zc, _kerc, _c1);
            _kerd = _mm512_fmadd_pd(_zd, _kerd, _c1);
            _kere = _mm512_fmadd_pd(_ze, _kere, _c1);
            _kerf = _mm512_fmadd_pd(_zf, _kerf, _c1);
            _kerg = _mm512_fmadd_pd(_zg, _kerg, _c1);
            _kerh = _mm512_fmadd_pd(_zh, _kerh, _c1);
        }

        {
            __m512d _c0 = _mm512_load_pd(c70 + 0);
            _kera = _mm512_fmadd_pd(_za, _kera, _c0);
            _kerb = _mm512_fmadd_pd(_zb, _kerb, _c0);
            _kerc = _mm512_fmadd_pd(_zc, _kerc, _c0);
            _kerd = _mm512_fmadd_pd(_zd, _kerd, _c0);
            _kere = _mm512_fmadd_pd(_ze, _kere, _c0);
            _kerf = _mm512_fmadd_pd(_zf, _kerf, _c0);
            _kerg = _mm512_fmadd_pd(_zg, _kerg, _c0);
            _kerh = _mm512_fmadd_pd(_zh, _kerh, _c0);
        }

        _mm512_store_pd(kers[0] + 0, _kera);
        _mm512_store_pd(kers[1] + 0, _kerb);
        _mm512_store_pd(kers[2] + 0, _kerc);
        _mm512_store_pd(kers[3] + 0, _kerd);
        _mm512_store_pd(kers[4] + 0, _kere);
        _mm512_store_pd(kers[5] + 0, _kerf);
        _mm512_store_pd(kers[6] + 0, _kerg);
        _mm512_store_pd(kers[7] + 0, _kerh);

#ifdef _DEBUG
        for (int n = 0; n < npipes; n++) {
            constexpr int w = 7;
            FLT z = 2 * x[n] + w - 1.0;         // scale so local grid offset z in [-1,1]
            FLT test[8];
            for (int i = 0; i < 8; i++) {
                test[i] = c70[i] + z * (c71[i] + z * (c72[i] + z * (c73[i] + z * (c74[i] + z * (c75[i] + z * (c76[i] + z * (c77[i] + z * (c78[i] + z * (c79[i])))))))));
                assert(test[i] == 0.0 ? kers[n][i] == 0.0 : (abs(test[i] - kers[n][i]) / test[i]) < 1e-2);
            }
        }
#endif
    }
}
#endif
#else
#ifdef __AVX2__
#ifdef SINGLE
template<>
inline void tiled_eval_kernel_vec_Horner<7, true>(FLT** kers, const FLT* x, const spread_opts& opts)
{
    if (!(opts.flags & TF_OMIT_EVALUATE_KERNEL)) {
        constexpr FLT z_shift = FLT(6.0);
        __m256 _z;
        __m256 _za, _zb, _zc, _zd;
        __m256 _kera, _kerb, _kerc, _kerd;
        
        {
            __m256 _x = _mm256_load_ps(x + 0);
            _z = _mm256_fmadd_ps(
                _x,
                _mm256_set1_ps(2.0),
                _mm256_set1_ps(z_shift));
        }

        {
            __m128 _z_lo = _mm256_castps256_ps128(_z);
            _za = _mm256_broadcastss_ps(_z_lo);
            _zb = _mm256_broadcastss_ps(_mm_permute_ps(_z_lo, 0x39));
            _zc = _mm256_broadcastss_ps(_mm_permute_ps(_z_lo, 0x8e));
            _zd = _mm256_broadcastss_ps(_mm_permute_ps(_z_lo, 0x93));
        }

        {
            __m256 _c9 = _mm256_load_ps(c79 + 0);
            __m256 _c8 = _mm256_load_ps(c78 + 0);
            _kera = _mm256_fmadd_ps(_za, _c9, _c8);
            _kerb = _mm256_fmadd_ps(_zb, _c9, _c8);
            _kerc = _mm256_fmadd_ps(_zc, _c9, _c8);
            _kerd = _mm256_fmadd_ps(_zd, _c9, _c8);
        }

        {
            __m256 _c7 = _mm256_load_ps(c77 + 0);
            _kera = _mm256_fmadd_ps(_za, _kera, _c7);
            _kerb = _mm256_fmadd_ps(_zb, _kerb, _c7);
            _kerc = _mm256_fmadd_ps(_zc, _kerc, _c7);
            _kerd = _mm256_fmadd_ps(_zd, _kerd, _c7);
        }

        {
            __m256 _c6 = _mm256_load_ps(c76 + 0);
            _kera = _mm256_fmadd_ps(_za, _kera, _c6);
            _kerb = _mm256_fmadd_ps(_zb, _kerb, _c6);
            _kerc = _mm256_fmadd_ps(_zc, _kerc, _c6);
            _kerd = _mm256_fmadd_ps(_zd, _kerd, _c6);
        }

        {
            __m256 _c5 = _mm256_load_ps(c75 + 0);
            _kera = _mm256_fmadd_ps(_za, _kera, _c5);
            _kerb = _mm256_fmadd_ps(_zb, _kerb, _c5);
            _kerc = _mm256_fmadd_ps(_zc, _kerc, _c5);
            _kerd = _mm256_fmadd_ps(_zd, _kerd, _c5);
        }

        {
            __m256 _c4 = _mm256_load_ps(c74 + 0);
            _kera = _mm256_fmadd_ps(_za, _kera, _c4);
            _kerb = _mm256_fmadd_ps(_zb, _kerb, _c4);
            _kerc = _mm256_fmadd_ps(_zc, _kerc, _c4);
            _kerd = _mm256_fmadd_ps(_zd, _kerd, _c4);
        }

        {
            __m256 _c3 = _mm256_load_ps(c73 + 0);
            _kera = _mm256_fmadd_ps(_za, _kera, _c3);
            _kerb = _mm256_fmadd_ps(_zb, _kerb, _c3);
            _kerc = _mm256_fmadd_ps(_zc, _kerc, _c3);
            _kerd = _mm256_fmadd_ps(_zd, _kerd, _c3);
        }

        {
            __m256 _c2 = _mm256_load_ps(c72 + 0);
            _kera = _mm256_fmadd_ps(_za, _kera, _c2);
            _kerb = _mm256_fmadd_ps(_zb, _kerb, _c2);
            _kerc = _mm256_fmadd_ps(_zc, _kerc, _c2);
            _kerd = _mm256_fmadd_ps(_zd, _kerd, _c2);
        }

        {
            __m256 _c1 = _mm256_load_ps(c71 + 0);
            _kera = _mm256_fmadd_ps(_za, _kera, _c1);
            _kerb = _mm256_fmadd_ps(_zb, _kerb, _c1);
            _kerc = _mm256_fmadd_ps(_zc, _kerc, _c1);
            _kerd = _mm256_fmadd_ps(_zd, _kerd, _c1);
        }

        {
            __m256 _c0 = _mm256_load_ps(c70 + 0);
            _kera = _mm256_fmadd_ps(_za, _kera, _c0);
            _kerb = _mm256_fmadd_ps(_zb, _kerb, _c0);
            _kerc = _mm256_fmadd_ps(_zc, _kerc, _c0);
            _kerd = _mm256_fmadd_ps(_zd, _kerd, _c0);
        }

        _mm256_store_ps(kers[0] + 0, _kera);
        _mm256_store_ps(kers[1] + 0, _kerb);
        _mm256_store_ps(kers[2] + 0, _kerc);
        _mm256_store_ps(kers[3] + 0, _kerd);

        {
            __m128 _z_hi = _mm256_extractf128_ps(_z, 1);
            _za = _mm256_broadcastss_ps(_z_hi);
            _zb = _mm256_broadcastss_ps(_mm_permute_ps(_z_hi, 0x39));
            _zc = _mm256_broadcastss_ps(_mm_permute_ps(_z_hi, 0x8e));
            _zd = _mm256_broadcastss_ps(_mm_permute_ps(_z_hi, 0x93));
        }

        {
            __m256 _c9 = _mm256_load_ps(c79 + 0);
            __m256 _c8 = _mm256_load_ps(c78 + 0);
            _kera = _mm256_fmadd_ps(_za, _c9, _c8);
            _kerb = _mm256_fmadd_ps(_zb, _c9, _c8);
            _kerc = _mm256_fmadd_ps(_zc, _c9, _c8);
            _kerd = _mm256_fmadd_ps(_zd, _c9, _c8);
        }

        {
            __m256 _c7 = _mm256_load_ps(c77 + 0);
            _kera = _mm256_fmadd_ps(_za, _kera, _c7);
            _kerb = _mm256_fmadd_ps(_zb, _kerb, _c7);
            _kerc = _mm256_fmadd_ps(_zc, _kerc, _c7);
            _kerd = _mm256_fmadd_ps(_zd, _kerd, _c7);
        }

        {
            __m256 _c6 = _mm256_load_ps(c76 + 0);
            _kera = _mm256_fmadd_ps(_za, _kera, _c6);
            _kerb = _mm256_fmadd_ps(_zb, _kerb, _c6);
            _kerc = _mm256_fmadd_ps(_zc, _kerc, _c6);
            _kerd = _mm256_fmadd_ps(_zd, _kerd, _c6);
        }

        {
            __m256 _c5 = _mm256_load_ps(c75 + 0);
            _kera = _mm256_fmadd_ps(_za, _kera, _c5);
            _kerb = _mm256_fmadd_ps(_zb, _kerb, _c5);
            _kerc = _mm256_fmadd_ps(_zc, _kerc, _c5);
            _kerd = _mm256_fmadd_ps(_zd, _kerd, _c5);
        }

        {
            __m256 _c4 = _mm256_load_ps(c74 + 0);
            _kera = _mm256_fmadd_ps(_za, _kera, _c4);
            _kerb = _mm256_fmadd_ps(_zb, _kerb, _c4);
            _kerc = _mm256_fmadd_ps(_zc, _kerc, _c4);
            _kerd = _mm256_fmadd_ps(_zd, _kerd, _c4);
        }

        {
            __m256 _c3 = _mm256_load_ps(c73 + 0);
            _kera = _mm256_fmadd_ps(_za, _kera, _c3);
            _kerb = _mm256_fmadd_ps(_zb, _kerb, _c3);
            _kerc = _mm256_fmadd_ps(_zc, _kerc, _c3);
            _kerd = _mm256_fmadd_ps(_zd, _kerd, _c3);
        }

        {
            __m256 _c2 = _mm256_load_ps(c72 + 0);
            _kera = _mm256_fmadd_ps(_za, _kera, _c2);
            _kerb = _mm256_fmadd_ps(_zb, _kerb, _c2);
            _kerc = _mm256_fmadd_ps(_zc, _kerc, _c2);
            _kerd = _mm256_fmadd_ps(_zd, _kerd, _c2);
        }

        {
            __m256 _c1 = _mm256_load_ps(c71 + 0);
            _kera = _mm256_fmadd_ps(_za, _kera, _c1);
            _kerb = _mm256_fmadd_ps(_zb, _kerb, _c1);
            _kerc = _mm256_fmadd_ps(_zc, _kerc, _c1);
            _kerd = _mm256_fmadd_ps(_zd, _kerd, _c1);
        }

        {
            __m256 _c0 = _mm256_load_ps(c70 + 0);
            _kera = _mm256_fmadd_ps(_za, _kera, _c0);
            _kerb = _mm256_fmadd_ps(_zb, _kerb, _c0);
            _kerc = _mm256_fmadd_ps(_zc, _kerc, _c0);
            _kerd = _mm256_fmadd_ps(_zd, _kerd, _c0);
        }

        _mm256_store_ps(kers[4] + 0, _kera);
        _mm256_store_ps(kers[5] + 0, _kerb);
        _mm256_store_ps(kers[6] + 0, _kerc);
        _mm256_store_ps(kers[7] + 0, _kerd);

#ifdef _DEBUG
        for (int n = 0; n < npipes; n++) {
            constexpr int w = 7;
            FLT* ker = kers[n];
            FLT z = 2 * x[n] + w - 1.0;         // scale so local grid offset z in [-1,1]
            FLT test[8];
            for (int i = 0; i < 8; i++) {
                test[i] = c0[i] + z * (c1[i] + z * (c2[i] + z * (c3[i] + z * (c4[i] + z * (c5[i] + z * (c6[i] + z * (c7[i] + z * (c8[i] + z * (c9[i])))))))));
                assert( test[i] ==0.0 ? kers[n][i] == 0.0 : (abs(test[i] - kers[n][i]) / test[i]) < 1e-2);
            }
         }
#endif
    }
}
#else
template<>
inline void tiled_eval_kernel_vec_Horner<7, true>(FLT * *kers, const FLT * x, const spread_opts & opts)
{
    if (!(opts.flags & TF_OMIT_EVALUATE_KERNEL)) {
        constexpr FLT z_shift = FLT(6.0);
        __m256d _za, _zb, _zc, _zd;
        __m256d _kera, _kerb, _kerc, _kerd;

        {
            __m256d _x = _mm256_load_pd(x + 0);
            __m256d _z = _mm256_fmadd_pd(
                _x,
                _mm256_set1_pd(2.0),
                _mm256_set1_pd(z_shift));

            __m128d _z_lo = _mm256_castpd256_pd128(_z);
            _za = _mm256_broadcastsd_pd(_z_lo);
            _zb = _mm256_broadcastsd_pd(_mm_permute_pd(_z_lo, 0x1));

            __m128d _z_hi = _mm256_extractf128_pd(_z, 1);
            _zc = _mm256_broadcastsd_pd(_z_hi);
            _zd = _mm256_broadcastsd_pd(_mm_permute_pd(_z_hi, 0x1));
        }

        {
            __m256d _c9 = _mm256_load_pd(c79 + 0);
            __m256d _c8 = _mm256_load_pd(c78 + 0);
            _kera = _mm256_fmadd_pd(_za, _c9, _c8);
            _kerb = _mm256_fmadd_pd(_zb, _c9, _c8);
            _kerc = _mm256_fmadd_pd(_zc, _c9, _c8);
            _kerd = _mm256_fmadd_pd(_zd, _c9, _c8);
        }

        {
            __m256d _c7 = _mm256_load_pd(c77 + 0);
            _kera = _mm256_fmadd_pd(_za, _kera, _c7);
            _kerb = _mm256_fmadd_pd(_zb, _kerb, _c7);
            _kerc = _mm256_fmadd_pd(_zc, _kerc, _c7);
            _kerd = _mm256_fmadd_pd(_zd, _kerd, _c7);
                }

        {
            __m256d _c6 = _mm256_load_pd(c76 + 0);
            _kera = _mm256_fmadd_pd(_za, _kera, _c6);
            _kerb = _mm256_fmadd_pd(_zb, _kerb, _c6);
            _kerc = _mm256_fmadd_pd(_zc, _kerc, _c6);
            _kerd = _mm256_fmadd_pd(_zd, _kerd, _c6);
        }

        {
            __m256d _c5 = _mm256_load_pd(c75 + 0);
            _kera = _mm256_fmadd_pd(_za, _kera, _c5);
            _kerb = _mm256_fmadd_pd(_zb, _kerb, _c5);
            _kerc = _mm256_fmadd_pd(_zc, _kerc, _c5);
            _kerd = _mm256_fmadd_pd(_zd, _kerd, _c5);
        }

        {
            __m256d _c4 = _mm256_load_pd(c74 + 0);
            _kera = _mm256_fmadd_pd(_za, _kera, _c4);
            _kerb = _mm256_fmadd_pd(_zb, _kerb, _c4);
            _kerc = _mm256_fmadd_pd(_zc, _kerc, _c4);
            _kerd = _mm256_fmadd_pd(_zd, _kerd, _c4);
        }

        {
            __m256d _c3 = _mm256_load_pd(c73 + 0);
            _kera = _mm256_fmadd_pd(_za, _kera, _c3);
            _kerb = _mm256_fmadd_pd(_zb, _kerb, _c3);
            _kerc = _mm256_fmadd_pd(_zc, _kerc, _c3);
            _kerd = _mm256_fmadd_pd(_zd, _kerd, _c3);
        }

        {
            __m256d _c2 = _mm256_load_pd(c72 + 0);
            _kera = _mm256_fmadd_pd(_za, _kera, _c2);
            _kerb = _mm256_fmadd_pd(_zb, _kerb, _c2);
            _kerc = _mm256_fmadd_pd(_zc, _kerc, _c2);
            _kerd = _mm256_fmadd_pd(_zd, _kerd, _c2);
        }

        {
            __m256d _c1 = _mm256_load_pd(c71 + 0);
            _kera = _mm256_fmadd_pd(_za, _kera, _c1);
            _kerb = _mm256_fmadd_pd(_zb, _kerb, _c1);
            _kerc = _mm256_fmadd_pd(_zc, _kerc, _c1);
            _kerd = _mm256_fmadd_pd(_zd, _kerd, _c1);
        }

        {
            __m256d _c0 = _mm256_load_pd(c70 + 0);
            _kera = _mm256_fmadd_pd(_za, _kera, _c0);
            _kerb = _mm256_fmadd_pd(_zb, _kerb, _c0);
            _kerc = _mm256_fmadd_pd(_zc, _kerc, _c0);
            _kerd = _mm256_fmadd_pd(_zd, _kerd, _c0);
        }

        _mm256_store_pd(kers[0] + 0, _kera);
        _mm256_store_pd(kers[1] + 0, _kerb);
        _mm256_store_pd(kers[2] + 0, _kerc);
        _mm256_store_pd(kers[3] + 0, _kerd);

        {
            __m256d _x = _mm256_load_pd(x + 4);
            __m256d _z = _mm256_fmadd_pd(
                _x,
                _mm256_set1_pd(2.0),
                _mm256_set1_pd(z_shift));

            __m128d _z_lo = _mm256_castpd256_pd128(_z);
            _za = _mm256_broadcastsd_pd(_z_lo);
            _zb = _mm256_broadcastsd_pd(_mm_permute_pd(_z_lo, 0x1));

            __m128d _z_hi = _mm256_extractf128_pd(_z, 1);
            _zc = _mm256_broadcastsd_pd(_z_hi);
            _zd = _mm256_broadcastsd_pd(_mm_permute_pd(_z_hi, 0x1));
        }

        {
            __m256d _c9 = _mm256_load_pd(c79 + 0);
            __m256d _c8 = _mm256_load_pd(c78 + 0);
            _kera = _mm256_fmadd_pd(_za, _c9, _c8);
            _kerb = _mm256_fmadd_pd(_zb, _c9, _c8);
            _kerc = _mm256_fmadd_pd(_zc, _c9, _c8);
            _kerd = _mm256_fmadd_pd(_zd, _c9, _c8);
        }

        {
            __m256d _c7 = _mm256_load_pd(c77 + 0);
            _kera = _mm256_fmadd_pd(_za, _kera, _c7);
            _kerb = _mm256_fmadd_pd(_zb, _kerb, _c7);
            _kerc = _mm256_fmadd_pd(_zc, _kerc, _c7);
            _kerd = _mm256_fmadd_pd(_zd, _kerd, _c7);
        }

        {
            __m256d _c6 = _mm256_load_pd(c76 + 0);
            _kera = _mm256_fmadd_pd(_za, _kera, _c6);
            _kerb = _mm256_fmadd_pd(_zb, _kerb, _c6);
            _kerc = _mm256_fmadd_pd(_zc, _kerc, _c6);
            _kerd = _mm256_fmadd_pd(_zd, _kerd, _c6);
        }

        {
            __m256d _c5 = _mm256_load_pd(c75 + 0);
            _kera = _mm256_fmadd_pd(_za, _kera, _c5);
            _kerb = _mm256_fmadd_pd(_zb, _kerb, _c5);
            _kerc = _mm256_fmadd_pd(_zc, _kerc, _c5);
            _kerd = _mm256_fmadd_pd(_zd, _kerd, _c5);
        }

        {
            __m256d _c4 = _mm256_load_pd(c74 + 0);
            _kera = _mm256_fmadd_pd(_za, _kera, _c4);
            _kerb = _mm256_fmadd_pd(_zb, _kerb, _c4);
            _kerc = _mm256_fmadd_pd(_zc, _kerc, _c4);
            _kerd = _mm256_fmadd_pd(_zd, _kerd, _c4);
        }

        {
            __m256d _c3 = _mm256_load_pd(c73 + 0);
            _kera = _mm256_fmadd_pd(_za, _kera, _c3);
            _kerb = _mm256_fmadd_pd(_zb, _kerb, _c3);
            _kerc = _mm256_fmadd_pd(_zc, _kerc, _c3);
            _kerd = _mm256_fmadd_pd(_zd, _kerd, _c3);
        }

        {
            __m256d _c2 = _mm256_load_pd(c72 + 0);
            _kera = _mm256_fmadd_pd(_za, _kera, _c2);
            _kerb = _mm256_fmadd_pd(_zb, _kerb, _c2);
            _kerc = _mm256_fmadd_pd(_zc, _kerc, _c2);
            _kerd = _mm256_fmadd_pd(_zd, _kerd, _c2);
        }

        {
            __m256d _c1 = _mm256_load_pd(c71 + 0);
            _kera = _mm256_fmadd_pd(_za, _kera, _c1);
            _kerb = _mm256_fmadd_pd(_zb, _kerb, _c1);
            _kerc = _mm256_fmadd_pd(_zc, _kerc, _c1);
            _kerd = _mm256_fmadd_pd(_zd, _kerd, _c1);
        }

        {
            __m256d _c0 = _mm256_load_pd(c70 + 0);
            _kera = _mm256_fmadd_pd(_za, _kera, _c0);
            _kerb = _mm256_fmadd_pd(_zb, _kerb, _c0);
            _kerc = _mm256_fmadd_pd(_zc, _kerc, _c0);
            _kerd = _mm256_fmadd_pd(_zd, _kerd, _c0);
        }

        _mm256_store_pd(kers[4] + 0, _kera);
        _mm256_store_pd(kers[5] + 0, _kerb);
        _mm256_store_pd(kers[6] + 0, _kerc);
        _mm256_store_pd(kers[7] + 0, _kerd);

        {
            __m256d _x = _mm256_load_pd(x + 0);
            __m256d _z = _mm256_fmadd_pd(
                _x,
                _mm256_set1_pd(2.0),
                _mm256_set1_pd(z_shift));

            __m128d _z_lo = _mm256_castpd256_pd128(_z);
            _za = _mm256_broadcastsd_pd(_z_lo);
            _zb = _mm256_broadcastsd_pd(_mm_permute_pd(_z_lo, 0x1));

            __m128d _z_hi = _mm256_extractf128_pd(_z, 1);
            _zc = _mm256_broadcastsd_pd(_z_hi);
            _zd = _mm256_broadcastsd_pd(_mm_permute_pd(_z_hi, 0x1));
        }

        {
            __m256d _c9 = _mm256_load_pd(c79 + 4);
            __m256d _c8 = _mm256_load_pd(c78 + 4);
            _kera = _mm256_fmadd_pd(_za, _c9, _c8);
            _kerb = _mm256_fmadd_pd(_zb, _c9, _c8);
            _kerc = _mm256_fmadd_pd(_zc, _c9, _c8);
            _kerd = _mm256_fmadd_pd(_zd, _c9, _c8);
        }

        {
            __m256d _c7 = _mm256_load_pd(c77 + 4);
            _kera = _mm256_fmadd_pd(_za, _kera, _c7);
            _kerb = _mm256_fmadd_pd(_zb, _kerb, _c7);
            _kerc = _mm256_fmadd_pd(_zc, _kerc, _c7);
            _kerd = _mm256_fmadd_pd(_zd, _kerd, _c7);
        }

        {
            __m256d _c6 = _mm256_load_pd(c76 + 4);
            _kera = _mm256_fmadd_pd(_za, _kera, _c6);
            _kerb = _mm256_fmadd_pd(_zb, _kerb, _c6);
            _kerc = _mm256_fmadd_pd(_zc, _kerc, _c6);
            _kerd = _mm256_fmadd_pd(_zd, _kerd, _c6);
        }

        {
            __m256d _c5 = _mm256_load_pd(c75 + 4);
            _kera = _mm256_fmadd_pd(_za, _kera, _c5);
            _kerb = _mm256_fmadd_pd(_zb, _kerb, _c5);
            _kerc = _mm256_fmadd_pd(_zc, _kerc, _c5);
            _kerd = _mm256_fmadd_pd(_zd, _kerd, _c5);
        }

        {
            __m256d _c4 = _mm256_load_pd(c74 + 4);
            _kera = _mm256_fmadd_pd(_za, _kera, _c4);
            _kerb = _mm256_fmadd_pd(_zb, _kerb, _c4);
            _kerc = _mm256_fmadd_pd(_zc, _kerc, _c4);
            _kerd = _mm256_fmadd_pd(_zd, _kerd, _c4);
        }

        {
            __m256d _c3 = _mm256_load_pd(c73 + 4);
            _kera = _mm256_fmadd_pd(_za, _kera, _c3);
            _kerb = _mm256_fmadd_pd(_zb, _kerb, _c3);
            _kerc = _mm256_fmadd_pd(_zc, _kerc, _c3);
            _kerd = _mm256_fmadd_pd(_zd, _kerd, _c3);
        }

        {
            __m256d _c2 = _mm256_load_pd(c72 + 4);
            _kera = _mm256_fmadd_pd(_za, _kera, _c2);
            _kerb = _mm256_fmadd_pd(_zb, _kerb, _c2);
            _kerc = _mm256_fmadd_pd(_zc, _kerc, _c2);
            _kerd = _mm256_fmadd_pd(_zd, _kerd, _c2);
        }

        {
            __m256d _c1 = _mm256_load_pd(c71 + 4);
            _kera = _mm256_fmadd_pd(_za, _kera, _c1);
            _kerb = _mm256_fmadd_pd(_zb, _kerb, _c1);
            _kerc = _mm256_fmadd_pd(_zc, _kerc, _c1);
            _kerd = _mm256_fmadd_pd(_zd, _kerd, _c1);
        }

        {
            __m256d _c0 = _mm256_load_pd(c70 + 4);
            _kera = _mm256_fmadd_pd(_za, _kera, _c0);
            _kerb = _mm256_fmadd_pd(_zb, _kerb, _c0);
            _kerc = _mm256_fmadd_pd(_zc, _kerc, _c0);
            _kerd = _mm256_fmadd_pd(_zd, _kerd, _c0);
        }

        _mm256_store_pd(kers[0] + 4, _kera);
        _mm256_store_pd(kers[1] + 4, _kerb);
        _mm256_store_pd(kers[2] + 4, _kerc);
        _mm256_store_pd(kers[3] + 4, _kerd);

        {
            __m256d _x = _mm256_load_pd(x + 4);
            __m256d _z = _mm256_fmadd_pd(
                _x,
                _mm256_set1_pd(2.0),
                _mm256_set1_pd(z_shift));

            __m128d _z_lo = _mm256_castpd256_pd128(_z);
            _za = _mm256_broadcastsd_pd(_z_lo);
            _zb = _mm256_broadcastsd_pd(_mm_permute_pd(_z_lo, 0x1));

            __m128d _z_hi = _mm256_extractf128_pd(_z, 1);
            _zc = _mm256_broadcastsd_pd(_z_hi);
            _zd = _mm256_broadcastsd_pd(_mm_permute_pd(_z_hi, 0x1));
        }

        {
            __m256d _c9 = _mm256_load_pd(c79 + 4);
            __m256d _c8 = _mm256_load_pd(c78 + 4);
            _kera = _mm256_fmadd_pd(_za, _c9, _c8);
            _kerb = _mm256_fmadd_pd(_zb, _c9, _c8);
            _kerc = _mm256_fmadd_pd(_zc, _c9, _c8);
            _kerd = _mm256_fmadd_pd(_zd, _c9, _c8);
        }

        {
            __m256d _c7 = _mm256_load_pd(c77 + 4);
            _kera = _mm256_fmadd_pd(_za, _kera, _c7);
            _kerb = _mm256_fmadd_pd(_zb, _kerb, _c7);
            _kerc = _mm256_fmadd_pd(_zc, _kerc, _c7);
            _kerd = _mm256_fmadd_pd(_zd, _kerd, _c7);
        }

        {
            __m256d _c6 = _mm256_load_pd(c76 + 4);
            _kera = _mm256_fmadd_pd(_za, _kera, _c6);
            _kerb = _mm256_fmadd_pd(_zb, _kerb, _c6);
            _kerc = _mm256_fmadd_pd(_zc, _kerc, _c6);
            _kerd = _mm256_fmadd_pd(_zd, _kerd, _c6);
        }

        {
            __m256d _c5 = _mm256_load_pd(c75 + 4);
            _kera = _mm256_fmadd_pd(_za, _kera, _c5);
            _kerb = _mm256_fmadd_pd(_zb, _kerb, _c5);
            _kerc = _mm256_fmadd_pd(_zc, _kerc, _c5);
            _kerd = _mm256_fmadd_pd(_zd, _kerd, _c5);
        }

        {
            __m256d _c4 = _mm256_load_pd(c74 + 4);
            _kera = _mm256_fmadd_pd(_za, _kera, _c4);
            _kerb = _mm256_fmadd_pd(_zb, _kerb, _c4);
            _kerc = _mm256_fmadd_pd(_zc, _kerc, _c4);
            _kerd = _mm256_fmadd_pd(_zd, _kerd, _c4);
        }

        {
            __m256d _c3 = _mm256_load_pd(c73 + 4);
            _kera = _mm256_fmadd_pd(_za, _kera, _c3);
            _kerb = _mm256_fmadd_pd(_zb, _kerb, _c3);
            _kerc = _mm256_fmadd_pd(_zc, _kerc, _c3);
            _kerd = _mm256_fmadd_pd(_zd, _kerd, _c3);
        }

        {
            __m256d _c2 = _mm256_load_pd(c72 + 4);
            _kera = _mm256_fmadd_pd(_za, _kera, _c2);
            _kerb = _mm256_fmadd_pd(_zb, _kerb, _c2);
            _kerc = _mm256_fmadd_pd(_zc, _kerc, _c2);
            _kerd = _mm256_fmadd_pd(_zd, _kerd, _c2);
        }

        {
            __m256d _c1 = _mm256_load_pd(c71 + 4);
            _kera = _mm256_fmadd_pd(_za, _kera, _c1);
            _kerb = _mm256_fmadd_pd(_zb, _kerb, _c1);
            _kerc = _mm256_fmadd_pd(_zc, _kerc, _c1);
            _kerd = _mm256_fmadd_pd(_zd, _kerd, _c1);
        }

        {
            __m256d _c0 = _mm256_load_pd(c70 + 4);
            _kera = _mm256_fmadd_pd(_za, _kera, _c0);
            _kerb = _mm256_fmadd_pd(_zb, _kerb, _c0);
            _kerc = _mm256_fmadd_pd(_zc, _kerc, _c0);
            _kerd = _mm256_fmadd_pd(_zd, _kerd, _c0);
        }

        _mm256_store_pd(kers[4] + 4, _kera);
        _mm256_store_pd(kers[5] + 4, _kerb);
        _mm256_store_pd(kers[6] + 4, _kerc);
        _mm256_store_pd(kers[7] + 4, _kerd);

#ifdef _DEBUG
        for (int n = 0; n < npipes; n++) {
            constexpr int w = 7;
            FLT z = 2 * x[n] + w - 1.0;         // scale so local grid offset z in [-1,1]
            FLT test[8];
            for (int i = 0; i < 8; i++) {
                test[i] = c0[i] + z * (c1[i] + z * (c2[i] + z * (c3[i] + z * (c4[i] + z * (c5[i] + z * (c6[i] + z * (c7[i] + z * (c8[i] + z * (c9[i])))))))));
                assert(test[i] == 0.0 ? kers[n][i] == 0.0 : (abs(test[i] - kers[n][i]) / test[i]) < 1e-2);
            }
        }
#endif
    }
}
#endif
#else
template<>
inline void tiled_eval_kernel_vec_Horner<7, true>(FLT** kers, const FLT* x, const spread_opts& opts)
{
    if (!(opts.flags & TF_OMIT_EVALUATE_KERNEL)) {
        for (int n = 0; n < npipes; n++) {
            constexpr int w = 7;
            FLT z = 2 * x[n] + w - 1.0;         // scale so local grid offset z in [-1,1]
            for (int i = 0; i < 8; i++) kers[n][i] = c70[i] + z * (c71[i] + z * (c72[i] + z * (c73[i] + z * (c74[i] + z * (c75[i] + z * (c76[i] + z * (c77[i] + z * (c78[i] + z * (c79[i])))))))));
        }
    }
}
#endif
#endif
#endif