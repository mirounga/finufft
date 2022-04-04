#ifndef ONEDIM_FSERIES_KERNEL
#define ONEDIM_FSERIES_KERNEL

#include <dataTypes.h>
#include <spread_opts.h>
extern "C" {
#include "../contrib/legendre_rule_fast.h"
}

#include <tbb/tbb.h>
#include <immintrin.h>

template<class T>
void onedim_fseries_kernel(BIGINT nf, T* fwkerhalf, spread_opts opts)
/*
  Approximates exact Fourier series coeffs of cnufftspread's real symmetric
  kernel, directly via q-node quadrature on Euler-Fourier formula, exploiting
  narrowness of kernel. Uses phase winding for cheap eval on the regular freq
  grid. Note that this is also the Fourier transform of the non-periodized
  kernel. The FT definition is f(k) = int e^{-ikx} f(x) dx. The output has an
  overall prefactor of 1/h, which is needed anyway for the correction, and
  arises because the quadrature weights are scaled for grid units not x units.

  Inputs:
  nf - size of 1d uniform spread grid, must be even.
  opts - spreading opts object, needed to eval kernel (must be already set up)

  Outputs:
  fwkerhalf - real Fourier series coeffs from indices 0 to nf/2 inclusive,
              divided by h = 2pi/n.
              (should be allocated for at least nf/2+1 FLTs)

  Compare onedim_dct_kernel which has same interface, but computes DFT of
  sampled kernel, not quite the same object.

  Barnett 2/7/17. openmp (since slow vs fftw in 1D large-N case) 3/3/18.
  Fixed num_threads 7/20/20
 */
{
    T J2 = opts.nspread / T(2.0);            // J/2, half-width of ker z-support
    // # quadr nodes in z (from 0 to J/2; reflections will be added)...
    int q = (int)(2 + 3.0 * J2);  // not sure why so large? cannot exceed MAX_NQUAD
    int qPadded = 4 * (1 + (q - 1) / 4); // pad q to mult of 4
    T f[MAX_NQUAD];
    double z[2 * MAX_NQUAD], w[2 * MAX_NQUAD];
    legendre_compute_glr(2 * q, z, w);        // only half the nodes used, eg on (0,1)
    std::complex<T> a[MAX_NQUAD];
    for (int n = 0; n < q; ++n) {               // set up nodes z_n and vals f_n
        z[n] *= J2;                         // rescale nodes
        f[n] = J2 * (T)w[n] * evaluate_kernel((T)z[n], opts); // vals & quadr wei
        a[n] = exp(2 * PI * IMA * (T)(nf / 2 - z[n]) / (T)nf);  // phase winding rates
    }
    BIGINT nout = nf / 2 + 1;                   // how many values we're writing to

    tbb::parallel_for(tbb::blocked_range<BIGINT>(0, nout, 10000), // loop along output array
        [&](const tbb::blocked_range<BIGINT>& r) {
            const BIGINT begin = r.begin(), end = r.end();

            std::complex<T> aj[MAX_NQUAD];    // phase rotator for this thread
            for (int n = 0; n < q; ++n)
                aj[n] = pow(a[n], begin);

            for (BIGINT j = begin; j < end; ++j) {
                T x = 0.0;                      // accumulator for answer at this j
                for (int n = 0; n < q; ++n) {
                    x += f[n] * 2 * real(aj[n]);      // include the negative freq
                    aj[n] *= a[n];                  // wind the phases
                }

                fwkerhalf[j] = x;
            }
        });
}

#ifdef __AVX2__
template<>
void onedim_fseries_kernel<double>(BIGINT nf, double* fwkerhalf, spread_opts opts)
{
    double J2 = opts.nspread / double(2.0);            // J/2, half-width of ker z-support
    // # quadr nodes in z (from 0 to J/2; reflections will be added)...
    int q = (int)(2 + 3.0 * J2);  // not sure why so large? cannot exceed MAX_NQUAD
    int qPadded = 4 * (1 + (q - 1) / 4); // pad q to mult of 4
    double f[MAX_NQUAD];
    double z[2 * MAX_NQUAD], w[2 * MAX_NQUAD];
    legendre_compute_glr(2 * q, z, w);        // only half the nodes used, eg on (0,1)
    std::complex<double> a[MAX_NQUAD];
    for (int n = 0; n < q; ++n) {               // set up nodes z_n and vals f_n
        z[n] *= J2;                         // rescale nodes
        f[n] = J2 * (double)w[n] * evaluate_kernel((double)z[n], opts); // vals & quadr wei
        a[n] = exp(std::complex<double>(0.0, 2.0 * PI * (double)(nf / 2 - z[n]) / (double)nf));  // phase winding rates
    }
    BIGINT nout = nf / 2 + 1;                   // how many values we're writing to

    alignas(64) double bj_re[16 * MAX_NQUAD];    // 0-based phase rotator
    alignas(64) double bj_im[16 * MAX_NQUAD];

    alignas(64) double an_re[4 * MAX_NQUAD];
    alignas(64) double an_im[4 * MAX_NQUAD];

    double* pXre = bj_re;
    double* pXim = bj_im;
    double* pYre = an_re;
    double* pYim = an_im;
    for (int n = 0; n < q; ++n) {
        std::complex<double> aj(1.0, 0.0);

        for (int d = 0; d < 16; d++) {
            *pXre++ = 2.0 * aj.real();
            *pXim++ = 2.0 * aj.imag();

            aj *= a[n];
        }

        for (int d = 0; d < 4; d++) {
            *pYre++ = aj.real();
            *pYim++ = aj.imag();
        }
    }

    tbb::parallel_for(tbb::blocked_range<BIGINT>(0, nout, 10000), // loop along output array
        [fwkerhalf, a, f, q, bj_re, bj_im, an_re, an_im](const tbb::blocked_range<BIGINT>& r) {
            const BIGINT begin = r.begin(), end = r.end();

            alignas(64) double aj_re[16 * MAX_NQUAD];    // phase rotator for this thread
            alignas(64) double aj_im[16 * MAX_NQUAD];

            const double* pBre = bj_re;
            const double* pBim = bj_im;
            double* pAre = aj_re;
            double* pAim = aj_im;

            for (int n = 0; n < q; ++n) {
                std::complex<double> ap = std::pow(a[n], begin);

                __m256d _ap_re = _mm256_set1_pd(ap.real());
                __m256d _ap_im = _mm256_set1_pd(ap.imag());

                {
                    __m256d _bj_re0 = _mm256_loadu_pd(pBre + 0);
                    __m256d _bj_im0 = _mm256_loadu_pd(pBim + 0);
                    __m256d _bj_re1 = _mm256_loadu_pd(pBre + 4);
                    __m256d _bj_im1 = _mm256_loadu_pd(pBim + 4);

                    __m256d _aj_re0 = _mm256_fmsub_pd(_bj_re0, _ap_re, _mm256_mul_pd(_bj_im0, _ap_im));
                    __m256d _aj_im0 = _mm256_fmadd_pd(_bj_re0, _ap_im, _mm256_mul_pd(_bj_im0, _ap_re));
                    __m256d _aj_re1 = _mm256_fmsub_pd(_bj_re1, _ap_re, _mm256_mul_pd(_bj_im1, _ap_im));
                    __m256d _aj_im1 = _mm256_fmadd_pd(_bj_re1, _ap_im, _mm256_mul_pd(_bj_im1, _ap_re));

                    _mm256_store_pd(pAre + 0, _aj_re0);
                    _mm256_store_pd(pAim + 0, _aj_im0);
                    _mm256_store_pd(pAre + 4, _aj_re1);
                    _mm256_store_pd(pAim + 4, _aj_im1);
                }
                {
                    __m256d _bj_re2 = _mm256_loadu_pd(pBre + 8);
                    __m256d _bj_im2 = _mm256_loadu_pd(pBim + 8);
                    __m256d _bj_re3 = _mm256_loadu_pd(pBre + 12);
                    __m256d _bj_im3 = _mm256_loadu_pd(pBim + 12);

                    __m256d _aj_re2 = _mm256_fmsub_pd(_bj_re2, _ap_re, _mm256_mul_pd(_bj_im2, _ap_im));
                    __m256d _aj_im2 = _mm256_fmadd_pd(_bj_re2, _ap_im, _mm256_mul_pd(_bj_im2, _ap_re));
                    __m256d _aj_re3 = _mm256_fmsub_pd(_bj_re3, _ap_re, _mm256_mul_pd(_bj_im3, _ap_im));
                    __m256d _aj_im3 = _mm256_fmadd_pd(_bj_re3, _ap_im, _mm256_mul_pd(_bj_im3, _ap_re));

                    _mm256_store_pd(pAre + 8, _aj_re2);
                    _mm256_store_pd(pAim + 8, _aj_im2);
                    _mm256_store_pd(pAre + 12, _aj_re3);
                    _mm256_store_pd(pAim + 12, _aj_im3);
                }

                pBre += 16;
                pBim += 16;
                pAre += 16;
                pAim += 16;
            }

            BIGINT end16 = begin + ((end - begin) & ~0x0fll);

            for (BIGINT j = begin; j < end16; j += 16) {
                __m256d _x0 = _mm256_setzero_pd(); // accumulator for answer at this j
                __m256d _x1 = _mm256_setzero_pd();
                __m256d _x2 = _mm256_setzero_pd();
                __m256d _x3 = _mm256_setzero_pd();

                pAre = aj_re;
                pAim = aj_im;
                const double * pNre = an_re;
                const double * pNim = an_im;
                for (int n = 0; n < q; ++n) {
                    __m256d _fn = _mm256_set1_pd(f[n]);

                    __m256d _an_re = _mm256_loadu_pd(pNre);
                    __m256d _an_im = _mm256_loadu_pd(pNim);

                    {
                        __m256d _aj_re0 = _mm256_load_pd(pAre + 0);
                        __m256d _aj_im0 = _mm256_load_pd(pAim + 0);
                        __m256d _aj_re1 = _mm256_load_pd(pAre + 4);
                        __m256d _aj_im1 = _mm256_load_pd(pAim + 4);

                        // include the negative freq
                        _x0 = _mm256_fmadd_pd(_fn, _aj_re0, _x0);
                        _x1 = _mm256_fmadd_pd(_fn, _aj_re1, _x1);

                        // wind the phases
                        __m256d _ak_re0 = _mm256_fmsub_pd(_aj_re0, _an_re, _mm256_mul_pd(_aj_im0, _an_im));
                        __m256d _ak_im0 = _mm256_fmadd_pd(_aj_re0, _an_im, _mm256_mul_pd(_aj_im0, _an_re));
                        __m256d _ak_re1 = _mm256_fmsub_pd(_aj_re1, _an_re, _mm256_mul_pd(_aj_im1, _an_im));
                        __m256d _ak_im1 = _mm256_fmadd_pd(_aj_re1, _an_im, _mm256_mul_pd(_aj_im1, _an_re));

                        _mm256_store_pd(pAre + 0, _ak_re0);
                        _mm256_store_pd(pAim + 0, _ak_im0);
                        _mm256_store_pd(pAre + 4, _ak_re1);
                        _mm256_store_pd(pAim + 4, _ak_im1);
                    }
                    {
                        __m256d _aj_re2 = _mm256_load_pd(pAre + 8);
                        __m256d _aj_im2 = _mm256_load_pd(pAim + 8);
                        __m256d _aj_re3 = _mm256_load_pd(pAre + 12);
                        __m256d _aj_im3 = _mm256_load_pd(pAim + 12);

                        // include the negative freq
                        _x2 = _mm256_fmadd_pd(_fn, _aj_re2, _x2);
                        _x3 = _mm256_fmadd_pd(_fn, _aj_re3, _x3);

                        // wind the phases
                        __m256d _ak_re2 = _mm256_fmsub_pd(_aj_re2, _an_re, _mm256_mul_pd(_aj_im2, _an_im));
                        __m256d _ak_im2 = _mm256_fmadd_pd(_aj_re2, _an_im, _mm256_mul_pd(_aj_im2, _an_re));
                        __m256d _ak_re3 = _mm256_fmsub_pd(_aj_re3, _an_re, _mm256_mul_pd(_aj_im3, _an_im));
                        __m256d _ak_im3 = _mm256_fmadd_pd(_aj_re3, _an_im, _mm256_mul_pd(_aj_im3, _an_re));

                        _mm256_store_pd(pAre + 8, _ak_re2);
                        _mm256_store_pd(pAim + 8, _ak_im2);
                        _mm256_store_pd(pAre + 12, _ak_re3);
                        _mm256_store_pd(pAim + 12, _ak_im3);
                    }

                    pAre += 16;
                    pAim += 16;
                    pNre += 4;
                    pNim += 4;
                }

                _mm256_store_pd(fwkerhalf + j + 0, _x0);
                _mm256_store_pd(fwkerhalf + j + 4, _x1);
                _mm256_store_pd(fwkerhalf + j + 8, _x2);
                _mm256_store_pd(fwkerhalf + j + 12, _x3);
            }

            for (BIGINT j = end16, jj = 0; j < end; ++j, ++jj) {
                pAre = aj_re + jj;
                double x = 0.0;                      // accumulator for answer at this j
                for (int n = 0; n < q; ++n) {
                    x += f[n] * pAre[0];      // include the negative freq
                    pAre += 16;
                }
                fwkerhalf[j] = x;
            }
        });
}

template<>
void onedim_fseries_kernel<float>(BIGINT nf, float* fwkerhalf, spread_opts opts)
{
    float J2 = opts.nspread / float(2.0);            // J/2, half-width of ker z-support
    // # quadr nodes in z (from 0 to J/2; reflections will be added)...
    int q = (int)(2 + 3.0 * J2);  // not sure why so large? cannot exceed MAX_NQUAD
    int qPadded = 4 * (1 + (q - 1) / 4); // pad q to mult of 4
    float f[MAX_NQUAD];
    double z[2 * MAX_NQUAD], w[2 * MAX_NQUAD];
    legendre_compute_glr(2 * q, z, w);        // only half the nodes used, eg on (0,1)
    std::complex<float> a[MAX_NQUAD];
    for (int n = 0; n < q; ++n) {               // set up nodes z_n and vals f_n
        z[n] *= J2;                         // rescale nodes
        f[n] = J2 * (float)w[n] * evaluate_kernel((float)z[n], opts); // vals & quadr wei
        a[n] = exp(std::complex<float>(0.0, 2.0 * PI * (float)(nf / 2 - z[n]) / (float)nf));  // phase winding rates
    }
    BIGINT nout = nf / 2 + 1;                   // how many values we're writing to

    alignas(64) float bj_re[32 * MAX_NQUAD];    // 0-based phase rotator
    alignas(64) float bj_im[32 * MAX_NQUAD];

    alignas(64) float an_re[8 * MAX_NQUAD];
    alignas(64) float an_im[8 * MAX_NQUAD];

    float* pXre = bj_re;
    float* pXim = bj_im;
    float* pYre = an_re;
    float* pYim = an_im;
    for (int n = 0; n < q; ++n) {
        std::complex<float> aj(1.0f, 0.0f);

        for (int d = 0; d < 32; d++) {
            *pXre++ = 2.0f * aj.real();
            *pXim++ = 2.0f * aj.imag();

            aj *= a[n];
        }

        for (int d = 0; d < 8; d++) {
            *pYre++ = aj.real();
            *pYim++ = aj.imag();
        }
    }

    tbb::parallel_for(tbb::blocked_range<BIGINT>(0, nout, 10000), // loop along output array
        [fwkerhalf, a, f, q, bj_re, bj_im, an_re, an_im](const tbb::blocked_range<BIGINT>& r) {
            const BIGINT begin = r.begin(), end = r.end();

            alignas(64) float aj_re[32 * MAX_NQUAD];    // phase rotator for this thread
            alignas(64) float aj_im[32 * MAX_NQUAD];

            const float* pBre = bj_re;
            const float* pBim = bj_im;
            float* pAre = aj_re;
            float* pAim = aj_im;

            for (int n = 0; n < q; ++n) {
                std::complex<float> ap = std::pow(a[n], static_cast<float>(begin));

                __m256 _ap_re = _mm256_set1_ps(ap.real());
                __m256 _ap_im = _mm256_set1_ps(ap.imag());

                {
                    __m256 _bj_re0 = _mm256_loadu_ps(pBre + 0);
                    __m256 _bj_im0 = _mm256_loadu_ps(pBim + 0);
                    __m256 _bj_re1 = _mm256_loadu_ps(pBre + 8);
                    __m256 _bj_im1 = _mm256_loadu_ps(pBim + 8);

                    __m256 _aj_re0 = _mm256_fmsub_ps(_bj_re0, _ap_re, _mm256_mul_ps(_bj_im0, _ap_im));
                    __m256 _aj_im0 = _mm256_fmadd_ps(_bj_re0, _ap_im, _mm256_mul_ps(_bj_im0, _ap_re));
                    __m256 _aj_re1 = _mm256_fmsub_ps(_bj_re1, _ap_re, _mm256_mul_ps(_bj_im1, _ap_im));
                    __m256 _aj_im1 = _mm256_fmadd_ps(_bj_re1, _ap_im, _mm256_mul_ps(_bj_im1, _ap_re));

                    _mm256_store_ps(pAre + 0, _aj_re0);
                    _mm256_store_ps(pAim + 0, _aj_im0);
                    _mm256_store_ps(pAre + 8, _aj_re1);
                    _mm256_store_ps(pAim + 8, _aj_im1);
                }
                {
                    __m256 _bj_re2 = _mm256_loadu_ps(pBre + 16);
                    __m256 _bj_im2 = _mm256_loadu_ps(pBim + 16);
                    __m256 _bj_re3 = _mm256_loadu_ps(pBre + 24);
                    __m256 _bj_im3 = _mm256_loadu_ps(pBim + 24);

                    __m256 _aj_re2 = _mm256_fmsub_ps(_bj_re2, _ap_re, _mm256_mul_ps(_bj_im2, _ap_im));
                    __m256 _aj_im2 = _mm256_fmadd_ps(_bj_re2, _ap_im, _mm256_mul_ps(_bj_im2, _ap_re));
                    __m256 _aj_re3 = _mm256_fmsub_ps(_bj_re3, _ap_re, _mm256_mul_ps(_bj_im3, _ap_im));
                    __m256 _aj_im3 = _mm256_fmadd_ps(_bj_re3, _ap_im, _mm256_mul_ps(_bj_im3, _ap_re));

                    _mm256_store_ps(pAre + 16, _aj_re2);
                    _mm256_store_ps(pAim + 16, _aj_im2);
                    _mm256_store_ps(pAre + 24, _aj_re3);
                    _mm256_store_ps(pAim + 24, _aj_im3);
                }

                pBre += 32;
                pBim += 32;
                pAre += 32;
                pAim += 32;
            }

            BIGINT end32 = begin + ((end - begin) & ~0x1fll);

            for (BIGINT j = begin; j < end32; j += 32) {
                __m256 _x0 = _mm256_setzero_ps(); // accumulator for answer at this j
                __m256 _x1 = _mm256_setzero_ps();
                __m256 _x2 = _mm256_setzero_ps();
                __m256 _x3 = _mm256_setzero_ps();

                pAre = aj_re;
                pAim = aj_im;
                const float * pNre = an_re;
                const float * pNim = an_im;
                for (int n = 0; n < q; ++n) {
                    __m256 _fn = _mm256_set1_ps(f[n]);

                    __m256 _an_re = _mm256_loadu_ps(pNre);
                    __m256 _an_im = _mm256_loadu_ps(pNim);

                    {
                        __m256 _aj_re0 = _mm256_load_ps(pAre + 0);
                        __m256 _aj_im0 = _mm256_load_ps(pAim + 0);
                        __m256 _aj_re1 = _mm256_load_ps(pAre + 8);
                        __m256 _aj_im1 = _mm256_load_ps(pAim + 8);

                        // include the negative freq
                        _x0 = _mm256_fmadd_ps(_fn, _aj_re0, _x0);
                        _x1 = _mm256_fmadd_ps(_fn, _aj_re1, _x1);

                        // wind the phases
                        __m256 _ak_re0 = _mm256_fmsub_ps(_aj_re0, _an_re, _mm256_mul_ps(_aj_im0, _an_im));
                        __m256 _ak_im0 = _mm256_fmadd_ps(_aj_re0, _an_im, _mm256_mul_ps(_aj_im0, _an_re));
                        __m256 _ak_re1 = _mm256_fmsub_ps(_aj_re1, _an_re, _mm256_mul_ps(_aj_im1, _an_im));
                        __m256 _ak_im1 = _mm256_fmadd_ps(_aj_re1, _an_im, _mm256_mul_ps(_aj_im1, _an_re));

                        _mm256_store_ps(pAre + 0, _ak_re0);
                        _mm256_store_ps(pAim + 0, _ak_im0);
                        _mm256_store_ps(pAre + 8, _ak_re1);
                        _mm256_store_ps(pAim + 8, _ak_im1);
                    }
                    {
                        __m256 _aj_re2 = _mm256_load_ps(pAre + 16);
                        __m256 _aj_im2 = _mm256_load_ps(pAim + 16);
                        __m256 _aj_re3 = _mm256_load_ps(pAre + 24);
                        __m256 _aj_im3 = _mm256_load_ps(pAim + 24);

                        // include the negative freq
                        _x2 = _mm256_fmadd_ps(_fn, _aj_re2, _x2);
                        _x3 = _mm256_fmadd_ps(_fn, _aj_re3, _x3);

                        // wind the phases
                        __m256 _ak_re2 = _mm256_fmsub_ps(_aj_re2, _an_re, _mm256_mul_ps(_aj_im2, _an_im));
                        __m256 _ak_im2 = _mm256_fmadd_ps(_aj_re2, _an_im, _mm256_mul_ps(_aj_im2, _an_re));
                        __m256 _ak_re3 = _mm256_fmsub_ps(_aj_re3, _an_re, _mm256_mul_ps(_aj_im3, _an_im));
                        __m256 _ak_im3 = _mm256_fmadd_ps(_aj_re3, _an_im, _mm256_mul_ps(_aj_im3, _an_re));

                        _mm256_store_ps(pAre + 16, _ak_re2);
                        _mm256_store_ps(pAim + 16, _ak_im2);
                        _mm256_store_ps(pAre + 24, _ak_re3);
                        _mm256_store_ps(pAim + 24, _ak_im3);
                    }

                    pAre += 32;
                    pAim += 32;
                    pNre += 8;
                    pNim += 8;
                }

                _mm256_store_ps(fwkerhalf + j + 0, _x0);
                _mm256_store_ps(fwkerhalf + j + 8, _x1);
                _mm256_store_ps(fwkerhalf + j + 16, _x2);
                _mm256_store_ps(fwkerhalf + j + 24, _x3);
            }

            for (BIGINT j = end32, jj = 0; j < end; ++j, ++jj) {// loop along output array
                pAre = aj_re + jj;
                float x = 0.0f;                      // accumulator for answer at this j
                for (int n = 0; n < q; ++n) {
                    x += f[n] * pAre[0];      // include the negative freq
                    pAre += 32;
                }
                fwkerhalf[j] = x;
            }
        });

}
#endif
#endif