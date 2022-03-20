#ifndef ONEDIM_FSERIES_KERNEL
#define ONEDIM_FSERIES_KERNEL

#include <dataTypes.h>
#include <spread_opts.h>
extern "C" {
#include "../contrib/legendre_rule_fast.h"
}

#include <tbb/tbb.h>

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

    std::complex<T> aj[MAX_NQUAD];    // phase rotator for this thread
    for (int n = 0; n < q; ++n)
        aj[n] = std::complex<T>(1, 0);

    tbb::parallel_for(tbb::blocked_range<BIGINT>(0, nout, 10000), // loop along output array
        [&](const tbb::blocked_range<BIGINT>& r) {
            for (BIGINT j = r.begin(); j < r.end(); ++j) {
                T x = 0.0;                      // accumulator for answer at this j
                for (int n = 0; n < q; ++n) {
                    x += f[n] * 2 * real(aj[n]);      // include the negative freq
                    aj[n] *= a[n];                  // wind the phases
                }
                fwkerhalf[j] = x;
            }
        });
}

#endif