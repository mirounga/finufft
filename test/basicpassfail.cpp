#include <test_defs.h>

// Basic pass-fail test of one routine in library w/ default opts.
// exit code 0 success, failure otherwise. This is useful for brew recipe.
// Works for either single/double, hence use of FLT and CPX.
// Simplified from Amit Moscovitz and example1d1. Barnett 11/1/18.
// Using vectors and default opts, 2/29/20; dual-prec lib 7/3/20.

int main()
{
  BIGINT M = 1e3, N = 1e3;   // defaults: M = # srcs, N = # modes out
  double tol = 1e-5;         // req tol, covers both single & double prec cases
  int isign = +1;            // exponential sign for NUFFT
  static const CPX I = CPX(0.0,1.0);  // imaginary unit. Note: avoid (CPX) cast


  // Make the input data....................................
  // initialize RNG
  VSLStreamStatePtr stream;
  int errcode = vslNewStream(&stream, VSL_BRNG_SFMT19937, 111);


  FLT* x = (FLT*)scalable_aligned_malloc(sizeof(FLT) * M, 64);        // NU pts
  CPX* c = (CPX*)scalable_aligned_malloc(sizeof(CPX) * M, 64);   // strengths 
  CPX* F = (CPX*)scalable_aligned_malloc(sizeof(CPX) * N, 64);   // mode ampls

  // fill x with [-pi,pi)
  errcode = vxRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, M, x, -M_PI, M_PI);

  // fill c with [-1.0,1.0)
  errcode = vxRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, 2 * M, reinterpret_cast<FLT*>(c), -1.0, 1.0);

  // Run it (NULL = default opts) .......................................
  int ier = FINUFFT1D1(M,x,c,isign,tol,N,F,NULL);
  if (ier!=0) {
    printf("basicpassfail: finufft1d1 error (ier=%d)!",ier);
    exit(ier);
  }
  // Check correct math for a single mode...................
  BIGINT n = (BIGINT)(0.37*N);   // choose some mode near the top (N/2)
  CPX Ftest = CPX(0.0,0.0);      // crude exact answer & error check...
  for (BIGINT j=0; j<M; ++j)
    Ftest += c[j] * exp((FLT)isign*I*(FLT)n*x[j]);
  BIGINT nout = n+N/2;           // index in output array for freq mode n
  FLT Finfnrm = 0.0;             // compute inf norm of F...
  for (int m=0; m<N; ++m) {
    FLT aF = abs(F[m]);          // note C++ abs complex type, not C fabs(f)
    if (aF>Finfnrm) Finfnrm=aF;
  }
  FLT relerr = abs(F[nout] - Ftest)/Finfnrm;
  printf("requested tol %.3g: rel err for one mode %.3g\n",tol,relerr);
  scalable_aligned_free(x); scalable_aligned_free(c); scalable_aligned_free(F);
  return (std::isnan(relerr) || relerr > 10.0*tol);    // true reports failure
}
