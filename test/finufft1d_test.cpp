#include <test_defs.h>

// this enforces recompilation, responding to SINGLE...
#include "directft/dirft1d.cpp"
using namespace std;

const char* help[]={
  "Tester for FINUFFT in 1d, all 3 types, either precision.",
  "",
  "Usage: finufft1d_test Nmodes Nsrc [tol [debug [spread_sort [upsampfac [errfail]]]]]",
  "\teg:\tfinufft1d_test 1e6 1e6 1e-6 1 2 2.0 1e-5",
  "\tnotes:\tif errfail present, exit code 1 if any error > errfail",
  NULL};
// Barnett 1/22/17 onwards

int main(int argc, char* argv[])
{
  BIGINT M, N;   // M = # srcs, N = # modes out
  double w, tol = 1e-6;         // default
  double err, errfail = INFINITY, errmax = 0;
  nufft_opts opts; FINUFFT_DEFAULT_OPTS(&opts);  // put defaults in opts
  int isign = +1;            // choose which exponential sign to test
  if (argc<3 || argc>8) {
    for (int i=0; help[i]; ++i)
      fprintf(stderr,"%s\n",help[i]);
    return 2;
  }
  sscanf(argv[1],"%lf",&w); N = (BIGINT)w;
  sscanf(argv[2],"%lf",&w); M = (BIGINT)w;
  if (argc>3) sscanf(argv[3],"%lf",&tol);
  if (argc>4) sscanf(argv[4],"%d",&opts.debug);
  opts.spread_debug = (opts.debug>1) ? 1 : 0;  // see output from spreader
  if (argc>5) sscanf(argv[5],"%d",&opts.spread_sort);
  if (argc>6) { sscanf(argv[6],"%lf",&w); opts.upsampfac=(FLT)w; }
  if (argc>7) sscanf(argv[7],"%lf",&errfail);
  
  cout << scientific << setprecision(15);

  // initialize RNG
  VSLStreamStatePtr stream;
  int errcode = vslNewStream(&stream, VSL_BRNG_SFMT19937, 111);

  FLT *x = (FLT*)scalable_aligned_malloc(sizeof(FLT)*M, 64);        // NU pts
  CPX* c = (CPX*)scalable_aligned_malloc(sizeof(CPX)*M, 64);   // strengths 
  CPX* F = (CPX*)scalable_aligned_malloc(sizeof(CPX)*N, 64);   // mode ampls

  // fill x with [-pi,pi)
  errcode = vxRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, M, x, -M_PI, M_PI);

  // fill c with [-1.0,1.0)
  errcode = vxRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, 2 * M, reinterpret_cast<FLT*>(c), -1.0, 1.0);

  //for (BIGINT j=0; j<M; ++j) x[j] = 0.999 * PI*randm11();  // avoid ends
  //for (BIGINT j=0; j<M; ++j) x[j] = PI*(2*j/(FLT)M-1);  // test a grid

  printf("test 1d type 1:\n"); // -------------- type 1
  CNTime timer; timer.start();
  int ier = FINUFFT1D1(M,x,c,isign,tol,N,F,&opts);
  //for (int j=0;j<N;++j) cout<<F[j]<<endl;
  double t=timer.elapsedsec();
  if (ier>1) {
    printf("error (ier=%d)!\n",ier);
    return ier;
  } else
    printf("\t%lld NU pts to %lld modes in %.3g s \t%.3g NU pts/s\n",(long long)M,(long long)N,t,M/t);

  BIGINT nt = (BIGINT)(0.37*N);   // check arb choice of mode near the top (N/2)

  CPX Ft1 = tbb::parallel_reduce(
      tbb::blocked_range<BIGINT>(0, M),
      CPX(0.0, 0.0),
      [&](tbb::blocked_range<BIGINT>& r, CPX init) -> CPX {
          for (BIGINT j = r.begin(); j < r.end(); j++) {
              init += c[j] * exp(IMA * ((FLT)(isign * nt)) * x[j]);
          }

          return init;
      },
      [](CPX a, CPX b) -> CPX {
          return a + b;
      });

  err = abs(Ft1 - F[N/2+nt])/infnorm(N,F);
  printf("\tone mode: rel err in F[%lld] is %.3g\n",(long long)nt,err);
  errmax = max(err,errmax);
  if (((int64_t)M)*N<=TEST_BIGPROB) {                  // also full direct eval
    CPX* Ft = (CPX*)scalable_aligned_malloc(sizeof(CPX)*N, 64);
    dirft1d1(M,x,c,isign,N,Ft);
    err = relerrtwonorm(N,Ft,F);
    errmax = max(err,errmax);
    printf("\tdirft1d: rel l2-err of result F is %.3g\n",err);
    scalable_aligned_free(Ft);
  }

  printf("test 1d type 2:\n"); // -------------- type 2

  // fill F with [-1.0,1.0)
  errcode = vxRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, 2 * N, reinterpret_cast<FLT*>(F), -1.0, 1.0);

  timer.restart();
  ier = FINUFFT1D2(M,x,c,isign,tol,N,F,&opts);
  //cout<<"c:\n"; for (int j=0;j<M;++j) cout<<c[j]<<endl;
  t=timer.elapsedsec();
  if (ier>1) {
    printf("error (ier=%d)!\n",ier);
    return ier;
  } else
    printf("\t%lld modes to %lld NU pts in %.3g s \t%.3g NU pts/s\n",(long long)N,(long long)M,t,M/t);

  BIGINT jt = M/2;          // check arbitrary choice of one targ pt

  CPX ct = CPX(0, 0);
  for (BIGINT m=0, m1 = -(N/2); m1 <= (N - 1) / 2; ++m1)
      ct += F[m++] * exp(IMA * ((FLT)(isign * m1)) * x[jt]);   // crude direct

  err = abs(ct - c[jt]) / infnorm(M, c);
  errmax = max(err,errmax);
  printf("\tone targ: rel err in c[%lld] is %.3g\n",(long long)jt,err);
  if (((int64_t)M)*N<=TEST_BIGPROB) {                  // also full direct eval
    CPX* ct = (CPX*)scalable_aligned_malloc(sizeof(CPX)*M, 64);
    dirft1d2(M,x,ct,isign,N,F);
    err = relerrtwonorm(M,ct,c);
    errmax = max(err,errmax);
    printf("\tdirft1d: rel l2-err of result c is %.3g\n",err);
    //cout<<"c/ct:\n"; for (int j=0;j<M;++j) cout<<c[j]/ct[j]<<endl;
    scalable_aligned_free(ct);
  }

  printf("test 1d type 3:\n"); // -------------- type 3
  // reuse the strengths c, interpret N as number of targs:
  // fill x with [2.0-pi,2.0+pi)
  errcode = vxRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, M, x, 2.0-M_PI, 2.0+M_PI);

  FLT* s = (FLT*)scalable_aligned_malloc(sizeof(FLT)*N, 64);    // targ freqs
  FLT S = (FLT)N/2;                   // choose freq range sim to type 1

  // fill s with S*(1.7 + k/(FLT)N); offset
  errcode = vxRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, N, s, S*0.7, S*2.7);

  timer.restart();
  ier = FINUFFT1D3(M,x,c,isign,tol,N,s,F,&opts);
  t=timer.elapsedsec();
  if (ier>0) {
    printf("error (ier=%d)!\n",ier);
    return ier;
  } else
    printf("\t%lld NU to %lld NU in %.3g s        \t%.3g tot NU pts/s\n",(long long)M,(long long)N,t,(M+N)/t);

  BIGINT kt = N/2;          // check arbitrary choice of one targ pt

  CPX Ft3 = tbb::parallel_reduce(
      tbb::blocked_range<BIGINT>(0, M),
      CPX(0.0, 0.0),
      [&](tbb::blocked_range<BIGINT>& r, CPX init) -> CPX {
          for (BIGINT j = r.begin(); j < r.end(); j++) {
              init += c[j] * exp(IMA * (FLT)isign * s[kt] * x[j]);
          }

          return init;
      },
      [](CPX a, CPX b) -> CPX {
          return a + b;
      });

  err = abs(Ft3 - F[kt]) / infnorm(N, F);
  errmax = max(err,errmax);
  printf("\tone targ: rel err in F[%lld] is %.3g\n",(long long)kt,err);
  if (((int64_t)M)*N<=TEST_BIGPROB) {                  // also full direct eval
    CPX* Ft = (CPX*)scalable_aligned_malloc(sizeof(CPX)*N, 64);
    dirft1d3(M,x,c,isign,N,s,Ft);       // writes to F
    err = relerrtwonorm(N,Ft,F);
    errmax = max(err,errmax);
    printf("\tdirft1d: rel l2-err of result F is %.3g\n",err);
    //cout<<"s, F, Ft:\n"; for (int k=0;k<N;++k) cout<<s[k]<<" "<<F[k]<<"\t"<<Ft[k]<<"\t"<<F[k]/Ft[k]<<endl;
    scalable_aligned_free(Ft);
  }

  errcode = vslDeleteStream(&stream);

  scalable_aligned_free(x); scalable_aligned_free(c); scalable_aligned_free(F); scalable_aligned_free(s);
  return (errmax>errfail);
}
