#include <test_defs.h>

// this enforces recompilation, responding to SINGLE...
#include "directft/dirft3d.cpp"
using namespace std;

const char* help[]={
  "Tester for FINUFFT in 3d, all 3 types, either precision.",
  "",
  "Usage: finufft3d_test Nmodes1 Nmodes2 Nmodes3 Nsrc [tol [debug [spread_sort [upsampfac [errfail]]]]]",
  "\teg:\tfinufft3d_test 100 200 50 1e6 1e-12 0 2 0.0 1e-11",
  "\tnotes:\tif errfail present, exit code 1 if any error > errfail",
  NULL};
// Barnett 2/2/17 onwards.

int main(int argc, char* argv[])
{
  BIGINT M, N1, N2, N3;       // M = # srcs, N1,N2,N3 = # modes
  double w, tol = 1e-6;       // default
  double err, errfail = INFINITY, errmax = 0;
  nufft_opts opts; FINUFFT_DEFAULT_OPTS(&opts);
  //opts.spread_max_sp_size = 3e4; // override test
  //opts.spread_nthr_atomic = 15;  // "
  int isign = +1;             // choose which exponential sign to test
  if (argc<5 || argc>10) {
    for (int i=0; help[i]; ++i)
      fprintf(stderr,"%s\n",help[i]);
    return 2;
  }
  sscanf(argv[1],"%lf",&w); N1 = (BIGINT)w;
  sscanf(argv[2],"%lf",&w); N2 = (BIGINT)w;
  sscanf(argv[3],"%lf",&w); N3 = (BIGINT)w;
  sscanf(argv[4],"%lf",&w); M = (BIGINT)w;
  if (argc>5) sscanf(argv[5],"%lf",&tol);
  if (argc>6) sscanf(argv[6],"%d",&opts.debug);  // can be 0,1 or 2
  opts.spread_debug = (opts.debug>1) ? 1 : 0;  // see output from spreader
  if (argc>7) sscanf(argv[7],"%d",&opts.spread_sort);
  if (argc>8) { sscanf(argv[8],"%lf",&w); opts.upsampfac=(FLT)w; }
  if (argc>9) sscanf(argv[9],"%lf",&errfail);
  
  cout << scientific << setprecision(15);
  BIGINT N = N1*N2*N3;

  // initialize RNG
  VSLStreamStatePtr stream;
  int errcode = vslNewStream(&stream, VSL_BRNG_SFMT19937, 111);

  FLT *x = (FLT *)scalable_aligned_malloc(sizeof(FLT)*M, 64);        // NU pts x coords
  FLT *y = (FLT *)scalable_aligned_malloc(sizeof(FLT)*M, 64);        // NU pts y coords
  FLT *z = (FLT *)scalable_aligned_malloc(sizeof(FLT)*M, 64);        // NU pts z coords
  CPX* c = (CPX*)scalable_aligned_malloc(sizeof(CPX)*M, 64);   // strengths 
  CPX* F = (CPX*)scalable_aligned_malloc(sizeof(CPX)*N, 64);   // mode ampls

  // fill x with [-pi,pi)
  errcode = vxRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, M, x, -M_PI, M_PI);

  // fill y with [-pi,pi)
  errcode = vxRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, M, y, -M_PI, M_PI);

  // fill z with [-pi,pi)
  errcode = vxRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, M, z, -M_PI, M_PI);

  // fill c with [-1.0,1.0)
  errcode = vxRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, 2 * M, reinterpret_cast<FLT*>(c), -1.0, 1.0);

  printf("test 3d type 1:\n"); // -------------- type 1
  CNTime timer; timer.start();
  int ier = FINUFFT3D1(M,x,y,z,c,isign,tol,N1,N2,N3,F,&opts);
  double ti=timer.elapsedsec();
  if (ier>1) {
    printf("error (ier=%d)!\n",ier);
    return ier;
  } else
    printf("     %lld NU pts to (%lld,%lld,%lld) modes in %.3g s \t%.3g NU pts/s\n",
	   (long long)M,(long long)N1,(long long)N2,(long long)N3,ti,M/ti);

  BIGINT nt1 = (BIGINT)(0.37*N1), nt2 = (BIGINT)(0.26*N2), nt3 = (BIGINT)(-0.39*N3);  // choose mode to check

  CPX Ft1 = tbb::parallel_reduce(
      tbb::blocked_range<BIGINT>(0, M),
      CPX(0.0, 0.0),
      [&](tbb::blocked_range<BIGINT>& r, CPX init) -> CPX {
          for (BIGINT j = r.begin(); j < r.end(); j++) {
              init += c[j] * exp(IMA * (nt1 * x[j] + nt2 * y[j] + nt3 * z[j]));
          }

          return init;
      },
      [](CPX a, CPX b) -> CPX {
          return a + b;
      });

  // index in complex F as 1d array...
  BIGINT it = N1/2+nt1 + N1*(N2/2+nt2) + N1*N2*(N3/2+nt3);
  err = abs(Ft1 - F[it])/infnorm(N,F);
  errmax = max(err,errmax);
  printf("\tone mode: rel err in F[%lld,%lld,%lld] is %.3g\n",(long long)nt1,(long long)nt2,(long long)nt3,err);
  if ((int64_t)M*N<=TEST_BIGPROB) {                   // also check vs full direct eval
    CPX* Ft = (CPX*)scalable_aligned_malloc(sizeof(CPX)*N, 64);
    dirft3d1(M,x,y,z,c,isign,N1,N2,N3,Ft);
    err = relerrtwonorm(N,Ft,F);
    errmax = max(err,errmax);
    printf("\tdirft3d: rel l2-err of result F is %.3g\n",err);
    scalable_aligned_free(Ft);
  }
  
  printf("test 3d type 2:\n"); // -------------- type 2

  // fill F with [-1.0,1.0)
  errcode = vxRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, 2 * M, reinterpret_cast<FLT*>(F), -1.0, 1.0);

  timer.restart();
  ier = FINUFFT3D2(M,x,y,z,c,isign,tol,N1,N2,N3,F,&opts);
  ti=timer.elapsedsec();
  if (ier>1) {
    printf("error (ier=%d)!\n",ier);
    return ier;
  } else
    printf("     (%lld,%lld,%lld) modes to %lld NU pts in %.3g s \t%.3g NU pts/s\n",
	   (long long)N1,(long long)N2,(long long)N3,(long long)M,ti,M/ti);

  BIGINT jt = M/2;          // check arbitrary choice of one targ pt
  CPX ct = CPX(0,0);
  BIGINT m=0;
  for (BIGINT m3=-(N3/2); m3<=(N3-1)/2; ++m3)   // loop in F order
    for (BIGINT m2=-(N2/2); m2<=(N2-1)/2; ++m2)
      for (BIGINT m1=-(N1/2); m1<=(N1-1)/2; ++m1)
	ct += F[m++] * exp(IMA*(FLT)isign*(m1*x[jt] + m2*y[jt] + m3*z[jt]));
  err = abs(ct-c[jt])/infnorm(M,c);
  errmax = max(err,errmax);
  printf("\tone targ: rel err in c[%lld] is %.3g\n",(long long)jt,err);
  if ((int64_t)M*N<=TEST_BIGPROB) {                  // also full direct eval
    CPX* ct = (CPX*)scalable_aligned_malloc(sizeof(CPX)*M, 64);
    dirft3d2(M,x,y,z,ct,isign,N1,N2,N3,F);
    err = relerrtwonorm(M,ct,c);
    errmax = max(err,errmax);
    printf("\tdirft3d: rel l2-err of result c is %.3g\n",err);
    scalable_aligned_free(ct);
  }

  printf("test 3d type 3:\n"); // -------------- type 3
  // reuse the strengths c, interpret N as number of targs:

  // fill x with [2.0-pi,2.0+pi)
  errcode = vxRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, M, x, 2.0 - M_PI, 2.0 + M_PI);

  // fill y with [-3.0-pi,-3.0+pi)
  errcode = vxRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, M, y, -3.0 - M_PI, -3.0 + M_PI);

  // fill z with [1.0-pi,1.0+pi)
  errcode = vxRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, M, z, 1.0 - M_PI, 1.0 + M_PI);

  FLT* s = (FLT*)scalable_aligned_malloc(sizeof(FLT) * N, 64);    // targ freqs (1-cmpt)
  FLT* t = (FLT*)scalable_aligned_malloc(sizeof(FLT) * N, 64);    // targ freqs (2-cmpt)
  FLT* u = (FLT*)scalable_aligned_malloc(sizeof(FLT) * N, 64);    // targ freqs (3-cmpt)
  FLT S1 = (FLT)N1 / 2;                   // choose freq range sim to type 1
  FLT S2 = (FLT)N2 / 2;
  FLT S3 = (FLT)N3 / 2;

  // fill s with S1*(1.7 + k/(FLT)N); offset
  errcode = vxRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, N, s, S1 * 0.7, S1 * 2.7);

  // fill t with S2*(-0.5 + k/(FLT)N); offset
  errcode = vxRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, N, t, S2 * (-1.5), S2 * 0.5);

  // fill u with S2*(0.9 + k/(FLT)N); offset
  errcode = vxRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, N, u, S2 * (-0.1), S2 * 1.9);

  timer.restart();
  ier = FINUFFT3D3(M,x,y,z,c,isign,tol,N,s,t,u,F,&opts);
  ti=timer.elapsedsec();
  if (ier>1) {
    printf("error (ier=%d)!\n",ier);
    return ier;
  } else
    printf("\t%lld NU to %lld NU in %.3g s         \t%.3g tot NU pts/s\n",(long long)M,(long long)N,ti,(M+N)/ti);

  BIGINT kt = N/2;          // check arbitrary choice of one targ pt

  CPX Ft3 = tbb::parallel_reduce(
      tbb::blocked_range<BIGINT>(0, M),
      CPX(0.0, 0.0),
      [&](tbb::blocked_range<BIGINT>& r, CPX init) -> CPX {
          for (BIGINT j = r.begin(); j < r.end(); j++) {
              init += c[j] * exp(IMA * (FLT)isign * (s[kt] * x[j] + t[kt] * y[j] + u[kt] * z[j]));
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
    dirft3d3(M,x,y,z,c,isign,N,s,t,u,Ft);       // writes to F
    err = relerrtwonorm(N,Ft,F);
    errmax = max(err,errmax);
    printf("\tdirft3d: rel l2-err of result F is %.3g\n",err);
    //cout<<"s t u, F, Ft, F/Ft:\n"; for (int k=0;k<N;++k) cout<<s[k]<<" "<<t[k]<<" "<<u[k]<<", "<<F[k]<<",\t"<<Ft[k]<<",\t"<<F[k]/Ft[k]<<endl;
    scalable_aligned_free(Ft);
  }

  scalable_aligned_free(x); scalable_aligned_free(y); scalable_aligned_free(z); scalable_aligned_free(c); scalable_aligned_free(F); 
  scalable_aligned_free(s); scalable_aligned_free(t); scalable_aligned_free(u);
  return (errmax>errfail);
}
