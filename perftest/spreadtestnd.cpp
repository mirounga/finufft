#include <spreadinterp.h>
#include <defs.h>
#include <utils.h>
#include <utils_precindep.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include <tbb/tbb.h>
#include <mkl_vsl.h>

#undef max
#include <algorithm>

void usage()
{
  printf("usage: spreadtestnd dims [M N [tol [sort [flags [debug [kerpad [kerevalmeth [upsampfac]]]]]]]]\n\twhere dims=1,2 or 3\n\tM=# nonuniform pts\n\tN=# uniform pts\n\ttol=requested accuracy\n\tsort=0 (don't sort NU pts), 1 (do), or 2 (maybe sort; default)\n\tflags: expert timing flags, 0 is default (see spreadinterp.h)\n\tdebug=0 (less text out), 1 (more), 2 (lots)\n\tkerpad=0 (no pad to mult of 4), 1 (do, for kerevalmeth=0 only)\n\tkerevalmeth=0 (direct), 1 (Horner ppval)\n\tupsampfac>1; 2 or 1.25 for Horner\n\nexample: ./spreadtestnd 1 1e6 1e6 1e-6 2 0 1\n");
}

int main(int argc, char* argv[])
/* Test executable for the 1D, 2D, or 3D C++ spreader, both directions.
 * It checks speed, and basic correctness via the grid sum of the result.
 * See usage() for usage.  Note it currently tests only pirange=0, which is not
 * the use case in finufft, and can differ in speed (see devel/foldrescale*)
 *
 * Example: spreadtestnd 3 8e6 8e6 1e-6 2 0 1
 *
 * Compilation (also check ../makefile):
 *    g++ spreadtestnd.cpp ../src/spreadinterp.o ../src/utils.o -o spreadtestnd -fPIC -Ofast -funroll-loops -fopenmp
 *
 * Magland; expanded by Barnett 1/14/17. Better cmd line args 3/13/17
 * indep setting N 3/27/17. parallel rand() & sort flag 3/28/17
 * timing_flags 6/14/17. debug control 2/8/18. sort=2 opt 3/5/18, pad 4/24/18.
 * ier=1 warning not error, upsampfac 6/14/20.
 */
{
  int d = 3;            // Cmd line args & their defaults:  default #dims
  double w, tol = 1e-6; // default (eg 1e-6 has nspread=7)
  BIGINT M = 1e6;       // default # NU pts
  BIGINT roughNg = 1e6; // default # U pts
  int sort = 2;         // spread_sort
  int flags = 0;        // default
  int debug = 0;        // default
  int kerpad = 0;       // default
  int kerevalmeth = 1;  // default: Horner
  FLT upsampfac = 2.0;  // standard
  
  if (argc<2 || argc==3 || argc>11) {
    usage(); return (argc>1);
  }
  sscanf(argv[1],"%d",&d);
  if (d<1 || d>3) {
    printf("d must be 1, 2 or 3!\n"); usage(); return 1;
  }
  if (argc>2) {
    sscanf(argv[2],"%lf",&w); M = (BIGINT)w;       // to read "1e6" right!
    if (M<1) {
      printf("M (# NU pts) must be positive!\n"); usage(); return 1;
    }
    sscanf(argv[3],"%lf",&w); roughNg = (BIGINT)w;
    if (roughNg<1) {
      printf("N (# U pts) must be positive!\n"); usage(); return 1;
    }
  }
  if (argc>4) sscanf(argv[4],"%lf",&tol);
  if (argc>5) {
    sscanf(argv[5],"%d",&sort);
    if ((sort!=0) && (sort!=1) && (sort!=2)) {
      printf("sort must be 0, 1 or 2!\n"); usage(); return 1;
    }
  }
  if (argc>6)
    sscanf(argv[6],"%d",&flags);
  if (argc>7) {
    sscanf(argv[7],"%d",&debug);
    if ((debug<0) || (debug>2)) {
      printf("debug must be 0, 1 or 2!\n"); usage(); return 1;
    }
  }
  if (argc>8) {
    sscanf(argv[8],"%d",&kerpad);
    if ((kerpad<0) || (kerpad>1)) {
      printf("kerpad must be 0 or 1!\n"); usage(); return 1;
    }
  }
  if (argc>9) {
    sscanf(argv[9],"%d",&kerevalmeth);
    if ((kerevalmeth<0) || (kerevalmeth>1)) {
      printf("kerevalmeth must be 0 or 1!\n"); usage(); return 1;
    }
  }
  if (argc>10) {
    sscanf(argv[10],"%lf",&w); upsampfac = (FLT)w;
    if (upsampfac<=1.0) {
      printf("upsampfac must be >1.0!\n"); usage(); return 1;
    }
  }

  //tbb::global_control thread_limit(tbb::global_control::max_allowed_parallelism, 1);

  int dodir1 = true;                        // control if dir=1 tested at all
  BIGINT N = (BIGINT)round(pow(roughNg,1.0/d));     // Fourier grid size per dim
  BIGINT Ng = (BIGINT)pow(N,d);                     // actual total grid points
  BIGINT N2 = (d>=2) ? N : 1, N3 = (d==3) ? N : 1;  // the y and z grid sizes
  std::vector<FLT> kx(M),ky(1),kz(1),d_nonuniform(2*M);    // NU, Re & Im
  if (d>1) ky.resize(M);                           // only alloc needed coords
  if (d>2) kz.resize(M);
  std::vector<FLT> d_uniform(2*Ng);                        // Re and Im

  spread_opts opts;
  int ier_set = setup_spreader(opts,(FLT)tol,upsampfac,kerevalmeth,debug,1,d);
  if (ier_set>1) {       // exit gracefully if can't set up.
    printf("error when setting up spreader (ier_set=%d)!\n",ier_set);
    return ier_set;
  }
  opts.pirange = 0;  // crucial, since below has NU pts on [0,Nd] in each dim
  opts.chkbnds = 0;  // only for debug, since below code has correct bounds);
                     // however, 1 can make a >5% slowdown for low-tol 3D.
  opts.debug = debug;   // print more diagnostics?
  opts.sort = sort;
  opts.flags = flags;
  opts.kerpad = kerpad;
  opts.upsampfac = upsampfac;
  opts.nthreads = 0;  // max # threads used, or 0 to use what's avail
  opts.sort_threads = 0;
  //opts.max_subproblem_size = 1e5;
  FLT maxerr, ansmod;
  
  // spread a single source, only for reference accuracy check...
  opts.spread_direction=1;
  d_nonuniform[0] = 1.0; d_nonuniform[1] = 0.0;   // unit strength
  kx[0] = ky[0] = kz[0] = N/FLT(2.0);                  // at center
  int ier = spreadinterp(N,N2,N3,d_uniform.data(),1,kx.data(),ky.data(),kz.data(),d_nonuniform.data(),opts);          // vector::data officially C++11 but works
  if (ier!=0) {
    printf("error when spreading M=1 pt for ref acc check (ier=%d)!\n",ier);
    return ier;
  }

  CPX kersum = tbb::parallel_reduce(  // sum kernel on uniform grid
      tbb::blocked_range<BIGINT>(0, Ng),
      CPX(0.0, 0.0),
      [&](tbb::blocked_range<BIGINT>& r, CPX init) -> CPX {
          for (BIGINT j = r.begin(); j < r.end(); j++) {
              init += CPX(d_uniform[2 * j], d_uniform[2 * j + 1]);
          }

          return init;
      },
      [](CPX a, CPX b) -> CPX {
          return a + b;
      });

  // now do the large-scale test w/ random sources..
  // initialize RNG
  VSLStreamStatePtr vslStream;
  int errcode = vslNewStream(&vslStream, VSL_BRNG_SFMT19937, 111);
  printf("making random data...\n");

#ifdef SINGLE
  switch (d) {
  case 3:
      errcode = vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, vslStream, M, kz.data(), 0.0f, 1.0f);
      // fall through
  case 2:
      errcode = vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, vslStream, M, ky.data(), 0.0f, 1.0f);
      // fall through
  case 1:
      errcode = vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, vslStream, M, kx.data(), 0.0f, 1.0f);
      errcode = vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, vslStream, 2 * M, d_nonuniform.data(), -1.0f, 1.0f);
      break;
  }
#else
  switch (d) {
  case 3:
      errcode = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, vslStream, M, kz.data(), 0.0, 1.0);
      // fall through
  case 2:
      errcode = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, vslStream, M, ky.data(), 0.0, 1.0);
      // fall through
  case 1:
      errcode = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, vslStream, M, kx.data(), 0.0, 1.0);
      errcode = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, vslStream, 2 * M, d_nonuniform.data(), -1.0, 1.0);
      break;
  }
#endif

  CPX str = tbb::parallel_reduce(
      tbb::blocked_range<BIGINT>(0, M),
      CPX(0.0, 0.0),
      [&](tbb::blocked_range<BIGINT>& r, CPX init) -> CPX {
          for (BIGINT j = r.begin(); j < r.end(); j++) {
              init += CPX(d_nonuniform[2 * j], d_nonuniform[2 * j + 1]);
          }

          return init;
      },
      [](CPX a, CPX b) -> CPX {
          return a + b;
      });

  CNTime timer;
  double t;
  if (dodir1) {   // test direction 1 (NU -> U spreading) ......................
    printf("spreadinterp %dD, %.3g U pts, dir=%d, tol=%.3g: nspread=%d\n",d,(double)Ng,opts.spread_direction,tol,opts.nspread);
    timer.start();
    ier = spreadinterp(N,N2,N3,d_uniform.data(),M,kx.data(),ky.data(),kz.data(),d_nonuniform.data(),opts);
    t=timer.elapsedsec();
    if (ier!=0) {
      printf("error (ier=%d)!\n",ier);
      return ier;
    } else
      printf("    %.3g NU pts in %.3g s \t%.3g pts/s \t%.3g spread pts/s\n",(double)M,t,M/t,pow(opts.nspread,d)*M/t);
  
    CPX sum = tbb::parallel_reduce(  // check spreading accuracy, wrapping
        tbb::blocked_range<BIGINT>(0, Ng),
        CPX(0.0, 0.0),
        [&](tbb::blocked_range<BIGINT>& r, CPX init) -> CPX {
            for (BIGINT j = r.begin(); j < r.end(); j++) {
                init += CPX(d_uniform[2 * j], d_uniform[2 * j + 1]);
            }

            return init;
        },
        [](CPX a, CPX b) -> CPX {
            return a + b;
        });

    CPX p = kersum*str;   // pred ans, complex mult
    FLT maxerr = std::max(abs(sum.real()-p.real()), abs(sum.imag()-p.imag()));
    FLT ansmod = abs(sum);
    printf("    rel err in total over grid:      %.3g\n",maxerr/ansmod);
    // note this is weaker than below dir=2 test, but is good indicator that
    // periodic wrapping is correct
  }

  // test direction 2 (U -> NU interpolation) ..............................
  printf("making more random NU pts...\n");
  for (BIGINT i=0;i<Ng;++i) {     // unit grid data
    d_uniform[2*i] = 1.0;
    d_uniform[2*i+1] = 0.0;
  }

#ifdef SINGLE
  switch (d) {
  case 3:
      errcode = vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, vslStream, M, kz.data(), 0.0f, 1.0f);
      // fall through
  case 2:
      errcode = vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, vslStream, M, ky.data(), 0.0f, 1.0f);
      // fall through
  case 1:
      errcode = vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, vslStream, M, kx.data(), 0.0f, 1.0f);
      break;
  }
#else
  switch (d) {
  case 3:
      errcode = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, vslStream, M, kz.data(), 0.0, 1.0);
      // fall through
  case 2:
      errcode = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, vslStream, M, ky.data(), 0.0, 1.0);
      // fall through
  case 1:
      errcode = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, vslStream, M, kx.data(), 0.0, 1.0);
      break;
  }
#endif

  opts.spread_direction=2;
  printf("spreadinterp %dD, %.3g U pts, dir=%d, tol=%.3g: nspread=%d\n",d,(double)Ng,opts.spread_direction,tol,opts.nspread);
  timer.restart();
  ier = spreadinterp(N,N2,N3,d_uniform.data(),M,kx.data(),ky.data(),kz.data(),d_nonuniform.data(),opts);
  t=timer.elapsedsec();
  if (ier!=0) {
    printf("error (ier=%d)!\n",ier);
    return 1;
  } else
    printf("    %.3g NU pts in %.3g s \t%.3g pts/s \t%.3g spread pts/s\n",(double)M,t,M/t,pow(opts.nspread,d)*M/t);

  // math test is worst-case error from pred value (kersum) on interp pts:
  maxerr = 0.0;
  for (BIGINT i=0;i<M;++i) {
    FLT err = std::max(abs(d_nonuniform[2*i]-kersum.real()),
		       abs(d_nonuniform[2*i+1]-kersum.imag()));
    if (err>maxerr) maxerr=err;
  }
  ansmod = abs(kersum);
  printf("    max rel err in values at NU pts: %.3g\n",maxerr/ansmod);
  // this is stronger test than for dir=1, since it tests sum of kernel for
  // each NU pt. However, it cannot detect reading
  // from wrong grid pts (they are all unity)

  errcode = vslDeleteStream(&vslStream);

  return 0;
}
