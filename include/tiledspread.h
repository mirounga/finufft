// --------------------------------------------------------------------------
// Defines templates for the tiled spreading code
// And optimized specializations.

#ifndef PIPEDSPREAD_H
#define PIPEDSPREAD_H

#include <vector>
#include <defs.h>
#include <tiledeval.h>

#include <immintrin.h>

// declarations of purely internal functions...
void spread_subproblem_1d(BIGINT off1, BIGINT size1, FLT* du0, BIGINT M0, FLT* kx0,
    FLT* dd0, const spread_opts& opts);
void spread_subproblem_2d(BIGINT off1, BIGINT off2, BIGINT size1, BIGINT size2,
    FLT* du0, BIGINT M0,
    FLT* kx0, FLT* ky0, FLT* dd0, const spread_opts& opts);
void spread_subproblem_3d(BIGINT off1, BIGINT off2, BIGINT off3, BIGINT size1,
    BIGINT size2, BIGINT size3, FLT* du0, BIGINT M0,
    FLT* kx0, FLT* ky0, FLT* kz0, FLT* dd0,
    const spread_opts& opts);
void add_wrapped_subgrid(BIGINT offset1, BIGINT offset2, BIGINT offset3,
    BIGINT size1, BIGINT size2, BIGINT size3, BIGINT N1,
    BIGINT N2, BIGINT N3, FLT* data_uniform, FLT* du0);
void add_wrapped_subgrid_thread_safe(BIGINT offset1, BIGINT offset2, BIGINT offset3,
    BIGINT size1, BIGINT size2, BIGINT size3, BIGINT N1,
    BIGINT N2, BIGINT N3, FLT* data_uniform, FLT* du0);
void get_subgrid(BIGINT &offset1,BIGINT &offset2,BIGINT &offset3,BIGINT &size1,
		 BIGINT &size2,BIGINT &size3,BIGINT M0,FLT* kx0,FLT* ky0,
		 FLT* kz0,int ns, int ndims);

template<int ns>
void tiled_spread_cube(BIGINT off1, BIGINT off2, BIGINT off3,
    BIGINT size1, BIGINT size2, BIGINT size3, FLT* dd,
    BIGINT* i1, BIGINT* i2, BIGINT* i3,
    FLT** ker1, FLT** ker2, FLT** ker3, FLT* du)
{
    for (int n = 0; n < npipes; n++) {
        FLT re0 = dd[2 * n];
        FLT im0 = dd[2 * n + 1];

        // Combine kernel with complex source value to simplify inner loop
        FLT ker1val[2 * MAX_NSPREAD];    // here 2* is because of complex
        for (int j = 0; j < ns; j++) {
            ker1val[2 * j] = re0 * ker1[n][j];
            ker1val[2 * j + 1] = im0 * ker1[n][j];
        }

        // critical inner loop:
        for (int dz = 0; dz < ns; ++dz) {
            BIGINT oz = size1 * size2 * (i3[n] - off3 + dz);        // offset due to z
            for (int dy = 0; dy < ns; ++dy) {
                BIGINT j = oz + size1 * (i2[n] - off2 + dy) + i1[n] - off1;   // should be in subgrid
                FLT kerval = ker2[n][dy] * ker3[n][dz];
                FLT* trg = du + 2 * j;
                for (int dx = 0; dx < 2 * ns; ++dx) {
                    trg[dx] += kerval * ker1val[dx];
                }
            }
        }
    }
}

template<int ns, bool isUpsampfacHigh>
void tiled_spread_subproblem_3d(BIGINT off1, BIGINT off2, BIGINT off3, BIGINT size1,
    BIGINT size2, BIGINT size3, FLT* du, BIGINT M,
    FLT* kx, FLT* ky, FLT* kz, FLT* dd,
    const spread_opts& opts)
    /* spreader from dd (NU) to du (uniform) in 3D without wrapping.
       See above docs/notes for spread_subproblem_2d.
       kx,ky,kz (size M) are NU locations in [off+ns/2,off+size-1-ns/2] in each dim.
       dd (size M complex) are complex source strengths
       du (size size1*size2*size3) is uniform complex output array
     */
{
    FLT ns2 = (FLT)ns / 2;          // half spread width
    for (BIGINT i = 0; i < 2 * size1 * size2 * size3; ++i)
        du[i] = 0.0;
    alignas(64) FLT kernel_args[3 * MAX_NSPREAD];
    // Kernel values stored in consecutive memory. This allows us to compute
    // values in all three directions in a single kernel evaluation call.
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

    BIGINT M0 = M % npipes;

    // Prologue
    for (BIGINT i = 0; i < M0; i++) {           // loop over NU pts
        FLT re0 = dd[2 * i];
        FLT im0 = dd[2 * i + 1];
        // ceil offset, hence rounding, must match that in get_subgrid...
        BIGINT i1 = (BIGINT)std::ceil(kx[i] - ns2);   // fine grid start indices
        BIGINT i2 = (BIGINT)std::ceil(ky[i] - ns2);
        BIGINT i3 = (BIGINT)std::ceil(kz[i] - ns2);
        FLT x1 = (FLT)i1 - kx[i];
        FLT x2 = (FLT)i2 - ky[i];
        FLT x3 = (FLT)i3 - kz[i];
        if (opts.kerevalmeth == 0) {          // faster Horner poly method
            set_kernel_args(kernel_args, x1, opts);
            set_kernel_args(kernel_args + ns, x2, opts);
            set_kernel_args(kernel_args + 2 * ns, x3, opts);
            evaluate_kernel_vector(kernel_values, kernel_args, opts, 3 * ns);
        }
        else {
            eval_kernel_vec_Horner(ker1[0], x1, ns, opts);
            eval_kernel_vec_Horner(ker2[0], x2, ns, opts);
            eval_kernel_vec_Horner(ker3[0], x3, ns, opts);
        }
        // Combine kernel with complex source value to simplify inner loop
        FLT ker1val[2 * MAX_NSPREAD];    // here 2* is because of complex
        for (int j = 0; j < ns; j++) {
            ker1val[2 * j] = re0 * ker1[0][j];
            ker1val[2 * j + 1] = im0 * ker1[0][j];
        }
        // critical inner loop:
        for (int dz = 0; dz < ns; ++dz) {
            BIGINT oz = size1 * size2 * (i3 - off3 + dz);        // offset due to z
            for (int dy = 0; dy < ns; ++dy) {
                BIGINT j = oz + size1 * (i2 - off2 + dy) + i1 - off1;   // should be in subgrid
                FLT kerval = ker2[0][dy] * ker3[0][dz];
                FLT* trg = du + 2 * j;
                for (int dx = 0; dx < 2 * ns; ++dx) {
                    trg[dx] += kerval * ker1val[dx];
                }
            }
        }
    }

    // Main loop over NU pts
    for (BIGINT m = M0; m < M; m += npipes) {
        alignas(64) BIGINT i1[npipes];   // fine grid start indices
        alignas(64) BIGINT i2[npipes];
        alignas(64) BIGINT i3[npipes];
        alignas(64) FLT x1[npipes];
        alignas(64) FLT x2[npipes];
        alignas(64) FLT x3[npipes];

        for (int n = 0; n < npipes; n++) {
            BIGINT i = m + n;
            // ceil offset, hence rounding, must match that in get_subgrid...
            i1[n] = (BIGINT)std::ceil(kx[i] - ns2);
            i2[n] = (BIGINT)std::ceil(ky[i] - ns2);
            i3[n] = (BIGINT)std::ceil(kz[i] - ns2);
            x1[n] = (FLT)i1[n] - kx[i];
            x2[n] = (FLT)i2[n] - ky[i];
            x3[n] = (FLT)i3[n] - kz[i];
        }

        if (opts.kerevalmeth == 0) {          // faster Horner poly method
            for (int n = 0; n < npipes; n++) {
                set_kernel_args(kernel_args, x1[n], opts);
                set_kernel_args(kernel_args + ns, x2[n], opts);
                set_kernel_args(kernel_args + 2 * ns, x3[n], opts);
                evaluate_kernel_vector(ker1[n], kernel_args, opts, 3 * ns);
            }
        }
        else {
            tiled_eval_kernel_vec_Horner<ns, isUpsampfacHigh>(ker1, x1, opts);
            tiled_eval_kernel_vec_Horner<ns, isUpsampfacHigh>(ker2, x2, opts);
            tiled_eval_kernel_vec_Horner<ns, isUpsampfacHigh>(ker3, x3, opts);
        }
       
        tiled_spread_cube<ns>(off1, off2, off3,
            size1, size2, size3, dd + 2 * m,
            i1, i2, i3, ker1, ker2, ker3, du);
    }
}

template<int ndims, int ns, bool isUpsampfacHigh>
int tiled_spreadSorted(BIGINT* sort_indices,BIGINT N1, BIGINT N2, BIGINT N3,
		      FLT *data_uniform,BIGINT M, FLT *kx, FLT *ky, FLT *kz,
		      FLT *data_nonuniform, const spread_opts& opts, int did_sort)
// Spread NU pts in sorted order to a uniform grid. See spreadinterp() for doc.
{
  CNTime timer;
  BIGINT N=N1*N2*N3;            // output array size
  int nthr = MY_OMP_GET_MAX_THREADS();  // # threads to use to spread
  if (opts.nthreads>0)
    nthr = min(nthr,opts.nthreads);     // user override up to max avail
  if (opts.debug)
    printf("\tspread %dD (M=%lld; N1=%lld,N2=%lld,N3=%lld; pir=%d), nthr=%d\n",ndims,(long long)M,(long long)N1,(long long)N2,(long long)N3,opts.pirange,nthr);
  
  timer.start();
  for (BIGINT i=0; i<2*N; i++) // zero the output array. std::fill is no faster
    data_uniform[i]=0.0;
  if (opts.debug) printf("\tzero output array\t%.3g s\n",timer.elapsedsec());
  if (M==0)                     // no NU pts, we're done
    return 0;
  
  int spread_single = (nthr==1) || (M*100<N);     // low-density heuristic?
  spread_single = 0;                 // for now
  timer.start();
  if (spread_single) {    // ------- Basic single-core t1 spreading ------
    for (BIGINT j=0; j<M; j++) {
      // *** todo, not urgent
      // ... (question is: will the index wrapping per NU pt slow it down?)
    }
    if (opts.debug) printf("\tt1 simple spreading:\t%.3g s\n",timer.elapsedsec());
    
  } else {           // ------- Fancy multi-core blocked t1 spreading ----
                     // Splits sorted inds (jfm's advanced2), could double RAM.
    // choose nb (# subprobs) via used nthreads:
    BIGINT nb = min((BIGINT)nthr,M);         // simply split one subprob per thr...
    if (nb*(BIGINT)opts.max_subproblem_size<M) {  // ...or more subprobs to cap size
      nb = 1 + (M-1)/opts.max_subproblem_size;  // int div does ceil(M/opts.max_subproblem_size)
      if (opts.debug) printf("\tcapping subproblem sizes to max of %d\n",opts.max_subproblem_size);
    }
    if (M*1000<N) {         // low-density heuristic: one thread per NU pt!
      nb = M;
      if (opts.debug) printf("\tusing low-density speed rescue nb=M...\n");
    }
    if (!did_sort && nthr==1) {
      nb = 1;
      if (opts.debug) printf("\tunsorted nthr=1: forcing single subproblem...\n");
    }
    if (opts.debug && nthr>opts.atomic_threshold)
      printf("\tnthr big: switching add_wrapped OMP from critical to atomic (!)\n");
      
    std::vector<BIGINT> brk(nb+1); // NU index breakpoints defining nb subproblems
    for (int p=0;p<=nb;++p)
      brk[p] = (BIGINT)(0.5 + M*p/(double)nb);
    
#pragma omp parallel for num_threads(nthr) schedule(dynamic,1)  // each is big
      for (int isub=0; isub<nb; isub++) {   // Main loop through the subproblems
        BIGINT M0 = brk[isub+1]-brk[isub];  // # NU pts in this subproblem
        // copy the location and data vectors for the nonuniform points
        FLT *kx0=(FLT*)malloc(sizeof(FLT)*M0), *ky0=NULL, *kz0=NULL;
        if (N2>1)
          ky0=(FLT*)malloc(sizeof(FLT)*M0);
        if (N3>1)
          kz0=(FLT*)malloc(sizeof(FLT)*M0);
        FLT *dd0=(FLT*)malloc(sizeof(FLT)*M0*2);    // complex strength data
        for (BIGINT j=0; j<M0; j++) {           // todo: can avoid this copying?
          BIGINT kk=sort_indices[j+brk[isub]];  // NU pt from subprob index list
          kx0[j]=FOLDRESCALE(kx[kk],N1,opts.pirange);
          if (N2>1) ky0[j]=FOLDRESCALE(ky[kk],N2,opts.pirange);
          if (N3>1) kz0[j]=FOLDRESCALE(kz[kk],N3,opts.pirange);
          dd0[j*2]=data_nonuniform[kk*2];     // real part
          dd0[j*2+1]=data_nonuniform[kk*2+1]; // imag part
        }
        // get the subgrid which will include padding by roughly nspread/2
        BIGINT offset1,offset2,offset3,size1,size2,size3; // get_subgrid sets
        get_subgrid(offset1,offset2,offset3,size1,size2,size3,M0,kx0,ky0,kz0,ns,ndims);  // sets offsets and sizes
        if (opts.debug>1) { // verbose
          if (ndims==1)
            printf("\tsubgrid: off %lld\t siz %lld\t #NU %lld\n",(long long)offset1,(long long)size1,(long long)M0);
          else if (ndims==2)
            printf("\tsubgrid: off %lld,%lld\t siz %lld,%lld\t #NU %lld\n",(long long)offset1,(long long)offset2,(long long)size1,(long long)size2,(long long)M0);
          else
            printf("\tsubgrid: off %lld,%lld,%lld\t siz %lld,%lld,%lld\t #NU %lld\n",(long long)offset1,(long long)offset2,(long long)offset3,(long long)size1,(long long)size2,(long long)size3,(long long)M0);
	}
        // allocate output data for this subgrid
        FLT *du0=(FLT*)malloc(sizeof(FLT)*2*size1*size2*size3); // complex
        
        // Spread to subgrid without need for bounds checking or wrapping
        if (!(opts.flags & TF_OMIT_SPREADING)) {
          if (ndims==1)
            spread_subproblem_1d(offset1,size1,du0,M0,kx0,dd0,opts);
          else if (ndims==2)
            spread_subproblem_2d(offset1,offset2,size1,size2,du0,M0,kx0,ky0,dd0,opts);
          else
            tiled_spread_subproblem_3d<ns, isUpsampfacHigh>(offset1,offset2,offset3,size1,size2,size3,du0,M0,kx0,ky0,kz0,dd0,opts);
	}
        
        // do the adding of subgrid to output
        if (!(opts.flags & TF_OMIT_WRITE_TO_GRID)) {
          if (nthr > opts.atomic_threshold)   // see above for debug reporting
            add_wrapped_subgrid_thread_safe(offset1,offset2,offset3,size1,size2,size3,N1,N2,N3,data_uniform,du0);   // R Blackwell's atomic version
          else {
#pragma omp critical
            add_wrapped_subgrid(offset1,offset2,offset3,size1,size2,size3,N1,N2,N3,data_uniform,du0);
          }
        }

        // free up stuff from this subprob... (that was malloc'ed by hand)
        free(dd0);
        free(du0);
        free(kx0);
        if (N2>1) free(ky0);
        if (N3>1) free(kz0); 
      }     // end main loop over subprobs
      if (opts.debug) printf("\tt1 fancy spread: \t%.3g s (%ld subprobs)\n",timer.elapsedsec(), nb);
    }   // end of choice of which t1 spread type to use
    return 0;
};

#ifdef __AVX512F__
#ifdef SINGLE
template<>
void tiled_spread_cube<7>(BIGINT off1, BIGINT off2, BIGINT off3,
    BIGINT size1, BIGINT size2, BIGINT size3, float* dd,
    BIGINT* i1, BIGINT* i2, BIGINT* i3,
    float** ker1, float** ker2, float** ker3, float* du)
{
    constexpr int ns = 7;
    const __mmask16 _n2_mask = 0x3fff;

    const __m512i _perm = _mm512_set_epi32(7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0);

    for (int n = 0; n < npipes; n++) {
        // Combine kernel with complex source value to simplify inner loop
        __m512 _dd = _mm512_broadcast_f32x2(_mm_maskz_loadu_ps(0x03, dd + 2 * n));

        __m512 _ker1val = _mm512_mul_ps(_dd, _mm512_permutexvar_ps(_perm, _mm512_maskz_loadu_ps(0x007f, ker1[n])));

        // critical inner loop:
        for (int dz = 0; dz < ns; ++dz) {
            BIGINT oz = size1 * size2 * (i3[n] - off3 + dz);        // offset due to z
            BIGINT j = oz + size1 * (i2[n] - off2) + i1[n] - off1;   // should be in subgrid

            float* pdua = du + 2 * j;
            __m512 _kervala = _mm512_set1_ps(ker2[n][0] * ker3[n][dz]);
            __m512 _trga = _mm512_fmadd_ps(_ker1val, _kervala, _mm512_maskz_loadu_ps(_n2_mask, pdua));
            _mm512_mask_storeu_ps(pdua, _n2_mask, _trga);

            float* pdub = pdua + 2 * size1;
            __m512 _kervalb = _mm512_set1_ps(ker2[n][1] * ker3[n][dz]);
            __m512 _trgb = _mm512_fmadd_ps(_ker1val, _kervalb, _mm512_maskz_loadu_ps(_n2_mask, pdub));
            _mm512_mask_storeu_ps(pdub, _n2_mask, _trgb);

            float* pduc = pdub + 2 * size1;
            __m512 _kervalc = _mm512_set1_ps(ker2[n][2] * ker3[n][dz]);
            __m512 _trgc = _mm512_fmadd_ps(_ker1val, _kervalc, _mm512_maskz_loadu_ps(_n2_mask, pduc));
            _mm512_mask_storeu_ps(pduc, _n2_mask, _trgc);

            float* pdud = pduc + 2 * size1;
            __m512 _kervald = _mm512_set1_ps(ker2[n][3] * ker3[n][dz]);
            __m512 _trgd = _mm512_fmadd_ps(_ker1val, _kervald, _mm512_maskz_loadu_ps(_n2_mask, pdud));
            _mm512_mask_storeu_ps(pdud, _n2_mask, _trgd);

            float* pdue = pdud + 2 * size1;
            __m512 _kervale = _mm512_set1_ps(ker2[n][4] * ker3[n][dz]);
            __m512 _trge = _mm512_fmadd_ps(_ker1val, _kervale, _mm512_maskz_loadu_ps(_n2_mask, pdue));
            _mm512_mask_storeu_ps(pdue, _n2_mask, _trge);

            float* pduf = pdue + 2 * size1;
            __m512 _kervalf = _mm512_set1_ps(ker2[n][5] * ker3[n][dz]);
            __m512 _trgf = _mm512_fmadd_ps(_ker1val, _kervalf, _mm512_maskz_loadu_ps(_n2_mask, pduf));
            _mm512_mask_storeu_ps(pduf, _n2_mask, _trgf);

            float* pdug = pduf + 2 * size1;
            __m512 _kervalg = _mm512_set1_ps(ker2[n][6] * ker3[n][dz]);
            __m512 _trgg = _mm512_fmadd_ps(_ker1val, _kervalg, _mm512_maskz_loadu_ps(_n2_mask, pdug));
            _mm512_mask_storeu_ps(pdug, _n2_mask, _trgg);
        }
    }
}
#else
template<>
void tiled_spread_cube<7>(BIGINT off1, BIGINT off2, BIGINT off3,
    BIGINT size1, BIGINT size2, BIGINT size3, double* dd,
    BIGINT* i1, BIGINT* i2, BIGINT* i3,
    double** ker1, double** ker2, double** ker3, double* du)
{
    constexpr int ns = 7;
    const __m512i _perm0 = _mm512_set_epi64(7, 7, 2, 2, 1, 1, 0, 0);
    const __m512i _perm1 = _mm512_set_epi64(6, 6, 5, 5, 4, 4, 3, 3);
    for (int n = 0; n < npipes; n++) {
        // Combine kernel with complex source value to simplify inner loop
        __m128d _d = _mm_load_pd(dd + 2 * n);
        __m512d _dd = _mm512_broadcast_f64x2(_d);

        __m512d _ker1x = _mm512_maskz_loadu_pd(0x7f, ker1[n]);

        __m512d _ker1val0 = _mm512_mul_pd(_dd, _mm512_permutexvar_pd(_perm0, _ker1x));
        __m512d _ker1val1 = _mm512_mul_pd(_dd, _mm512_permutexvar_pd(_perm1, _ker1x));

        // critical inner loop:
        for (int dz = 0; dz < ns; ++dz) {
            BIGINT oz = size1 * size2 * (i3[n] - off3 + dz);        // offset due to z
            BIGINT j = oz + size1 * (i2[n] - off2) + i1[n] - off1;   // should be in subgrid
            double* pdu;

            pdu = du + 2 * j;
            __m512d _kervala = _mm512_set1_pd(ker2[n][0] * ker3[n][dz]);
            __m512d _trg0a = _mm512_fmadd_pd(_ker1val0, _kervala, _mm512_loadu_pd(pdu + 0));
            __m512d _trg1a = _mm512_fmadd_pd(_ker1val1, _kervala, _mm512_loadu_pd(pdu + 6));
            _mm512_storeu_pd(pdu + 0, _trg0a);
            _mm512_storeu_pd(pdu + 6, _trg1a);

            pdu += 2 * size1;
            __m512d _kervalb = _mm512_set1_pd(ker2[n][1] * ker3[n][dz]);
            __m512d _trg0b = _mm512_fmadd_pd(_ker1val0, _kervalb, _mm512_loadu_pd(pdu + 0));
            __m512d _trg1b = _mm512_fmadd_pd(_ker1val1, _kervalb, _mm512_loadu_pd(pdu + 6));
            _mm512_storeu_pd(pdu + 0, _trg0b);
            _mm512_storeu_pd(pdu + 6, _trg1b);

            pdu += 2 * size1;
            __m512d _kervalc = _mm512_set1_pd(ker2[n][2] * ker3[n][dz]);
            __m512d _trg0c = _mm512_fmadd_pd(_ker1val0, _kervalc, _mm512_loadu_pd(pdu + 0));
            __m512d _trg1c = _mm512_fmadd_pd(_ker1val1, _kervalc, _mm512_loadu_pd(pdu + 6));
            _mm512_storeu_pd(pdu + 0, _trg0c);
            _mm512_storeu_pd(pdu + 6, _trg1c);

            pdu += 2 * size1;
            __m512d _kervald = _mm512_set1_pd(ker2[n][3] * ker3[n][dz]);
            __m512d _trg0d = _mm512_fmadd_pd(_ker1val0, _kervald, _mm512_loadu_pd(pdu + 0));
            __m512d _trg1d = _mm512_fmadd_pd(_ker1val1, _kervald, _mm512_loadu_pd(pdu + 6));
            _mm512_storeu_pd(pdu + 0, _trg0d);
            _mm512_storeu_pd(pdu + 6, _trg1d);

            pdu += 2 * size1;
            __m512d _kervale = _mm512_set1_pd(ker2[n][4] * ker3[n][dz]);
            __m512d _trg0e = _mm512_fmadd_pd(_ker1val0, _kervale, _mm512_loadu_pd(pdu + 0));
            __m512d _trg1e = _mm512_fmadd_pd(_ker1val1, _kervale, _mm512_loadu_pd(pdu + 6));
            _mm512_storeu_pd(pdu + 0, _trg0e);
            _mm512_storeu_pd(pdu + 6, _trg1e);

            pdu += 2 * size1;
            __m512d _kervalf = _mm512_set1_pd(ker2[n][5] * ker3[n][dz]);
            __m512d _trg0f = _mm512_fmadd_pd(_ker1val0, _kervalf, _mm512_loadu_pd(pdu + 0));
            __m512d _trg1f = _mm512_fmadd_pd(_ker1val1, _kervalf, _mm512_loadu_pd(pdu + 6));
            _mm512_storeu_pd(pdu + 0, _trg0f);
            _mm512_storeu_pd(pdu + 6, _trg1f);

            pdu += 2 * size1;
            __m512d _kervalg = _mm512_set1_pd(ker2[n][6] * ker3[n][dz]);
            __m512d _trg0g = _mm512_fmadd_pd(_ker1val0, _kervalg, _mm512_loadu_pd(pdu + 0));
            __m512d _trg1g = _mm512_fmadd_pd(_ker1val1, _kervalg, _mm512_loadu_pd(pdu + 6));
            _mm512_storeu_pd(pdu + 0, _trg0g);
            _mm512_storeu_pd(pdu + 6, _trg1g);
        }
    }
}
#endif
#else
#ifdef __AVX2__
#ifdef SINGLE
template<>
void tiled_spread_cube<7>(BIGINT off1, BIGINT off2, BIGINT off3,
    BIGINT size1, BIGINT size2, BIGINT size3, float* dd,
    BIGINT* i1, BIGINT* i2, BIGINT* i3,
    float** ker1, float** ker2, float** ker3, float* du)
{
    constexpr int ns = 7;
    const __m256i _interleave = _mm256_set_epi32(1, 0, 1, 0, 1, 0, 1, 0);
    const __m256i _duplicate = _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0);
    for (int n = 0; n < npipes; n++) {
        // Combine kernel with complex source value to simplify inner loop
        __m256 _dd = _mm256_permutevar8x32_ps(
            _mm256_maskload_ps(dd + 2 * n, _mm256_set_epi32(0, 0, 0, 0, 0, 0, -1, -1)),
            _interleave);

        __m256 _ker1val0 = _mm256_mul_ps(_dd, 
            _mm256_permutevar8x32_ps(
                _mm256_castps128_ps256(_mm_load_ps(ker1[n] + 0)),
                _duplicate));
        __m256 _ker1val1 = _mm256_mul_ps(_dd,
            _mm256_permutevar8x32_ps(
                _mm256_castps128_ps256(_mm_loadu_ps(ker1[n] + 3)), 
                _duplicate));

        //FLT re0 = dd[2 * n];
        //FLT im0 = dd[2 * n + 1];

        //// Combine kernel with complex source value to simplify inner loop
        //FLT ker1val[2 * MAX_NSPREAD];    // here 2* is because of complex
        //for (int j = 0; j < ns; j++) {
        //    ker1val[2 * j] = re0 * ker1[n][j];
        //    ker1val[2 * j + 1] = im0 * ker1[n][j];
        //    if (j < 3) {
        //        assert(ker1val[2 * j] == _ker1val0.m256_f32[2 * j]);
        //        assert(ker1val[2 * j + 1] == _ker1val0.m256_f32[2 * j + 1]);
        //    }
        //    else {
        //        assert(ker1val[2 * j] == _ker1val1.m256_f32[2 * (j-3)]);
        //        assert(ker1val[2 * j + 1] == _ker1val1.m256_f32[2 * (j-3) + 1]);
        //    }
        //}

        // critical inner loop:
        for (int dz = 0; dz < ns; ++dz) {
            BIGINT oz = size1 * size2 * (i3[n] - off3 + dz);        // offset due to z
            BIGINT j = oz + size1 * (i2[n] - off2) + i1[n] - off1;   // should be in subgrid
            float* pdua, * pdub;
            __m256 _kervala, _kervalb;
            __m256 _trg0a, _trg1a, _trg0b, _trg1b;

            pdub = du + 2 * j;
            _kervalb = _mm256_set1_ps(ker2[n][0] * ker3[n][dz]);
            _trg0b = _mm256_fmadd_ps(_ker1val0, _kervalb, _mm256_load_ps(pdub + 0));
            _trg1b = _mm256_fmadd_ps(_ker1val1, _kervalb, _mm256_load_ps(pdub + 6));
            _mm256_storeu_ps(pdub + 0, _trg0b);
            _mm256_storeu_ps(pdub + 6, _trg1b);

            pdua = pdub + 2 * size1;
            _kervala = _mm256_set1_ps(ker2[n][1] * ker3[n][dz]);
            _trg0a = _mm256_fmadd_ps(_ker1val0, _kervala, _mm256_load_ps(pdua + 0));
            _trg1a = _mm256_fmadd_ps(_ker1val1, _kervala, _mm256_load_ps(pdua + 6));
            _mm256_storeu_ps(pdua + 0, _trg0a);
            _mm256_storeu_ps(pdua + 6, _trg1a);

            pdub = pdua + 2 * size1;
            _kervalb = _mm256_set1_ps(ker2[n][2] * ker3[n][dz]);
            _trg0b = _mm256_fmadd_ps(_ker1val0, _kervalb, _mm256_load_ps(pdub + 0));
            _trg1b = _mm256_fmadd_ps(_ker1val1, _kervalb, _mm256_load_ps(pdub + 6));
            _mm256_storeu_ps(pdub + 0, _trg0b);
            _mm256_storeu_ps(pdub + 6, _trg1b);

            pdua = pdub + 2 * size1;
            _kervala = _mm256_set1_ps(ker2[n][3] * ker3[n][dz]);
            _trg0a = _mm256_fmadd_ps(_ker1val0, _kervala, _mm256_load_ps(pdua + 0));
            _trg1a = _mm256_fmadd_ps(_ker1val1, _kervala, _mm256_load_ps(pdua + 6));
            _mm256_storeu_ps(pdua + 0, _trg0a);
            _mm256_storeu_ps(pdua + 6, _trg1a);

            pdub = pdua + 2 * size1;
            _kervalb = _mm256_set1_ps(ker2[n][4] * ker3[n][dz]);
            _trg0b = _mm256_fmadd_ps(_ker1val0, _kervalb, _mm256_load_ps(pdub + 0));
            _trg1b = _mm256_fmadd_ps(_ker1val1, _kervalb, _mm256_load_ps(pdub + 6));
            _mm256_storeu_ps(pdub + 0, _trg0b);
            _mm256_storeu_ps(pdub + 6, _trg1b);

            pdua = pdub + 2 * size1;
            _kervala = _mm256_set1_ps(ker2[n][5] * ker3[n][dz]);
            _trg0a = _mm256_fmadd_ps(_ker1val0, _kervala, _mm256_load_ps(pdua + 0));
            _trg1a = _mm256_fmadd_ps(_ker1val1, _kervala, _mm256_load_ps(pdua + 6));
            _mm256_storeu_ps(pdua + 0, _trg0a);
            _mm256_storeu_ps(pdua + 6, _trg1a);

            pdub = pdua + 2 * size1;
            _kervalb = _mm256_set1_ps(ker2[n][6] * ker3[n][dz]);
            _trg0b = _mm256_fmadd_ps(_ker1val0, _kervalb, _mm256_load_ps(pdub + 0));
            _trg1b = _mm256_fmadd_ps(_ker1val1, _kervalb, _mm256_load_ps(pdub + 6));
            _mm256_storeu_ps(pdub + 0, _trg0b);
            _mm256_storeu_ps(pdub + 6, _trg1b);
        }
    }
}
#else
template<>
void tiled_spread_cube<7>(BIGINT off1, BIGINT off2, BIGINT off3,
    BIGINT size1, BIGINT size2, BIGINT size3, double* dd,
    BIGINT* i1, BIGINT* i2, BIGINT* i3,
    double** ker1, double** ker2, double** ker3, double* du)
{
    constexpr int ns = 7;
       
    for (int n = 0; n < npipes; n++) {
        // Combine kernel with complex source value to simplify inner loop
        __m128d _d = _mm_load_pd(dd + 2 * n);
        __m256d _dd = _mm256_set_m128d(_d, _d);

        __m128d _ker10 = _mm_loaddup_pd(ker1[n] + 0);
        __m256d _ker11 = _mm256_load_pd(ker1[n] + 1);
        __m256d _ker12 = _mm256_load_pd(ker1[n] + 3);

        __m128d _ker1val0 = _mm_mul_pd(_d, _ker10);
        __m256d _ker1val1 = _mm256_mul_pd(_dd, _mm256_permute4x64_pd(_ker11, 0x50));
        __m256d _ker1val2 = _mm256_mul_pd(_dd, _mm256_permute4x64_pd(_ker11, 0xfa));
        __m256d _ker1val3 = _mm256_mul_pd(_dd, _mm256_permute4x64_pd(_ker12, 0xfa));

        // critical inner loop:
        for (int dz = 0; dz < ns; ++dz) {
            BIGINT oz = size1 * size2 * (i3[n] - off3 + dz);        // offset due to z
            BIGINT j = oz + size1 * (i2[n] - off2) + i1[n] - off1;   // should be in subgrid
            double* pdu;
            __m256d _kerval;
            __m128d _trg0; __m256d _trg1, _trg2, _trg3;

            pdu = du + 2 * j;
            _kerval = _mm256_set1_pd(ker2[n][0] * ker3[n][dz]);
            _trg0 = _mm_fmadd_pd(_ker1val0, _mm256_castpd256_pd128(_kerval), _mm_load_pd(pdu + 0));
            _trg1 = _mm256_fmadd_pd(_ker1val1, _kerval, _mm256_load_pd(pdu + 2));
            _trg2 = _mm256_fmadd_pd(_ker1val2, _kerval, _mm256_load_pd(pdu + 6));
            _trg3 = _mm256_fmadd_pd(_ker1val3, _kerval, _mm256_load_pd(pdu + 10));
            _mm_storeu_pd(pdu + 0, _trg0);
            _mm256_storeu_pd(pdu + 2, _trg1);
            _mm256_storeu_pd(pdu + 6, _trg2);
            _mm256_storeu_pd(pdu + 10, _trg3);

            pdu += 2 * size1;
            _kerval = _mm256_set1_pd(ker2[n][1] * ker3[n][dz]);
            _trg0 = _mm_fmadd_pd(_ker1val0, _mm256_castpd256_pd128(_kerval), _mm_load_pd(pdu + 0));
            _trg1 = _mm256_fmadd_pd(_ker1val1, _kerval, _mm256_load_pd(pdu + 2));
            _trg2 = _mm256_fmadd_pd(_ker1val2, _kerval, _mm256_load_pd(pdu + 6));
            _trg3 = _mm256_fmadd_pd(_ker1val3, _kerval, _mm256_load_pd(pdu + 10));
            _mm_storeu_pd(pdu + 0, _trg0);
            _mm256_storeu_pd(pdu + 2, _trg1);
            _mm256_storeu_pd(pdu + 6, _trg2);
            _mm256_storeu_pd(pdu + 10, _trg3);

            pdu += 2 * size1;
            _kerval = _mm256_set1_pd(ker2[n][2] * ker3[n][dz]);
            _trg0 = _mm_fmadd_pd(_ker1val0, _mm256_castpd256_pd128(_kerval), _mm_load_pd(pdu + 0));
            _trg1 = _mm256_fmadd_pd(_ker1val1, _kerval, _mm256_load_pd(pdu + 2));
            _trg2 = _mm256_fmadd_pd(_ker1val2, _kerval, _mm256_load_pd(pdu + 6));
            _trg3 = _mm256_fmadd_pd(_ker1val3, _kerval, _mm256_load_pd(pdu + 10));
            _mm_storeu_pd(pdu + 0, _trg0);
            _mm256_storeu_pd(pdu + 2, _trg1);
            _mm256_storeu_pd(pdu + 6, _trg2);
            _mm256_storeu_pd(pdu + 10, _trg3);

            pdu += 2 * size1;
            _kerval = _mm256_set1_pd(ker2[n][3] * ker3[n][dz]);
            _trg0 = _mm_fmadd_pd(_ker1val0, _mm256_castpd256_pd128(_kerval), _mm_load_pd(pdu + 0));
            _trg1 = _mm256_fmadd_pd(_ker1val1, _kerval, _mm256_load_pd(pdu + 2));
            _trg2 = _mm256_fmadd_pd(_ker1val2, _kerval, _mm256_load_pd(pdu + 6));
            _trg3 = _mm256_fmadd_pd(_ker1val3, _kerval, _mm256_load_pd(pdu + 10));
            _mm_storeu_pd(pdu + 0, _trg0);
            _mm256_storeu_pd(pdu + 2, _trg1);
            _mm256_storeu_pd(pdu + 6, _trg2);
            _mm256_storeu_pd(pdu + 10, _trg3);

            pdu += 2 * size1;
            _kerval = _mm256_set1_pd(ker2[n][4] * ker3[n][dz]);
            _trg0 = _mm_fmadd_pd(_ker1val0, _mm256_castpd256_pd128(_kerval), _mm_load_pd(pdu + 0));
            _trg1 = _mm256_fmadd_pd(_ker1val1, _kerval, _mm256_load_pd(pdu + 2));
            _trg2 = _mm256_fmadd_pd(_ker1val2, _kerval, _mm256_load_pd(pdu + 6));
            _trg3 = _mm256_fmadd_pd(_ker1val3, _kerval, _mm256_load_pd(pdu + 10));
            _mm_storeu_pd(pdu + 0, _trg0);
            _mm256_storeu_pd(pdu + 2, _trg1);
            _mm256_storeu_pd(pdu + 6, _trg2);
            _mm256_storeu_pd(pdu + 10, _trg3);

            pdu += 2 * size1;
            _kerval = _mm256_set1_pd(ker2[n][5] * ker3[n][dz]);
            _trg0 = _mm_fmadd_pd(_ker1val0, _mm256_castpd256_pd128(_kerval), _mm_load_pd(pdu + 0));
            _trg1 = _mm256_fmadd_pd(_ker1val1, _kerval, _mm256_load_pd(pdu + 2));
            _trg2 = _mm256_fmadd_pd(_ker1val2, _kerval, _mm256_load_pd(pdu + 6));
            _trg3 = _mm256_fmadd_pd(_ker1val3, _kerval, _mm256_load_pd(pdu + 10));
            _mm_storeu_pd(pdu + 0, _trg0);
            _mm256_storeu_pd(pdu + 2, _trg1);
            _mm256_storeu_pd(pdu + 6, _trg2);
            _mm256_storeu_pd(pdu + 10, _trg3);

            pdu += 2 * size1;
            _kerval = _mm256_set1_pd(ker2[n][6] * ker3[n][dz]);
            _trg0 = _mm_fmadd_pd(_ker1val0, _mm256_castpd256_pd128(_kerval), _mm_load_pd(pdu + 0));
            _trg1 = _mm256_fmadd_pd(_ker1val1, _kerval, _mm256_load_pd(pdu + 2));
            _trg2 = _mm256_fmadd_pd(_ker1val2, _kerval, _mm256_load_pd(pdu + 6));
            _trg3 = _mm256_fmadd_pd(_ker1val3, _kerval, _mm256_load_pd(pdu + 10));
            _mm_storeu_pd(pdu + 0, _trg0);
            _mm256_storeu_pd(pdu + 2, _trg1);
            _mm256_storeu_pd(pdu + 6, _trg2);
            _mm256_storeu_pd(pdu + 10, _trg3);
        }
    }
}
#endif
#endif
#endif

#endif // PIPEDSPREAD_H