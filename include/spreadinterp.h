// Defines interface to spreading/interpolation code.
// Note: see defs.h for definition of MAX_NSPREAD (as of 9/24/18).
// RESCALE macro moved to spreadinterp.cpp, 7/15/20.

#ifndef SPREADINTERP_H
#define SPREADINTERP_H

#include <dataTypes.h>
#include <spread_opts.h>

/* Bitwise debugging timing flag (TF) definitions; see spread_opts.flags.
    This is an unobtrusive way to determine the time contributions of the
    different components of spreading/interp by selectively leaving them out.
    For example, running the following two tests shows the effect of the exp()
    in the kernel evaluation (the last argument is the flag):
    > perftest/spreadtestnd 3 8e6 8e6 1e-6 1 0 0 1 0
    > perftest/spreadtestnd 3 8e6 8e6 1e-6 1 4 0 1 0
    NOTE: non-zero values are for experts only, since
    NUMERICAL OUTPUT MAY BE INCORRECT UNLESS spread_opts.flags=0 !
*/
#define TF_OMIT_WRITE_TO_GRID        1 // don't add subgrids to out grid (dir=1)
#define TF_OMIT_EVALUATE_KERNEL      2 // don't evaluate the kernel at all
#define TF_OMIT_EVALUATE_EXPONENTIAL 4 // omit exp() in kernel (kereval=0 only)
#define TF_OMIT_SPREADING            8 // don't interp/spread (dir=1: to subgrids)

// things external (spreadinterp) interface needs...
int spreadinterp(BIGINT N1, BIGINT N2, BIGINT N3, FLT *data_uniform,
		 BIGINT M, FLT *kx, FLT *ky, FLT *kz,
		 FLT *data_nonuniform, const spread_opts& opts);
int spreadcheck(BIGINT N1, BIGINT N2, BIGINT N3,
                 BIGINT M, FLT *kx, FLT *ky, FLT *kz, const spread_opts& opts);
int indexSort(BIGINT* sort_indices, BIGINT N1, BIGINT N2, BIGINT N3, BIGINT M, 
               FLT *kx, FLT *ky, FLT *kz, const spread_opts& opts);
int interpSorted(BIGINT* sort_indices,BIGINT N1, BIGINT N2, BIGINT N3, 
		      FLT *data_uniform,BIGINT M, FLT *kx, FLT *ky, FLT *kz,
		 FLT *data_nonuniform, const spread_opts& opts, int did_sort);
int spreadSorted(BIGINT* sort_indices,BIGINT N1, BIGINT N2, BIGINT N3,
		      FLT *data_uniform,BIGINT M, FLT *kx, FLT *ky, FLT *kz,
		 FLT *data_nonuniform, const spread_opts& opts, int did_sort);
int spreadinterpSorted(BIGINT* sort_indices,BIGINT N1, BIGINT N2, BIGINT N3, 
		      FLT *data_uniform,BIGINT M, FLT *kx, FLT *ky, FLT *kz,
		      FLT *data_nonuniform, const spread_opts& opts, int did_sort);

int setup_spreader(spread_opts &opts,FLT eps,double upsampfac,int kerevalmeth, int debug, int showwarn, int dim);
void get_subgrid(BIGINT& offset, BIGINT& size, BIGINT* idx, BIGINT M, int ns);


inline int ndims_from_Ns(BIGINT N1, BIGINT N2, BIGINT N3)
/* rule for getting number of spreading dimensions from the list of Ns per dim.
   Split out, Barnett 7/26/18
*/
{
	int ndims = 1;                // decide ndims: 1,2 or 3
	if (N2 > 1) ++ndims;
	if (N3 > 1) ++ndims;
	return ndims;
}

#endif  // SPREADINTERP_H
