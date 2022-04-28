#include <spreadinterp.h>
#include <dataTypes.h>
#include <defs.h>
#include <utils.h>
#include <utils_precindep.h>

#include <interp.h>
#include <foldrescale.h>
#include <spread.h>

#include <stdlib.h>
#include <vector>
#include <math.h>
#include <stdio.h>

#include <tbb/tbb.h>

using namespace std;

// declarations of purely internal functions...
void add_wrapped_subgrid(BIGINT offset1, BIGINT offset2, BIGINT offset3,
	BIGINT size1, BIGINT size2, BIGINT size3, BIGINT N1,
	BIGINT N2, BIGINT N3, FLT* data_uniform, FLT* du0);
void add_wrapped_subgrid_thread_safe(BIGINT offset1, BIGINT offset2, BIGINT offset3,
	BIGINT size1, BIGINT size2, BIGINT size3, BIGINT N1,
	BIGINT N2, BIGINT N3, FLT* data_uniform, FLT* du0);
void bin_sort_singlethread(BIGINT* ret, BIGINT M, FLT* kx, FLT* ky, FLT* kz,
	BIGINT N1, BIGINT N2, BIGINT N3, int pirange,
	double bin_size_x, double bin_size_y, double bin_size_z, int debug);
void bin_sort_multithread(BIGINT* ret, BIGINT M, FLT* kx, FLT* ky, FLT* kz,
	BIGINT N1, BIGINT N2, BIGINT N3, int pirange,
	double bin_size_x, double bin_size_y, double bin_size_z, int debug,
	int nthr);

// ==========================================================================
int spreadinterp(
	BIGINT N1, BIGINT N2, BIGINT N3, FLT* data_uniform,
	BIGINT M, FLT* kx, FLT* ky, FLT* kz, FLT* data_nonuniform,
	spread_opts opts)
	/* ------------Spreader/interpolator for 1, 2, or 3 dimensions --------------
	   If opts.spread_direction=1, evaluate, in the 1D case,

							 N1-1
	   data_nonuniform[j] =  SUM phi(kx[j] - n) data_uniform[n],   for j=0...M-1
							 n=0

	   If opts.spread_direction=2, evaluate its transpose, in the 1D case,

						  M-1
	   data_uniform[n] =  SUM phi(kx[j] - n) data_nonuniform[j],   for n=0...N1-1
						  j=0

	   In each case phi is the spreading kernel, which has support
	   [-opts.nspread/2,opts.nspread/2]. In 2D or 3D, the generalization with
	   product of 1D kernels is performed.
	   For 1D set N2=N3=1; for 2D set N3=1; for 3D set N1,N2,N3>1.

	   Notes:
	   No particular normalization of the spreading kernel is assumed.
	   Uniform (U) points are centered at coords
	   [0,1,...,N1-1] in 1D, analogously in 2D and 3D. They are stored in x
	   fastest, y medium, z slowest ordering, up to however many
	   dimensions are relevant; note that this is Fortran-style ordering for an
	   array f(x,y,z), but C style for f[z][y][x]. This is to match the Fortran
	   interface of the original CMCL libraries.
	   Non-uniform (NU) points kx,ky,kz are real, and may lie in the central three
	   periods in each coordinate (these are folded into the central period).
	   If pirange=0, the periodic domain for kx is [0,N1], ky [0,N2], kz [0,N3].
	   If pirange=1, the periodic domain is instead [-pi,pi] for each coord.
	   The spread_opts struct must have been set up already by calling setup_kernel.
	   It is assumed that 2*opts.nspread < min(N1,N2,N3), so that the kernel
	   only ever wraps once when falls below 0 or off the top of a uniform grid
	   dimension.

	   Inputs:
	   N1,N2,N3 - grid sizes in x (fastest), y (medium), z (slowest) respectively.
				  If N2==1, 1D spreading is done. If N3==1, 2D spreading.
			  Otherwise, 3D.
	   M - number of NU pts.
	   kx, ky, kz - length-M real arrays of NU point coordinates (only kx read in
					1D, only kx and ky read in 2D).

			These should lie in the box 0<=kx<=N1 etc (if pirange=0),
					or -pi<=kx<=pi (if pirange=1). However, points up to +-1 period
					outside this domain are also correctly folded back into this
					domain, but pts beyond this either raise an error (if chkbnds=1)
					or a crash (if chkbnds=0).
	   opts - spread/interp options struct, documented in ../include/spread_opts.h

	   Inputs/Outputs:
	   data_uniform - output values on grid (dir=1) OR input grid data (dir=2)
	   data_nonuniform - input strengths of the sources (dir=1)
						 OR output values at targets (dir=2)
	   Returned value:
	   0 indicates success; other values have meanings in ../docs/error.rst, with
	   following modifications:
		  3 : one or more non-trivial box dimensions is less than 2.nspread.
		  4 : nonuniform points outside [-Nm,2*Nm] or [-3pi,3pi] in at least one
			  dimension m=1,2,3.
		  5 : failed allocate sort indices

	   Magland Dec 2016. Barnett openmp version, many speedups 1/16/17-2/16/17
	   error codes 3/13/17. pirange 3/28/17. Rewritten 6/15/17. parallel sort 2/9/18
	   No separate subprob indices in t-1 2/11/18.
	   sort_threads (since for M<<N, multithread sort slower than single) 3/27/18
	   kereval, kerpad 4/24/18
	   Melody Shih split into 3 routines: check, sort, spread. Jun 2018, making
	   this routine just a caller to them. Name change, Barnett 7/27/18
	   Tidy, Barnett 5/20/20. Tidy doc, Barnett 10/22/20.
	*/
{
	int ier = spreadcheck(N1, N2, N3, M, kx, ky, kz, opts);
	if (ier)
		return ier;
	BIGINT* sort_indices = (BIGINT*)malloc(sizeof(BIGINT) * M);
	if (!sort_indices) {
		fprintf(stderr, "%s failed to allocate sort_indices!\n", __func__);
		return ERR_SPREAD_ALLOC;
	}
	int did_sort = indexSort(sort_indices, N1, N2, N3, M, kx, ky, kz, opts);
	spreadinterpSorted(sort_indices, N1, N2, N3, data_uniform,
		M, kx, ky, kz, data_nonuniform, opts, did_sort);
	free(sort_indices);
	return 0;
}

static int ndims_from_Ns(BIGINT N1, BIGINT N2, BIGINT N3)
/* rule for getting number of spreading dimensions from the list of Ns per dim.
   Split out, Barnett 7/26/18
*/
{
	int ndims = 1;                // decide ndims: 1,2 or 3
	if (N2 > 1) ++ndims;
	if (N3 > 1) ++ndims;
	return ndims;
}

int spreadcheck(BIGINT N1, BIGINT N2, BIGINT N3, BIGINT M, FLT* kx, FLT* ky,
	FLT* kz, spread_opts opts)
	/* This does just the input checking and reporting for the spreader.
	   See spreadinterp() for input arguments and meaning of returned value.
	   Split out by Melody Shih, Jun 2018. Finiteness chk Barnett 7/30/18.
	   Bypass FOLDRESCALE macro which has inevitable rounding err even nr +pi,
	   giving fake invalids well inside the [-3pi,3pi] domain, 4/9/21.
	*/
{
	CNTime timer;
	// INPUT CHECKING & REPORTING .... cuboid not too small for spreading?
	int minN = 2 * opts.nspread;
	if (N1 < minN || (N2 > 1 && N2 < minN) || (N3 > 1 && N3 < minN)) {
		fprintf(stderr, "%s error: one or more non-trivial box dims is less than 2.nspread!\n", __func__);
		return ERR_SPREAD_BOX_SMALL;
	}
	if (opts.spread_direction != 1 && opts.spread_direction != 2) {
		fprintf(stderr, "%s error: opts.spread_direction must be 1 or 2!\n", __func__);
		return ERR_SPREAD_DIR;
	}
	int ndims = ndims_from_Ns(N1, N2, N3);

	// BOUNDS CHECKING .... check NU pts are valid (+-3pi if pirange, or [-N,2N])
	// exit gracefully as soon as invalid is found.
	// Note: isfinite() breaks with -Ofast
	if (opts.chkbnds) {
		timer.start();
		for (BIGINT i = 0; i < M; ++i) {
			if ((opts.pirange ? (abs(kx[i]) > 3.0 * PI) : (kx[i] < -N1 || kx[i]>2 * N1)) || !isfinite(kx[i])) {
				fprintf(stderr, "%s NU pt not in valid range (central three periods): kx[%lld]=%.16g, N1=%lld (pirange=%d)\n", __func__, (long long)i, kx[i], (long long)N1, opts.pirange);
				return ERR_SPREAD_PTS_OUT_RANGE;
			}
		}
		if (ndims > 1)
			for (BIGINT i = 0; i < M; ++i) {
				if ((opts.pirange ? (abs(ky[i]) > 3.0 * PI) : (ky[i] < -N2 || ky[i]>2 * N2)) || !isfinite(ky[i])) {
					fprintf(stderr, "%s NU pt not in valid range (central three periods): ky[%lld]=%.16g, N2=%lld (pirange=%d)\n", __func__, (long long)i, ky[i], (long long)N2, opts.pirange);
					return ERR_SPREAD_PTS_OUT_RANGE;
				}
			}
		if (ndims > 2)
			for (BIGINT i = 0; i < M; ++i) {
				if ((opts.pirange ? (abs(kz[i]) > 3.0 * PI) : (kz[i] < -N3 || kz[i]>2 * N3)) || !isfinite(kz[i])) {
					fprintf(stderr, "%s NU pt not in valid range (central three periods): kz[%lld]=%.16g, N3=%lld (pirange=%d)\n", __func__, (long long)i, kz[i], (long long)N3, opts.pirange);
					return ERR_SPREAD_PTS_OUT_RANGE;
				}
			}
		if (opts.debug) printf("\tNU bnds check:\t\t%.3g s\n", timer.elapsedsec());
	}
	return 0;
}


int indexSort(BIGINT* sort_indices, BIGINT N1, BIGINT N2, BIGINT N3, BIGINT M,
	FLT* kx, FLT* ky, FLT* kz, spread_opts opts)
	/* This makes a decision whether or not to sort the NU pts (influenced by
	   opts.sort), and if yes, calls either single- or multi-threaded bin sort,
	   writing reordered index list to sort_indices. If decided not to sort, the
	   identity permutation is written to sort_indices.
	   The permutation is designed to make RAM access close to contiguous, to
	   speed up spreading/interpolation, in the case of disordered NU points.

	   Inputs:
		M        - number of input NU points.
		kx,ky,kz - length-M arrays of real coords of NU pts, in the domain
				   for FOLDRESCALE, which includes [0,N1], [0,N2], [0,N3]
				   respectively, if opts.pirange=0; or [-pi,pi] if opts.pirange=1.
				   (only kz used in 1D, only kx and ky used in 2D.)
				   These must have been bounds-checked already; see spreadcheck.
		N1,N2,N3 - integer sizes of overall box (set N2=N3=1 for 1D, N3=1 for 2D).
				   1 = x (fastest), 2 = y (medium), 3 = z (slowest).
		opts     - spreading options struct, documented in ../include/spread_opts.h
	   Outputs:
		sort_indices - a good permutation of NU points. (User must preallocate
					   to length M.) Ie, kx[sort_indices[j]], j=0,..,M-1, is a good
					   ordering for the x-coords of NU pts, etc.
		returned value - whether a sort was done (1) or not (0).

	   Barnett 2017; split out by Melody Shih, Jun 2018.
	*/
{
	CNTime timer;
	int ndims = ndims_from_Ns(N1, N2, N3);
	BIGINT N = N1 * N2 * N3;            // U grid (periodic box) sizes

	// heuristic binning box size for U grid... affects performance:
	double bin_size_x = 16, bin_size_y = 4, bin_size_z = 4;
	// put in heuristics based on cache sizes (only useful for single-thread) ?

	int better_to_sort = !(ndims == 1 && (opts.spread_direction == 2 || (M > 1000 * N1))); // 1D small-N or dir=2 case: don't sort

	timer.start();                 // if needed, sort all the NU pts...
	int did_sort = 0;
	int maxnthr = MY_OMP_GET_MAX_THREADS();
	if (opts.nthreads > 0)           // user override up to max avail
		maxnthr = min(maxnthr, opts.nthreads);

	if (opts.sort == 1 || (opts.sort == 2 && better_to_sort)) {
		// store a good permutation ordering of all NU pts (dim=1,2 or 3)
		int sort_debug = (opts.debug >= 2);    // show timing output?
		int sort_nthr = opts.sort_threads;   // choose # threads for sorting
		if (sort_nthr == 0)   // use auto choice: when N>>M, one thread is better!
			sort_nthr = (10 * M > N) ? maxnthr : 1;      // heuristic
		if (sort_nthr == 1)
			bin_sort_singlethread(sort_indices, M, kx, ky, kz, N1, N2, N3, opts.pirange, bin_size_x, bin_size_y, bin_size_z, sort_debug);
		else                                      // sort_nthr>1, sets # threads
			bin_sort_multithread(sort_indices, M, kx, ky, kz, N1, N2, N3, opts.pirange, bin_size_x, bin_size_y, bin_size_z, sort_debug, sort_nthr);
		if (opts.debug)
			printf("\tsorted (%d threads):\t%.3g s\n", sort_nthr, timer.elapsedsec());
		did_sort = 1;
	}
	else {
#pragma omp parallel for num_threads(maxnthr) schedule(static,1000000)
		for (BIGINT i = 0; i < M; i++)                // here omp helps xeon, hinders i7
			sort_indices[i] = i;                      // the identity permutation
		if (opts.debug)
			printf("\tnot sorted (sort=%d): \t%.3g s\n", (int)opts.sort, timer.elapsedsec());
	}
	return did_sort;
}


int spreadinterpSorted(BIGINT* sort_indices, BIGINT N1, BIGINT N2, BIGINT N3,
	FLT* data_uniform, BIGINT M, FLT* kx, FLT* ky, FLT* kz,
	FLT* data_nonuniform, spread_opts opts, int did_sort)
	/* Logic to select the main spreading (dir=1) vs interpolation (dir=2) routine.
	   See spreadinterp() above for inputs arguments and definitions.
	   Return value should always be 0 (no error reporting).
	   Split out by Melody Shih, Jun 2018; renamed Barnett 5/20/20.
	*/
{
	if (opts.spread_direction == 1)  // ========= direction 1 (spreading) =======
		spreadSorted(sort_indices, N1, N2, N3, data_uniform, M, kx, ky, kz, data_nonuniform, opts, did_sort);

	else           // ================= direction 2 (interpolation) ===========
		interpSorted(sort_indices, N1, N2, N3, data_uniform, M, kx, ky, kz, data_nonuniform, opts, did_sort);

	return 0;
}

// --------------------------------------------------------------------------
int spreadSorted(BIGINT* sort_indices, BIGINT N1, BIGINT N2, BIGINT N3,
	FLT* data_uniform, BIGINT M, FLT* kx, FLT* ky, FLT* kz,
	FLT* data_nonuniform, spread_opts opts, int did_sort)
	// Spread NU pts in sorted order to a uniform grid. See spreadinterp() for doc.
{
	CNTime timer;
	int ndims = ndims_from_Ns(N1, N2, N3);
	BIGINT N = N1 * N2 * N3;            // output array size
	int ns = opts.nspread;          // abbrev. for w, kernel width
	FLT ns2 = (FLT)ns / 2;          // half spread width
	int nsPadded = 4 * (1 + (ns - 1) / 4); // pad ns to mult of 4
	int nthr = MY_OMP_GET_MAX_THREADS();  // # threads to use to spread
	if (opts.nthreads > 0)
		nthr = min(nthr, opts.nthreads);     // user override up to max avail
	if (opts.debug)
		printf("\tspread %dD (M=%lld; N1=%lld,N2=%lld,N3=%lld; pir=%d), nthr=%d\n", ndims, (long long)M, (long long)N1, (long long)N2, (long long)N3, opts.pirange, nthr);

	timer.start();
	for (BIGINT i = 0; i < 2 * N; i++) // zero the output array. std::fill is no faster
		data_uniform[i] = 0.0;
	if (opts.debug) printf("\tzero output array\t%.3g s\n", timer.elapsedsec());
	if (M == 0)                     // no NU pts, we're done
		return 0;

	int spread_single = (nthr == 1) || (M * 100 < N);     // low-density heuristic?
	spread_single = 0;                 // for now
	timer.start();
	if (spread_single) {    // ------- Basic single-core t1 spreading ------
		for (BIGINT j = 0; j < M; j++) {
			// *** todo, not urgent
			// ... (question is: will the index wrapping per NU pt slow it down?)
		}
		if (opts.debug) printf("\tt1 simple spreading:\t%.3g s\n", timer.elapsedsec());

	}
	else {           // ------- Fancy multi-core blocked t1 spreading ----
					// Splits sorted inds (jfm's advanced2), could double RAM.
   // choose nb (# subprobs) via used nthreads:
		BIGINT nb = min((BIGINT)nthr, M);         // simply split one subprob per thr...
		if (nb * (BIGINT)opts.max_subproblem_size < M) {  // ...or more subprobs to cap size
			nb = 1 + (M - 1) / opts.max_subproblem_size;  // int div does ceil(M/opts.max_subproblem_size)
			if (opts.debug) printf("\tcapping subproblem sizes to max of %d\n", opts.max_subproblem_size);
		}
		if (M * 1000 < N) {         // low-density heuristic: one thread per NU pt!
			nb = M;
			if (opts.debug) printf("\tusing low-density speed rescue nb=M...\n");
		}
		if (!did_sort && nthr == 1) {
			nb = 1;
			if (opts.debug) printf("\tunsorted nthr=1: forcing single subproblem...\n");
		}
		if (opts.debug && nthr > opts.atomic_threshold)
			printf("\tnthr big: switching add_wrapped OMP from critical to atomic (!)\n");

		BIGINT* i1 = NULL, * i2 = NULL, * i3 = NULL;
		FLT* x1 = NULL, * x2 = NULL, * x3 = NULL;
		FLT* kernel_vals1 = NULL, * kernel_vals2 = NULL, * kernel_vals3 = NULL;

		switch (ndims) {
		case 3:
			i3 = (BIGINT*)malloc(sizeof(BIGINT) * M);
			x3 = (FLT*)malloc(sizeof(FLT) * M);
			kernel_vals3 = (FLT*)_mm_malloc(nsPadded * M * sizeof(FLT), 64);
			// Fall through
		case 2:
			i2 = (BIGINT*)malloc(sizeof(BIGINT) * M);
			x2 = (FLT*)malloc(sizeof(FLT) * M);
			kernel_vals2 = (FLT*)_mm_malloc(nsPadded * M * sizeof(FLT), 64);
			// Fall through
		case 1:
			i1 = (BIGINT*)malloc(sizeof(BIGINT) * M);
			x1 = (FLT*)malloc(sizeof(FLT) * M);
			kernel_vals1 = (FLT*)_mm_malloc(nsPadded * M * sizeof(FLT), 64);
		}

		tbb::parallel_for(tbb::blocked_range<BIGINT>(0, M, 10000),
			[&](const tbb::blocked_range<BIGINT>& r) {
				// get the subgrid which will include padding by roughly nspread/2
				BIGINT offset1 = 0, offset2 = 0, offset3 = 0;
				BIGINT size1 = 1, size2 = 1, size3 = 1; // get_subgrid set

				switch (ndims) {
				case 3:
					foldrescale<FLT>(sort_indices, kz, i3, x3, N3, r.begin(), r.end(), opts);
					get_subgrid(offset3, size3, i3 + r.begin(), r.size(), ns);
					// Fall through
				case 2:
					foldrescale<FLT>(sort_indices, ky, i2, x2, N2, r.begin(), r.end(), opts);
					get_subgrid(offset2, size2, i2 + r.begin(), r.size(), ns);
					// Fall through
				case 1:
					foldrescale<FLT>(sort_indices, kx, i1, x1, N1, r.begin(), r.end(), opts);
					get_subgrid(offset1, size1, i1 + r.begin(), r.size(), ns);

					if (ndims == 1) {
						// x1 in [-w/2,-w/2+1], up to rounding
						// However if N1*epsmach>O(1) then can cause O(1) errors in x1, hence ppoly
						// kernel evaluation will fall outside their designed domains, >>1 errors.
						// This can only happen if the overall error would be O(1) anyway. Clip x1??
						for (BIGINT i = r.begin(); i < r.end(); i++) {           // loop over NU pts
							if (x1[i] < -ns2) x1[i] = -ns2;
							if (x1[i] > -ns2 + 1) x1[i] = -ns2 + 1;   // ***
						}
					}

					break;
				}

				if (opts.debug > 1) { // verbose
					if (ndims == 1)
						printf("\tsubgrid: off %lld\t siz %lld\t #NU %lld\n", (long long)offset1, (long long)size1, (long long)r.size());
					else if (ndims == 2)
						printf("\tsubgrid: off %lld,%lld\t siz %lld,%lld\t #NU %lld\n", (long long)offset1, (long long)offset2, (long long)size1, (long long)size2, (long long)r.size());
					else
						printf("\tsubgrid: off %lld,%lld,%lld\t siz %lld,%lld,%lld\t #NU %lld\n", (long long)offset1, (long long)offset2, (long long)offset3, (long long)size1, (long long)size2, (long long)size3, (long long)r.size());
				}
				// allocate output data for this subgrid
				// extra MAX_NSPREAD is for the vector spillover
				// 2x factor for complex
				FLT* du0 = (FLT*)malloc(sizeof(FLT) * 2 * (size1 * size2 * size3 + MAX_NSPREAD));

				// Spread to subgrid without need for bounds checking or wrapping
				if (!(opts.flags & TF_OMIT_SPREADING)) {
					switch (ndims) {
					case 1:
						evaluate_kernel(kernel_vals1, x1, r.begin(), r.end(), opts);
						spread_subproblem_1d<FLT>(sort_indices, offset1, size1, du0, data_nonuniform,
							i1, kernel_vals1,
							r.begin(), r.end(), opts);
						break;
					case 2:
						evaluate_kernel(kernel_vals1, x1, r.begin(), r.end(), opts);
						evaluate_kernel(kernel_vals2, x2, r.begin(), r.end(), opts);
						spread_subproblem_2d<FLT>(sort_indices, offset1, offset2, size1, size2,
							du0, data_nonuniform,
							i1, i2, kernel_vals1, kernel_vals2,
							r.begin(), r.end(), opts);
						break;
					case 3:
						evaluate_kernel(kernel_vals1, x1, r.begin(), r.end(), opts);
						evaluate_kernel(kernel_vals2, x2, r.begin(), r.end(), opts);
						evaluate_kernel(kernel_vals3, x3, r.begin(), r.end(), opts);
						spread_subproblem_3d<FLT>(sort_indices, offset1, offset2, offset3, size1, size2, size3,
							du0, data_nonuniform,
							i1, i2, i3, kernel_vals1, kernel_vals2, kernel_vals3,
							r.begin(), r.end(), opts);
						break;
					}
				}

				// do the adding of subgrid to output
				if (!(opts.flags & TF_OMIT_WRITE_TO_GRID)) {
					if (nthr > opts.atomic_threshold)   // see above for debug reporting
						add_wrapped_subgrid_thread_safe(offset1, offset2, offset3, size1, size2, size3, N1, N2, N3, data_uniform, du0);   // R Blackwell's atomic version
					else {
#pragma omp critical
						add_wrapped_subgrid(offset1, offset2, offset3, size1, size2, size3, N1, N2, N3, data_uniform, du0);
					}
				}

				// free up stuff from this subprob... (that was malloc'ed by hand)
				free(du0);
			}); // end main loop over subprobs

		switch (ndims) {
		case 3:
			_mm_free(kernel_vals3);
			free(x3);
			free(i3);
			// Fall through
		case 2:
			_mm_free(kernel_vals2);
			free(x2);
			free(i2);
			// Fall through
		case 1:
			_mm_free(kernel_vals1);
			free(x1);
			free(i1);
		}

		if (opts.debug) printf("\tt1 fancy spread: \t%.3g s (%lld subprobs)\n", timer.elapsedsec(), (long long)nb);
	}   // end of choice of which t1 spread type to use
	return 0;
}

static inline void padData_1d(FLT* data, FLT* data_padded, BIGINT N1) {
    copy(data, data + 2 * N1, data_padded + 2 * MAX_NSPREAD);
    copy(data, data + 2 * MAX_NSPREAD, data_padded + 2 * (N1 + MAX_NSPREAD));
    copy(data + 2 * (N1 - MAX_NSPREAD), data + 2 * N1, data_padded);
}

static inline void padData_2d(FLT* data, FLT* data_padded, BIGINT N1, BIGINT N2) {
    BIGINT paddedN1 = N1 + 2 * MAX_NSPREAD;
    for (BIGINT j = 0; j < N2; j++) {
        padData_1d(data + 2 * N1 * j, data_padded + 2 * paddedN1 * (j + MAX_NSPREAD), N1);
    }
    copy(data_padded + 2 * paddedN1 * N2, data_padded + 2 * paddedN1 * (N2 + MAX_NSPREAD), data_padded);
    copy(data_padded + 2 * paddedN1 * MAX_NSPREAD, data_padded + 4 * paddedN1 * MAX_NSPREAD, data_padded + 2 * paddedN1 * (N2 + MAX_NSPREAD));
}

static inline void padData_3d(FLT* data, FLT* data_padded, BIGINT N1, BIGINT N2, BIGINT N3) {
    BIGINT paddedN1 = N1 + 2 * MAX_NSPREAD;
    BIGINT paddedN2 = N2 + 2 * MAX_NSPREAD;
    for (BIGINT k = 0; k < N3; k++) {
        padData_2d(data + 2 * N1 * N2 * k, data_padded + 2 * paddedN1 * paddedN2 * (k + MAX_NSPREAD), N1, N2);
    }
    copy(data_padded + 2 * paddedN1 * paddedN2 * N3, data_padded + 2 * paddedN1 * paddedN2 * (N3 + MAX_NSPREAD), data_padded);
    copy(data_padded + 2 * paddedN1 * paddedN2 * MAX_NSPREAD, data_padded + 4 * paddedN1 * paddedN2 * MAX_NSPREAD, data_padded + 2 * paddedN1 * paddedN2 * (N3 + MAX_NSPREAD));
}

// --------------------------------------------------------------------------
int interpSorted(BIGINT* sort_indices, BIGINT N1, BIGINT N2, BIGINT N3,
    FLT* data_uniform, BIGINT M, FLT* kx, FLT* ky, FLT* kz,
    FLT* data_nonuniform, spread_opts opts, int did_sort)
    // Interpolate to NU pts in sorted order from a uniform grid.
    // See spreadinterp() for doc.
{
    CNTime timer;
    int ndims = ndims_from_Ns(N1, N2, N3);

    int ns = opts.nspread;          // abbrev. for w, kernel width
    FLT ns2 = (FLT)ns / 2;          // half spread width, used as stencil shift
    int nthr = MY_OMP_GET_MAX_THREADS();   // # threads to use to interp
    int nsPadded = 4 * (1 + (ns - 1) / 4); // pad ns to mult of 4
    if (opts.nthreads > 0)
        nthr = min(nthr, opts.nthreads);      // user override up to max avail
    if (opts.debug)
        printf("\tinterp %dD (M=%lld; N1=%lld,N2=%lld,N3=%lld; pir=%d), nthr=%d\n", ndims, (long long)M, (long long)N1, (long long)N2, (long long)N3, opts.pirange, nthr);

    timer.start();

    BIGINT* i1 = NULL, * i2 = NULL, * i3 = NULL;
    FLT* x1 = NULL, * x2 = NULL, * x3 = NULL;
    FLT* kernel_vals1 = NULL, * kernel_vals2 = NULL, * kernel_vals3 = NULL;

    switch (ndims) {
    case 3:
        i3 = (BIGINT*)malloc(sizeof(BIGINT) * M);
        x3 = (FLT*)malloc(sizeof(FLT) * M);
        kernel_vals3 = (FLT*)_mm_malloc(nsPadded * M * sizeof(FLT), 64);
        // Fall through
    case 2:
        i2 = (BIGINT*)malloc(sizeof(BIGINT) * M);
        x2 = (FLT*)malloc(sizeof(FLT) * M);
        kernel_vals2 = (FLT*)_mm_malloc(nsPadded * M * sizeof(FLT), 64);
        // Fall through
    case 1:
        i1 = (BIGINT*)malloc(sizeof(BIGINT) * M);
        x1 = (FLT*)malloc(sizeof(FLT) * M);
        kernel_vals1 = (FLT*)_mm_malloc(nsPadded * M * sizeof(FLT), 64);
        break;
    }

    BIGINT paddedN = 1, paddedN1, paddedN2, paddedN3;
    switch (ndims) {
    case 3:
        paddedN3 = N3 + 2 * MAX_NSPREAD;
        paddedN *= paddedN3;
        // fall through
    case 2:
        paddedN2 = N2 + 2 * MAX_NSPREAD;
        paddedN *= paddedN2;
        // fall through
    case 1:
        paddedN1 = N1 + 2 * MAX_NSPREAD;
        paddedN *= paddedN1;
        break;
    }

    FLT* du_padded = (FLT*)malloc(2 * sizeof(FLT) * paddedN);

    // eval kernel values patch and use to interpolate from uniform data...
    if (!(opts.flags & TF_OMIT_SPREADING)) {
        switch (ndims) {
        case 1:
            padData_1d(data_uniform, du_padded, N1);

            tbb::parallel_for(tbb::blocked_range<BIGINT>(0, M, 10000),
                [&](const tbb::blocked_range<BIGINT>& r) {
                    foldrescale(sort_indices, kx, i1, x1, N1, r.begin(), r.end(), opts);

                    evaluate_kernel(kernel_vals1, x1, r.begin(), r.end(), opts);

                    interp_line<FLT>(sort_indices, data_nonuniform, du_padded + 2 * MAX_NSPREAD,
                        kernel_vals1,
                        i1,
                        N1,
                        ns, r.begin(), r.end());
                });
            break;
        case 2:
            padData_2d(data_uniform, du_padded, N1, N2);

            tbb::parallel_for(tbb::blocked_range<BIGINT>(0, M, 10000),
                [&](const tbb::blocked_range<BIGINT>& r) {
                    foldrescale(sort_indices, kx, i1, x1, N1, r.begin(), r.end(), opts);
                    foldrescale(sort_indices, ky, i2, x2, N2, r.begin(), r.end(), opts);

                    evaluate_kernel(kernel_vals1, x1, r.begin(), r.end(), opts);
                    evaluate_kernel(kernel_vals2, x2, r.begin(), r.end(), opts);

                    interp_square<FLT>(sort_indices, data_nonuniform, du_padded + 2 * MAX_NSPREAD * (paddedN1 + 1),
                        kernel_vals1, kernel_vals2,
                        i1, i2,
                        N1, N2,
                        ns, r.begin(), r.end());
                });
            break;
        case 3:
            padData_3d(data_uniform, du_padded, N1, N2, N3);

            tbb::parallel_for(tbb::blocked_range<BIGINT>(0, M, 10000),
                [&](const tbb::blocked_range<BIGINT>& r) {
                    foldrescale(sort_indices, kx, i1, x1, N1, r.begin(), r.end(), opts);
                    foldrescale(sort_indices, ky, i2, x2, N2, r.begin(), r.end(), opts);
                    foldrescale(sort_indices, kz, i3, x3, N3, r.begin(), r.end(), opts);

                    evaluate_kernel(kernel_vals1, x1, r.begin(), r.end(), opts);
                    evaluate_kernel(kernel_vals2, x2, r.begin(), r.end(), opts);
                    evaluate_kernel(kernel_vals3, x3, r.begin(), r.end(), opts);


                    interp_cube<FLT>(sort_indices, data_nonuniform, du_padded + 2 * MAX_NSPREAD * (paddedN1 * paddedN2 + paddedN1 + 1),
                        kernel_vals1, kernel_vals2, kernel_vals3,
                        i1, i2, i3,
                        N1, N2, N3,
                        ns, r.begin(), r.end());
                });
            break;
        default: //can't get here
            break;
        }
    }

    free(du_padded);

    switch (ndims) {
    case 3:
        _mm_free(kernel_vals3);
		free(x3);
		free(i3);
        // Fall through
    case 2:
		_mm_free(kernel_vals2);
		free(x2);
		free(i2);
        // Fall through
    case 1:
		_mm_free(kernel_vals1);
		free(x1);
		free(i1);
    }

    if (opts.debug) printf("\tt2 spreading loop: \t%.3g s\n", timer.elapsedsec());
    return 0;
};


///////////////////////////////////////////////////////////////////////////

int setup_spreader(spread_opts& opts, FLT eps, double upsampfac,
	int kerevalmeth, int debug, int showwarn, int dim)
	/* Initializes spreader kernel parameters given desired NUFFT tolerance eps,
	   upsampling factor (=sigma in paper, or R in Dutt-Rokhlin), ker eval meth
	   (either 0:exp(sqrt()), 1: Horner ppval), and some debug-level flags.
	   Also sets all default options in spread_opts. See spread_opts.h for opts.
	   dim is spatial dimension (1,2, or 3).
	   See finufft.cpp:finufft_plan() for where upsampfac is set.
	   Must call this before any kernel evals done, otherwise segfault likely.
	   Returns:
		 0  : success
		 WARN_EPS_TOO_SMALL : requested eps cannot be achieved, but proceed with
							  best possible eps
		 otherwise : failure (see codes in defs.h); spreading must not proceed
	   Barnett 2017. debug, loosened eps logic 6/14/20.
	*/
{
	if (upsampfac != 2.0 && upsampfac != 1.25) {   // nonstandard sigma
		if (kerevalmeth == 1) {
			fprintf(stderr, "FINUFFT setup_spreader: nonstandard upsampfac=%.3g cannot be handled by kerevalmeth=1\n", upsampfac);
			return ERR_HORNER_WRONG_BETA;
		}
		if (upsampfac <= 1.0) {       // no digits would result
			fprintf(stderr, "FINUFFT setup_spreader: error, upsampfac=%.3g is <=1.0\n", upsampfac);
			return ERR_UPSAMPFAC_TOO_SMALL;
		}
		// calling routine must abort on above errors, since opts is garbage!
		if (showwarn && upsampfac > 4.0)
			fprintf(stderr, "FINUFFT setup_spreader warning: upsampfac=%.3g way too large to be beneficial.\n", upsampfac);
	}

	// write out default spread_opts (some overridden in setup_spreader_for_nufft)
	opts.spread_direction = 0;    // user should always set to 1 or 2 as desired
	opts.pirange = 1;             // user also should always set this
	opts.chkbnds = 0;
	opts.sort = 2;                // 2:auto-choice
	opts.kerpad = 0;              // affects only evaluate_kernel_vector
	opts.kerevalmeth = kerevalmeth;
	opts.upsampfac = upsampfac;
	opts.nthreads = 0;            // all avail
	opts.sort_threads = 0;        // 0:auto-choice
	// heuristic dir=1 chunking for nthr>>1, typical for intel i7 and skylake...
	opts.max_subproblem_size = (dim == 1) ? 10000 : 100000;
	opts.flags = 0;               // 0:no timing flags (>0 for experts only)
	opts.debug = 0;               // 0:no debug output
	// heuristic nthr above which switch OMP critical to atomic (add_wrapped...):
	opts.atomic_threshold = 10;   // R Blackwell's value

	int ns, ier = 0;  // Set kernel width w (aka ns, nspread) then copy to opts...
	if (eps < EPSILON) {            // safety; there's no hope of beating e_mach
		if (showwarn)
			fprintf(stderr, "%s warning: increasing tol=%.3g to eps_mach=%.3g.\n", __func__, (double)eps, (double)EPSILON);
		eps = EPSILON;              // only changes local copy (not any opts)
		ier = WARN_EPS_TOO_SMALL;
	}
	if (upsampfac == 2.0)           // standard sigma (see SISC paper)
		ns = std::ceil(-log10(eps / (FLT)10.0));          // 1 digit per power of 10
	else                          // custom sigma
		ns = std::ceil(-log(eps) / (PI * sqrt(1.0 - 1.0 / upsampfac)));  // formula, gam=1
	ns = max(2, ns);               // (we don't have ns=1 version yet)
	if (ns > MAX_NSPREAD) {         // clip to fit allocated arrays, Horner rules
		if (showwarn)
			fprintf(stderr, "%s warning: at upsampfac=%.3g, tol=%.3g would need kernel width ns=%d; clipping to max %d.\n", __func__,
				upsampfac, (double)eps, ns, MAX_NSPREAD);
		ns = MAX_NSPREAD;
		ier = WARN_EPS_TOO_SMALL;
	}
	opts.nspread = ns;

	// setup for reference kernel eval (via formula): select beta width param...
	// (even when kerevalmeth=1, this ker eval needed for FTs in onedim_*_kernel)
	opts.ES_halfwidth = (double)ns / 2;   // constants to help (see below routines)
	opts.ES_c = 4.0 / (double)(ns * ns);
	double betaoverns = 2.30;         // gives decent betas for default sigma=2.0
	if (ns == 2) betaoverns = 2.20;  // some small-width tweaks...
	if (ns == 3) betaoverns = 2.26;
	if (ns == 4) betaoverns = 2.38;
	if (upsampfac != 2.0) {          // again, override beta for custom sigma
		FLT gamma = 0.97;              // must match devel/gen_all_horner_C_code.m !
		betaoverns = gamma * PI * (1.0 - 1.0 / (2 * upsampfac));  // formula based on cutoff
	}
	opts.ES_beta = betaoverns * ns;   // set the kernel beta parameter
	if (debug)
		printf("%s (kerevalmeth=%d) eps=%.3g sigma=%.3g: chose ns=%d beta=%.3g\n", __func__, kerevalmeth, (double)eps, upsampfac, ns, opts.ES_beta);

	return ier;
}

void add_wrapped_subgrid(BIGINT offset1,BIGINT offset2,BIGINT offset3,
			 BIGINT size1,BIGINT size2,BIGINT size3,BIGINT N1,
			 BIGINT N2,BIGINT N3,FLT *data_uniform, FLT *du0)
/* Add a large subgrid (du0) to output grid (data_uniform),
   with periodic wrapping to N1,N2,N3 box.
   offset1,2,3 give the offset of the subgrid from the lowest corner of output.
   size1,2,3 give the size of subgrid.
   Works in all dims. Not thread-safe and must be called inside omp critical.
   Barnett 3/27/18 made separate routine, tried to speed up inner loop.
*/
{
	std::vector<BIGINT> o2(size2), o3(size3);
	BIGINT y = offset2, z = offset3;    // fill wrapped ptr lists in slower dims y,z...
	for (int i = 0; i < size2; ++i) {
		if (y < 0) y += N2;
		if (y >= N2) y -= N2;
		o2[i] = y++;
	}
	for (int i = 0; i < size3; ++i) {
		if (z < 0) z += N3;
		if (z >= N3) z -= N3;
		o3[i] = z++;
	}
	BIGINT nlo = (offset1 < 0) ? -offset1 : 0;          // # wrapping below in x
	BIGINT nhi = (offset1 + size1 > N1) ? offset1 + size1 - N1 : 0;    // " above in x
	// this triple loop works in all dims
	for (int dz = 0; dz < size3; dz++) {       // use ptr lists in each axis
		BIGINT oz = N1 * N2 * o3[dz];            // offset due to z (0 in <3D)
		for (int dy = 0; dy < size2; dy++) {
			BIGINT oy = oz + N1 * o2[dy];        // off due to y & z (0 in 1D)
			FLT* out = data_uniform + 2 * oy;
			FLT* in = du0 + 2 * size1 * (dy + size2 * dz);   // ptr to subgrid array
			BIGINT o = 2 * (offset1 + N1);         // 1d offset for output
			for (int j = 0; j < 2 * nlo; j++)        // j is really dx/2 (since re,im parts)
				out[j + o] += in[j];
			o = 2 * offset1;
			for (int j = 2 * nlo; j < 2 * (size1 - nhi); j++)
				out[j + o] += in[j];
			o = 2 * (offset1 - N1);
			for (int j = 2 * (size1 - nhi); j < 2 * size1; j++)
				out[j + o] += in[j];
		}
	}
}

void add_wrapped_subgrid_thread_safe(BIGINT offset1, BIGINT offset2, BIGINT offset3,
	BIGINT size1, BIGINT size2, BIGINT size3, BIGINT N1,
	BIGINT N2, BIGINT N3, FLT* data_uniform, FLT* du0)
	/* Add a large subgrid (du0) to output grid (data_uniform),
	   with periodic wrapping to N1,N2,N3 box.
	   offset1,2,3 give the offset of the subgrid from the lowest corner of output.
	   size1,2,3 give the size of subgrid.
	   Works in all dims. Thread-safe variant of the above routine,
	   using atomic writes (R Blackwell, Nov 2020).
	*/
{
	std::vector<BIGINT> o2(size2), o3(size3);
	BIGINT y = offset2, z = offset3;    // fill wrapped ptr lists in slower dims y,z...
	for (int i = 0; i < size2; ++i) {
		if (y < 0) y += N2;
		if (y >= N2) y -= N2;
		o2[i] = y++;
	}
	for (int i = 0; i < size3; ++i) {
		if (z < 0) z += N3;
		if (z >= N3) z -= N3;
		o3[i] = z++;
	}
	BIGINT nlo = (offset1 < 0) ? -offset1 : 0;          // # wrapping below in x
	BIGINT nhi = (offset1 + size1 > N1) ? offset1 + size1 - N1 : 0;    // " above in x
	// this triple loop works in all dims
	for (int dz = 0; dz < size3; dz++) {       // use ptr lists in each axis
		BIGINT oz = N1 * N2 * o3[dz];            // offset due to z (0 in <3D)
		for (int dy = 0; dy < size2; dy++) {
			BIGINT oy = oz + N1 * o2[dy];        // off due to y & z (0 in 1D)
			FLT* out = data_uniform + 2 * oy;
			FLT* in = du0 + 2 * size1 * (dy + size2 * dz);   // ptr to subgrid array
			BIGINT o = 2 * (offset1 + N1);         // 1d offset for output
			for (int j = 0; j < 2 * nlo; j++) { // j is really dx/2 (since re,im parts)
#pragma omp atomic
				out[j + o] += in[j];
			}
			o = 2 * offset1;
			for (int j = 2 * nlo; j < 2 * (size1 - nhi); j++) {
#pragma omp atomic
				out[j + o] += in[j];
			}
			o = 2 * (offset1 - N1);
			for (int j = 2 * (size1 - nhi); j < 2 * size1; j++) {
#pragma omp atomic
				out[j + o] += in[j];
			}
		}
	}
}


void bin_sort_singlethread(BIGINT* ret, BIGINT M, FLT* kx, FLT* ky, FLT* kz,
	BIGINT N1, BIGINT N2, BIGINT N3, int pirange,
	double bin_size_x, double bin_size_y, double bin_size_z, int debug)
	/* Returns permutation of all nonuniform points with good RAM access,
	 * ie less cache misses for spreading, in 1D, 2D, or 3D. Single-threaded version
	 *
	 * This is achieved by binning into cuboids (of given bin_size within the
	 * overall box domain), then reading out the indices within
	 * these bins in a Cartesian cuboid ordering (x fastest, y med, z slowest).
	 * Finally the permutation is inverted, so that the good ordering is: the
	 * NU pt of index ret[0], the NU pt of index ret[1],..., NU pt of index ret[M-1]
	 *
	 * Inputs: M - number of input NU points.
	 *         kx,ky,kz - length-M arrays of real coords of NU pts, in the domain
	 *                    for FOLDRESCALE, which includes [0,N1], [0,N2], [0,N3]
	 *                    respectively, if pirange=0; or [-pi,pi] if pirange=1.
	 *         N1,N2,N3 - integer sizes of overall box (N2=N3=1 for 1D, N3=1 for 2D)
	 *         bin_size_x,y,z - what binning box size to use in each dimension
	 *                    (in rescaled coords where ranges are [0,Ni] ).
	 *                    For 1D, only bin_size_x is used; for 2D, it & bin_size_y.
	 * Output:
	 *         writes to ret a vector list of indices, each in the range 0,..,M-1.
	 *         Thus, ret must have been preallocated for M BIGINTs.
	 *
	 * Notes: I compared RAM usage against declaring an internal vector and passing
	 * back; the latter used more RAM and was slower.
	 * Avoided the bins array, as in JFM's spreader of 2016,
	 * tidied up, early 2017, Barnett.
	 *
	 * Timings (2017): 3s for M=1e8 NU pts on 1 core of i7; 5s on 1 core of xeon.
	 */
{
	bool isky = (N2 > 1), iskz = (N3 > 1);  // ky,kz avail? (cannot access if not)
	// here the +1 is needed to allow round-off error causing i1=N1/bin_size_x,
	// for kx near +pi, ie foldrescale gives N1 (exact arith would be 0 to N1-1).
	// Note that round-off near kx=-pi stably rounds negative to i1=0.
	BIGINT nbins1 = N1 / bin_size_x + 1, nbins2, nbins3;
	nbins2 = isky ? N2 / bin_size_y + 1 : 1;
	nbins3 = iskz ? N3 / bin_size_z + 1 : 1;
	BIGINT nbins = nbins1 * nbins2 * nbins3;

	std::vector<BIGINT> counts(nbins, 0);  // count how many pts in each bin
	for (BIGINT i = 0; i < M; i++) {
		// find the bin index in however many dims are needed
		BIGINT i1 = FOLDRESCALE(kx[i], N1, pirange) / bin_size_x, i2 = 0, i3 = 0;
		if (isky) i2 = FOLDRESCALE(ky[i], N2, pirange) / bin_size_y;
		if (iskz) i3 = FOLDRESCALE(kz[i], N3, pirange) / bin_size_z;
		BIGINT bin = i1 + nbins1 * (i2 + nbins2 * i3);
		counts[bin]++;
	}
	std::vector<BIGINT> offsets(nbins);   // cumulative sum of bin counts
	offsets[0] = 0;     // do: offsets = [0 cumsum(counts(1:end-1)]
	for (BIGINT i = 1; i < nbins; i++)
		offsets[i] = offsets[i - 1] + counts[i - 1];

	std::vector<BIGINT> inv(M);           // fill inverse map
	for (BIGINT i = 0; i < M; i++) {
		// find the bin index (again! but better than using RAM)
		BIGINT i1 = FOLDRESCALE(kx[i], N1, pirange) / bin_size_x, i2 = 0, i3 = 0;
		if (isky) i2 = FOLDRESCALE(ky[i], N2, pirange) / bin_size_y;
		if (iskz) i3 = FOLDRESCALE(kz[i], N3, pirange) / bin_size_z;
		BIGINT bin = i1 + nbins1 * (i2 + nbins2 * i3);
		BIGINT offset = offsets[bin];
		offsets[bin]++;
		inv[i] = offset;
	}
	// invert the map, writing to output pointer (writing pattern is random)
	for (BIGINT i = 0; i < M; i++)
		ret[inv[i]] = i;
}

void bin_sort_multithread(BIGINT* ret, BIGINT M, FLT* kx, FLT* ky, FLT* kz,
	BIGINT N1, BIGINT N2, BIGINT N3, int pirange,
	double bin_size_x, double bin_size_y, double bin_size_z, int debug,
	int nthr)
	/* Mostly-OpenMP'ed version of bin_sort.
	   For documentation see: bin_sort_singlethread.
	   Caution: when M (# NU pts) << N (# U pts), is SLOWER than single-thread.
	   Barnett 2/8/18
	   Explicit #threads control argument 7/20/20.
	   Todo: if debug, print timing breakdowns.
	 */
{
	bool isky = (N2 > 1), iskz = (N3 > 1);  // ky,kz avail? (cannot access if not)
	BIGINT nbins1 = N1 / bin_size_x + 1, nbins2, nbins3;  // see above note on why +1
	nbins2 = isky ? N2 / bin_size_y + 1 : 1;
	nbins3 = iskz ? N3 / bin_size_z + 1 : 1;
	BIGINT nbins = nbins1 * nbins2 * nbins3;
	if (nthr == 0)
		fprintf(stderr, "[%s] nthr (%d) must be positive!\n", __func__, nthr);
	int nt = min(M, (BIGINT)nthr);     // handle case of less points than threads
	std::vector<BIGINT> brk(nt + 1);    // list of start NU pt indices per thread

	// distribute the NU pts to threads once & for all...
	for (int t = 0; t <= nt; ++t)
		brk[t] = (BIGINT)(0.5 + M * t / (double)nt);   // start index for t'th chunk

	std::vector<BIGINT> counts(nbins, 0);     // global counts: # pts in each bin
	// offsets per thread, size nt * nbins, init to 0 by copying the counts vec...
	std::vector< std::vector<BIGINT> > ot(nt, counts);
	{    // scope for ct, the 2d array of counts in bins for each thread's NU pts
		std::vector< std::vector<BIGINT> > ct(nt, counts);   // nt * nbins, init to 0

#pragma omp parallel num_threads(nt)
		{  // parallel binning to each thread's count. Block done once per thread
			int t = MY_OMP_GET_THREAD_NUM();     // (we assume all nt threads created)
			//printf("\tt=%d: [%d,%d]\n",t,jlo[t],jhi[t]);
			for (BIGINT i = brk[t]; i < brk[t + 1]; i++) {
				// find the bin index in however many dims are needed
				BIGINT i1 = FOLDRESCALE(kx[i], N1, pirange) / bin_size_x, i2 = 0, i3 = 0;
				if (isky) i2 = FOLDRESCALE(ky[i], N2, pirange) / bin_size_y;
				if (iskz) i3 = FOLDRESCALE(kz[i], N3, pirange) / bin_size_z;
				BIGINT bin = i1 + nbins1 * (i2 + nbins2 * i3);
				ct[t][bin]++;               // no clash btw threads
			}
		}
		// sum along thread axis to get global counts
		for (BIGINT b = 0; b < nbins; ++b)   // (not worth omp. Either loop order is ok)
			for (int t = 0; t < nt; ++t)
				counts[b] += ct[t][b];

		std::vector<BIGINT> offsets(nbins);   // cumulative sum of bin counts
		// do: offsets = [0 cumsum(counts(1:end-1))] ...
		offsets[0] = 0;
		for (BIGINT i = 1; i < nbins; i++)
			offsets[i] = offsets[i - 1] + counts[i - 1];

		for (BIGINT b = 0; b < nbins; ++b)  // now build offsets for each thread & bin:
			ot[0][b] = offsets[b];                     // init
		for (int t = 1; t < nt; ++t)   // (again not worth omp. Either loop order is ok)
			for (BIGINT b = 0; b < nbins; ++b)
				ot[t][b] = ot[t - 1][b] + ct[t - 1][b];        // cumsum along t axis

	}  // scope frees up ct here, before inv alloc

	std::vector<BIGINT> inv(M);           // fill inverse map, in parallel
#pragma omp parallel num_threads(nt)
	{
		int t = MY_OMP_GET_THREAD_NUM();
		for (BIGINT i = brk[t]; i < brk[t + 1]; i++) {
			// find the bin index (again! but better than using RAM)
			BIGINT i1 = FOLDRESCALE(kx[i], N1, pirange) / bin_size_x, i2 = 0, i3 = 0;
			if (isky) i2 = FOLDRESCALE(ky[i], N2, pirange) / bin_size_y;
			if (iskz) i3 = FOLDRESCALE(kz[i], N3, pirange) / bin_size_z;
			BIGINT bin = i1 + nbins1 * (i2 + nbins2 * i3);
			inv[i] = ot[t][bin];   // get the offset for this NU pt and thread
			ot[t][bin]++;               // no clash
		}
	}
	// invert the map, writing to output pointer (writing pattern is random)
#pragma omp parallel for num_threads(nt) schedule(dynamic,10000)
	for (BIGINT i = 0; i < M; i++)
		ret[inv[i]] = i;
}
