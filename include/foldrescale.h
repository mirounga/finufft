#ifndef FOLDRESCALE_H
#define FOLDRESCALE_H

#include <cassert>

/* local NU coord fold+rescale macro: does the following affine transform to x:
	 when p=true:   map [-3pi,-pi) and [-pi,pi) and [pi,3pi)    each to [0,N)
	 otherwise,     map [-N,0) and [0,N) and [N,2N)             each to [0,N)
   Thus, only one period either side of the principal domain is folded.
   (It is *so* much faster than slow std::fmod that we stick to it.)
   This explains FINUFFT's allowed input domain of [-3pi,3pi).
   Speed comparisons of this macro vs a function are in devel/foldrescale*.
   The macro wins hands-down on i7, even for modern GCC9.
*/
#define FOLDRESCALE(x,N,p) (p ?                                         \
         (x + (x>=-PI ? (x<PI ? PI : -PI) : 3*PI)) * ((FLT)M_1_2PI*N) : \
                        (x>=0.0 ? (x<(FLT)N ? x : x-(FLT)N) : x+(FLT)N))

template<class T>
void foldrescale(BIGINT* sort_indices, T* kx, BIGINT* idx, T* x, BIGINT N, BIGINT begin, BIGINT end, const spread_opts& opts)
{
	const int ns = opts.nspread;          // abbrev. for w, kernel width
	const T ns2 = (T)ns / 2;          // half spread width, used as stencil shift

	for (BIGINT i = begin; i < end; i++)
	{
		BIGINT si = sort_indices[i];

		FLT xj = FOLDRESCALE(kx[si], N, opts.pirange);

		// coords (x,y,z), spread block corner index (i1,i2,i3) of current NU targ
		FLT c = std::ceil(xj - ns2); // leftmost grid index

		// shift of ker center, in [-w/2,-w/2+1]
		x[i] = c - xj;

		idx[i] = (BIGINT)c;
	}
}

#endif