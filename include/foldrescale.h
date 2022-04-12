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
template<class T>
T FOLDRESCALE(T x, BIGINT N, int p) {
	return (p ?
		(x + (x >= -PI ? (x < PI ? PI : -PI) : 3 * PI)) * ((T)M_1_2PI * N) :
		(x >= 0.0 ? (x < static_cast<T>(N) ? x : x - (T)N) : x + (T)N));
}

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

#ifdef ___AVX2__
template<>
void foldrescale<double>(BIGINT* sort_indices, double* kx, BIGINT* idx, double* x, BIGINT N, BIGINT begin, BIGINT end, const spread_opts& opts)
{
	const int ns = opts.nspread;          // abbrev. for w, kernel width
	const double ns2 = (double)ns / 2;          // half spread width, used as stencil shift

	if (opts.pirange) {
		__m256d _plus_pi = _mm256_set1_pd(M_PI);
		__m256d _minus_pi = _mm256_set1_pd(-M_PI);
		__m256d _three_pi = _mm256_set1_pd(3.0 * M_PI);
		__m256d _n_two_pi = _mm256_set1_pd(M_1_2PI * N);
		__m256d _ns2 = _mm256_set1_pd(ns2);

		BIGINT end4 = begin + ((end - begin) & ~0x04);

		for (BIGINT i = begin; i < end4; i+=4)
		{ 
			__m256i _si0 = _mm256_loadu_epi64(sort_indices + i);

			__m256d _kx0 = _mm256_i64gather_pd(kx, _si0, 8);

			__m256d _cmpb0 = _mm256_cmp_pd(_kx0, _plus_pi, _CMP_NGE_UQ);
			__m256d _cmpa0 = _mm256_cmp_pd(_kx0, _minus_pi, _CMP_NLT_UQ);
			__m256d _tmpb0 = _mm256_blendv_pd(_minus_pi, _plus_pi, _cmpb0);
			__m256d _tmpa0 = _mm256_blendv_pd(_three_pi, _tmpb0, _cmpa0);
			__m256d _xj0 = _mm256_mul_pd(
				_mm256_add_pd(_kx0, _tmpa0),
				_n_two_pi);

			__m256d _c0 = _mm256_ceil_pd(_mm256_sub_pd(_xj0, _ns2));

			__m256d _x0 = _mm256_sub_pd(_c0, _xj0);

			__m256i _idx0 = _mm256_cvtpd_epi64(_c0);

			_mm256_storeu_pd(x + i, _x0);

			_mm256_storeu_epi64(idx + i, _idx0);
		}

		for (BIGINT i = end4; i < end; i++)
		{ 
			BIGINT si = sort_indices[i];

			double xj = FOLDRESCALE(kx[si], N, -1);

			// coords (x,y,z), spread block corner index (i1,i2,i3) of current NU targ
			double c = std::ceil(xj - ns2); // leftmost grid index

			// shift of ker center, in [-w/2,-w/2+1]
			x[i] = c - xj;

			idx[i] = (BIGINT)c;
		}
	}
	else {
		for (BIGINT i = begin; i < end; i++)
		{
			BIGINT si = sort_indices[i];

			FLT xj = FOLDRESCALE(kx[si], N, 0);

			// coords (x,y,z), spread block corner index (i1,i2,i3) of current NU targ
			FLT c = std::ceil(xj - ns2); // leftmost grid index

			// shift of ker center, in [-w/2,-w/2+1]
			x[i] = c - xj;

			idx[i] = (BIGINT)c;
		}
	}
}
#endif
#endif