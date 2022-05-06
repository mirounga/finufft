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
void foldrescale(BIGINT* sort_indices, T* kx, BIGINT* idx, T* x, BIGINT N, BIGINT size, const spread_opts& opts)
{
	const int ns = opts.nspread;          // abbrev. for w, kernel width
	const T ns2 = (T)ns / 2;          // half spread width, used as stencil shift

	for (BIGINT i = 0; i < size; i++)
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

#ifdef __AVX2__ // __AVX512VL__
template<>
inline void foldrescale<double>(BIGINT* sort_indices, double* kx, BIGINT* idx, double* x, BIGINT N, BIGINT size, const spread_opts& opts)
{
	const int ns = opts.nspread;          // abbrev. for w, kernel width
	const double ns2 = (double)ns / 2;          // half spread width, used as stencil shift

	if (opts.pirange) {
		__m512d _plus_pi = _mm512_set1_pd(M_PI);
		__m512d _minus_pi = _mm512_set1_pd(-M_PI);
		__m512d _three_pi = _mm512_set1_pd(3.0 * M_PI);
		__m512d _n_two_pi = _mm512_set1_pd(M_1_2PI * N);
		__m512d _ns2 = _mm512_set1_pd(ns2);

		BIGINT size8 = size & ~0x07;

		for (BIGINT i = 0; i < size8; i+=8)
		{ 
			__m512i _si0 = _mm512_loadu_epi64(sort_indices + i);

			__m512d _kx0 = _mm512_i64gather_pd(_si0, kx, 8);

			__mmask8 _cmpb0 = _mm512_cmp_pd_mask(_kx0, _plus_pi, _CMP_NGE_UQ);
			__mmask8 _cmpa0 = _mm512_cmp_pd_mask(_kx0, _minus_pi, _CMP_NLT_UQ);
			__m512d _tmpb0 = _mm512_mask_blend_pd(_cmpb0, _minus_pi, _plus_pi);
			__m512d _tmpa0 = _mm512_mask_blend_pd(_cmpa0, _three_pi, _tmpb0);
			__m512d _xj0 = _mm512_mul_pd(
				_mm512_add_pd(_kx0, _tmpa0),
				_n_two_pi);

			__m512d _c0 = _mm512_ceil_pd(_mm512_sub_pd(_xj0, _ns2));

			__m512d _x0 = _mm512_sub_pd(_c0, _xj0);

			__m512i _idx0 = _mm512_cvtpd_epi64(_c0);

			_mm512_store_pd(x + i, _x0);

			_mm512_store_epi64(idx + i, _idx0);
		}

		for (BIGINT i = size8; i < size; i++)
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
		for (BIGINT i = 0; i < size; i++)
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

template<>
inline void foldrescale<float>(BIGINT* sort_indices, float* kx, BIGINT* idx, float* x, BIGINT N, BIGINT size, const spread_opts& opts)
{
	const int ns = opts.nspread;          // abbrev. for w, kernel width
	const float ns2 = (float)ns / 2;          // half spread width, used as stencil shift

	if (opts.pirange  == 0) {
		__m256 _zero = _mm256_setzero_ps();
		__m256 _nn = _mm256_set1_ps(static_cast<float>(N));
		__m256 _ns2 = _mm256_set1_ps(ns2);

		BIGINT size8 = size & ~0x07;

		for (BIGINT i = 0; i < size8; i += 8)
		{
			__m512i _si0 = _mm512_loadu_epi64(sort_indices + i);

			__m256 _kx0 = _mm512_i64gather_ps(_si0, kx, 4);

			__m256 _kx0_minus_n = _mm256_sub_ps(_kx0, _nn);
			__m256 _kx0_plus_n = _mm256_add_ps(_kx0, _nn);

			__mmask8 _cmpb0 = _mm256_cmp_ps_mask(_kx0, _nn, _CMP_NGE_UQ);
			__mmask8 _cmpa0 = _mm256_cmp_ps_mask(_kx0, _zero, _CMP_NLT_UQ);
			__m256 _tmpb0 = _mm256_mask_blend_ps(_cmpb0, _kx0_minus_n, _kx0);
			__m256 _xj0 = _mm256_mask_blend_ps(_cmpa0, _tmpb0, _kx0_plus_n);

			__m256 _c0 = _mm256_ceil_ps(_mm256_sub_ps(_xj0, _ns2));

			__m256 _x0 = _mm256_sub_ps(_c0, _xj0);

			__m512i _idx0 = _mm512_cvtps_epi64(_c0);

			_mm256_store_ps(x + i, _x0);

			_mm512_store_epi64(idx + i, _idx0);
		}

		for (BIGINT i = size8; i < size; i++)
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
	else {
		__m256 _plus_pi = _mm256_set1_ps(M_PI);
		__m256 _minus_pi = _mm256_set1_ps(-M_PI);
		__m256 _three_pi = _mm256_set1_ps(3.0 * M_PI);
		__m256 _n_two_pi = _mm256_set1_ps(M_1_2PI * N);
		__m256 _ns2 = _mm256_set1_ps(ns2);

		BIGINT size8 = size & ~0x07;

		for (BIGINT i = 0; i < size8; i+=8)
		{ 
			__m512i _si0 = _mm512_loadu_epi64(sort_indices + i);

			__m256 _kx0 = _mm512_i64gather_ps(_si0, kx, 4);

			__mmask8 _cmpb0 = _mm256_cmp_ps_mask(_kx0, _plus_pi, _CMP_NGE_UQ);
			__mmask8 _cmpa0 = _mm256_cmp_ps_mask(_kx0, _minus_pi, _CMP_NLT_UQ);
			__m256 _tmpb0 = _mm256_mask_blend_ps(_cmpb0, _minus_pi, _plus_pi);
			__m256 _tmpa0 = _mm256_mask_blend_ps(_cmpa0, _three_pi, _tmpb0);
			__m256 _xj0 = _mm256_mul_ps(
				_mm256_add_ps(_kx0, _tmpa0),
				_n_two_pi);

			__m256 _c0 = _mm256_ceil_ps(_mm256_sub_ps(_xj0, _ns2));

			__m256 _x0 = _mm256_sub_ps(_c0, _xj0);

			__m512i _idx0 = _mm512_cvtps_epi64(_c0);

			_mm256_store_ps(x + i, _x0);

			_mm512_store_epi64(idx + i, _idx0);
		}

		for (BIGINT i = size8; i < size; i++)
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
}
#endif
#endif