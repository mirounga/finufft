#ifndef SPREAD_H
#define SPREAD_H

template<class T>
void spread_subproblem_1d(BIGINT* sort_indices, BIGINT off1, BIGINT size1, T* du, T* dd,
	BIGINT* i1, 
	T* kernel_vals1, 
	BIGINT begin, BIGINT end, const spread_opts& opts)
	/* 1D spreader from nonuniform to uniform subproblem grid, without wrapping.
	   Inputs:
	   off1 - integer offset of left end of du subgrid from that of overall fine
			  periodized output grid {0,1,..N-1}.
	   size1 - integer length of output subgrid du
	   M - number of NU pts in subproblem
	   kx (length M) - are rescaled NU source locations, should lie in
					   [off1+ns/2,off1+size1-1-ns/2] so as kernels stay in bounds
	   dd (length M complex, interleaved) - source strengths
	   Outputs:
	   du (length size1 complex, interleaved) - preallocated uniform subgrid array

	   The reason periodic wrapping is avoided in subproblems is speed: avoids
	   conditionals, indirection (pointers), and integer mod. Originally 2017.
	   Kernel eval mods by Ludvig al Klinteberg.
	   Fixed so rounding to integer grid consistent w/ get_subgrid, prevents
	   chance of segfault when epsmach*N1>O(1), assuming max() and ceil() commute.
	   This needed off1 as extra arg. AHB 11/30/20.
	*/
{
	int ns = opts.nspread;          // a.k.a. w
	T ns2 = (T)ns / 2;          // half spread width
	int nsPadded = 4 * (1 + (ns - 1) / 4); // pad ns to mult of 4
	for (BIGINT i = 0; i < 2 * size1; ++i)         // zero output
		du[i] = 0.0;


	T* pKer1 = kernel_vals1 + begin * nsPadded;

	for (BIGINT i = begin; i < end; i++) {           // loop over NU pts
		BIGINT si = sort_indices[i];
		T re0 = dd[2 * si];
		T im0 = dd[2 * si + 1];

		BIGINT j = i1[i] - off1;    // offset rel to subgrid, starts the output indices
		// critical inner loop:
		for (int dx = 0; dx < ns; ++dx) {
			T k = pKer1[dx];
			du[2 * j] += re0 * k;
			du[2 * j + 1] += im0 * k;
			++j;
		}

		pKer1 += nsPadded;
	}
}

#ifdef __AVX2__
#include <immintrin.h>

template<>
inline void spread_subproblem_1d<double>(BIGINT* sort_indices, BIGINT off1, BIGINT size1, double* du, double* dd,
	BIGINT* i1,
	double* kernel_vals1,
	BIGINT begin, BIGINT end, const spread_opts& opts)
{
	int ns = opts.nspread;          // a.k.a. w
	double ns2 = (double)ns / 2;          // half spread width
	int nsPadded = 4 * (1 + (ns - 1) / 4); // pad ns to mult of 4
	for (BIGINT i = 0; i < 2 * size1; ++i)         // zero output
		du[i] = 0.0;

	double* pKer1 = kernel_vals1 + begin * nsPadded;

	switch (nsPadded) {
	case 8:
		for (BIGINT i = begin; i < end; i++) {           // loop over NU pts
			BIGINT si = sort_indices[i];
			__m256d _dd0 = _mm256_permute4x64_pd(
				_mm256_castpd128_pd256(_mm_load_pd(dd + 2 * si)),
				0x44);

			// offset rel to subgrid, starts the output indices
			double* pDu = du + 2 * (i1[i] - off1);

			__m256d _k0 = _mm256_load_pd(pKer1 + 0);
			__m256d _k2 = _mm256_load_pd(pKer1 + 4);

			__m256d _kk0 = _mm256_permute4x64_pd(_k0, 0x50);
			__m256d _kk1 = _mm256_permute4x64_pd(_k0, 0xfa);
			__m256d _kk2 = _mm256_permute4x64_pd(_k2, 0x50);
			__m256d _kk3 = _mm256_permute4x64_pd(_k2, 0xfa);

			__m256d _du0 = _mm256_loadu_pd(pDu + 0);
			__m256d _du1 = _mm256_loadu_pd(pDu + 4);
			__m256d _du2 = _mm256_loadu_pd(pDu + 8);
			__m256d _du3 = _mm256_loadu_pd(pDu + 12);

			_du0 = _mm256_fmadd_pd(_dd0, _kk0, _du0);
			_du1 = _mm256_fmadd_pd(_dd0, _kk1, _du1);
			_du2 = _mm256_fmadd_pd(_dd0, _kk2, _du2);
			_du3 = _mm256_fmadd_pd(_dd0, _kk3, _du3);

			_mm256_storeu_pd(pDu + 0, _du0);
			_mm256_storeu_pd(pDu + 4, _du1);
			_mm256_storeu_pd(pDu + 8, _du2);
			_mm256_storeu_pd(pDu + 12, _du3);

			pKer1 += nsPadded;
		}
		break;
	case 12:
		for (BIGINT i = begin; i < end; i++) {           // loop over NU pts
			BIGINT si = sort_indices[i];
			__m256d _dd0 = _mm256_permute4x64_pd(
				_mm256_castpd128_pd256(_mm_load_pd(dd + 2 * si)),
				0x44);

			// offset rel to subgrid, starts the output indices
			double* pDu = du + 2 * (i1[i] - off1);

			__m256d _k0 = _mm256_load_pd(pKer1 + 0);
			__m256d _k2 = _mm256_load_pd(pKer1 + 4);
			__m256d _k4 = _mm256_load_pd(pKer1 + 8);

			__m256d _kk0 = _mm256_permute4x64_pd(_k0, 0x50);
			__m256d _kk1 = _mm256_permute4x64_pd(_k0, 0xfa);
			__m256d _kk2 = _mm256_permute4x64_pd(_k2, 0x50);
			__m256d _kk3 = _mm256_permute4x64_pd(_k2, 0xfa);
			__m256d _kk4 = _mm256_permute4x64_pd(_k4, 0x50);
			__m256d _kk5 = _mm256_permute4x64_pd(_k4, 0xfa);

			__m256d _du0 = _mm256_loadu_pd(pDu + 0);
			__m256d _du1 = _mm256_loadu_pd(pDu + 4);
			__m256d _du2 = _mm256_loadu_pd(pDu + 8);
			__m256d _du3 = _mm256_loadu_pd(pDu + 12);
			__m256d _du4 = _mm256_loadu_pd(pDu + 16);
			__m256d _du5 = _mm256_loadu_pd(pDu + 20);

			_du0 = _mm256_fmadd_pd(_dd0, _kk0, _du0);
			_du1 = _mm256_fmadd_pd(_dd0, _kk1, _du1);
			_du2 = _mm256_fmadd_pd(_dd0, _kk2, _du2);
			_du3 = _mm256_fmadd_pd(_dd0, _kk3, _du3);
			_du4 = _mm256_fmadd_pd(_dd0, _kk4, _du4);
			_du5 = _mm256_fmadd_pd(_dd0, _kk5, _du5);

			_mm256_storeu_pd(pDu + 0, _du0);
			_mm256_storeu_pd(pDu + 4, _du1);
			_mm256_storeu_pd(pDu + 8, _du2);
			_mm256_storeu_pd(pDu + 12, _du3);
			_mm256_storeu_pd(pDu + 16, _du4);
			_mm256_storeu_pd(pDu + 20, _du5);

			pKer1 += nsPadded;
		}
		break;
	case 16:
		for (BIGINT i = begin; i < end; i++) {           // loop over NU pts
			BIGINT si = sort_indices[i];
			__m256d _dd0 = _mm256_permute4x64_pd(
				_mm256_castpd128_pd256(_mm_load_pd(dd + 2 * si)),
				0x44);

			// offset rel to subgrid, starts the output indices
			double* pDu = du + 2 * (i1[i] - off1);

			__m256d _k0 = _mm256_load_pd(pKer1 + 0);
			__m256d _k2 = _mm256_load_pd(pKer1 + 4);

			__m256d _kk0 = _mm256_permute4x64_pd(_k0, 0x50);
			__m256d _kk1 = _mm256_permute4x64_pd(_k0, 0xfa);
			__m256d _kk2 = _mm256_permute4x64_pd(_k2, 0x50);
			__m256d _kk3 = _mm256_permute4x64_pd(_k2, 0xfa);

			__m256d _du0 = _mm256_loadu_pd(pDu + 0);
			__m256d _du1 = _mm256_loadu_pd(pDu + 4);
			__m256d _du2 = _mm256_loadu_pd(pDu + 8);
			__m256d _du3 = _mm256_loadu_pd(pDu + 12);

			_du0 = _mm256_fmadd_pd(_dd0, _kk0, _du0);
			_du1 = _mm256_fmadd_pd(_dd0, _kk1, _du1);
			_du2 = _mm256_fmadd_pd(_dd0, _kk2, _du2);
			_du3 = _mm256_fmadd_pd(_dd0, _kk3, _du3);

			_mm256_storeu_pd(pDu + 0, _du0);
			_mm256_storeu_pd(pDu + 4, _du1);
			_mm256_storeu_pd(pDu + 8, _du2);
			_mm256_storeu_pd(pDu + 12, _du3);

			_k0 = _mm256_load_pd(pKer1 + 8);
			_k2 = _mm256_load_pd(pKer1 + 12);

			_kk0 = _mm256_permute4x64_pd(_k0, 0x50);
			_kk1 = _mm256_permute4x64_pd(_k0, 0xfa);
			_kk2 = _mm256_permute4x64_pd(_k2, 0x50);
			_kk3 = _mm256_permute4x64_pd(_k2, 0xfa);

			_du0 = _mm256_loadu_pd(pDu + 16);
			_du1 = _mm256_loadu_pd(pDu + 20);
			_du2 = _mm256_loadu_pd(pDu + 24);
			_du3 = _mm256_loadu_pd(pDu + 28);

			_du0 = _mm256_fmadd_pd(_dd0, _kk0, _du0);
			_du1 = _mm256_fmadd_pd(_dd0, _kk1, _du1);
			_du2 = _mm256_fmadd_pd(_dd0, _kk2, _du2);
			_du3 = _mm256_fmadd_pd(_dd0, _kk3, _du3);

			_mm256_storeu_pd(pDu + 16, _du0);
			_mm256_storeu_pd(pDu + 20, _du1);
			_mm256_storeu_pd(pDu + 24, _du2);
			_mm256_storeu_pd(pDu + 28, _du3);

			pKer1 += nsPadded;
		}
		break;
	default:
		for (BIGINT i = begin; i < end; i++) {           // loop over NU pts
			BIGINT si = sort_indices[i];
			__m128d _dd0 = _mm_load_pd(dd + 2 * si);

			// offset rel to subgrid, starts the output indices
			double* pDu = du + 2 * (i1[i] - off1);

			// critical inner loop:
			for (int dx = 0; dx < ns; ++dx) {
				__m128d _kk0 = _mm_loaddup_pd(pKer1 + dx);
				__m128d _du0 = _mm_load_pd(pDu);
				_du0 = _mm_fmadd_pd(_dd0, _kk0, _du0);
				_mm_store_pd(pDu, _du0);

				pDu += 2;
			}

			pKer1 += nsPadded;
		}
		break;
	}
}

template<>
inline void spread_subproblem_1d<float>(BIGINT* sort_indices, BIGINT off1, BIGINT size1, float* du, float* dd,
	BIGINT* i1,
	float* kernel_vals1,
	BIGINT begin, BIGINT end, const spread_opts& opts)
{
	int ns = opts.nspread;          // a.k.a. w
	float ns2 = (float)ns / 2;          // half spread width
	int nsPadded = 4 * (1 + (ns - 1) / 4); // pad ns to mult of 4
	for (BIGINT i = 0; i < 2 * size1; ++i)         // zero output
		du[i] = 0.0;

	float* pKer1 = kernel_vals1 + begin * nsPadded;

	__m256i _mask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, -1, -1);
	__m256i _broadcast2 = _mm256_set_epi32(1, 0, 1, 0, 1, 0, 1, 0);
	__m256i _spreadlo = _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0);
	__m256i _spreadhi = _mm256_set_epi32(7, 7, 6, 6, 5, 5, 4, 4);

	switch (nsPadded) {
	case 8:
		for (BIGINT i = begin; i < end; i++) {           // loop over NU pts
			BIGINT si = sort_indices[i];
			__m256 _d0 = _mm256_maskload_ps(dd + 2 * si, _mask);
			__m256 _dd0 = _mm256_permutevar8x32_ps(_d0, _broadcast2);

			// offset rel to subgrid, starts the output indices
			float* pDu = du + 2 * (i1[i] - off1);

			__m256 _k0 = _mm256_load_ps(pKer1 + 0);

			__m256 _kk0 = _mm256_permutevar8x32_ps(_k0, _spreadlo);
			__m256 _kk1 = _mm256_permutevar8x32_ps(_k0, _spreadhi);

			__m256 _du0 = _mm256_loadu_ps(pDu + 0);
			__m256 _du1 = _mm256_loadu_ps(pDu + 8);

			_du0 = _mm256_fmadd_ps(_dd0, _kk0, _du0);
			_du1 = _mm256_fmadd_ps(_dd0, _kk1, _du1);

			_mm256_storeu_ps(pDu + 0, _du0);
			_mm256_storeu_ps(pDu + 8, _du1);

			pKer1 += nsPadded;
		}
		break;
	default:
		for (BIGINT i = begin; i < end; i++) {           // loop over NU pts
			BIGINT si = sort_indices[i];
			__m256 _dd0 = _mm256_maskload_ps(dd + 2 * si, _mask);

			// offset rel to subgrid, starts the output indices
			float* pDu = du + 2 * (i1[i] - off1);

			// critical inner loop:
			for (int dx = 0; dx < ns; ++dx) {
				__m256 _kk0 = _mm256_set1_ps(pKer1[dx]);
				__m256 _du0 = _mm256_maskload_ps(pDu, _mask);
				_du0 = _mm256_fmadd_ps(_dd0, _kk0, _du0);
				_mm256_maskstore_ps(pDu, _mask, _du0);

				pDu += 2;
			}

			pKer1 += nsPadded;
		}
		break;
	}
}
#endif

template<class T>
void spread_subproblem_2d(BIGINT* sort_indices,
	BIGINT off1, BIGINT off2,
	BIGINT size1, BIGINT size2,
	T* du, T* dd,
	BIGINT* i1, BIGINT* i2,
	T* kernel_vals1, T* kernel_vals2,
	BIGINT begin, BIGINT end, const spread_opts& opts)
	/* spreader from dd (NU) to du (uniform) in 2D without wrapping.
	   See above docs/notes for spread_subproblem_2d.
	   kx,ky (size M) are NU locations in [off+ns/2,off+size-1-ns/2] in both dims.
	   dd (size M complex) are complex source strengths
	   du (size size1*size2) is complex uniform output array
	 */
{
	int ns = opts.nspread;
	T ns2 = (T)ns / 2;          // half spread width
	int nsPadded = 4 * (1 + (ns - 1) / 4); // pad ns to mult of 4
	for (BIGINT i = 0; i < 2 * size1 * size2; ++i)
		du[i] = 0.0;

	T* pKer1 = kernel_vals1 + begin * nsPadded;
	T* pKer2 = kernel_vals2 + begin * nsPadded;

	for (BIGINT i = begin; i < end; i++) {           // loop over NU pts
		BIGINT si = sort_indices[i];
		T re0 = dd[2 * si];
		T im0 = dd[2 * si + 1];

		// Combine kernel with complex source value to simplify inner loop
		T ker1val[2 * MAX_NSPREAD];    // here 2* is because of complex
		for (int j = 0; j < ns; j++) {
			ker1val[2 * j] = re0 * pKer1[j];
			ker1val[2 * j + 1] = im0 * pKer1[j];
		}
		// critical inner loop:
		for (int dy = 0; dy < ns; ++dy) {
			BIGINT j = size1 * (i2[i] - off2 + dy) + i1[i] - off1;   // should be in subgrid
			T kerval = pKer2[dy];
			T* trg = du + 2 * j;
			for (int dx = 0; dx < 2 * ns; ++dx) {
				trg[dx] += kerval * ker1val[dx];
			}
		}

		pKer1 += nsPadded;
		pKer2 += nsPadded;
	}
}

#ifdef __AVX2__
template<>
inline void spread_subproblem_2d<double>(BIGINT* sort_indices,
	BIGINT off1, BIGINT off2,
	BIGINT size1, BIGINT size2,
	double* du, double* dd,
	BIGINT* i1, BIGINT* i2,
	double* kernel_vals1, double* kernel_vals2,
	BIGINT begin, BIGINT end, const spread_opts& opts)
{
	int ns = opts.nspread;
	double ns2 = (double)ns / 2;          // half spread width
	int nsPadded = 4 * (1 + (ns - 1) / 4); // pad ns to mult of 4
	for (BIGINT i = 0; i < 2 * size1 * size2; ++i)
		du[i] = 0.0;

	double* pKer1 = kernel_vals1 + begin * nsPadded;
	double* pKer2 = kernel_vals2 + begin * nsPadded;

	switch (nsPadded) {
	case 8:
		for (BIGINT i = begin; i < end; i++) {           // loop over NU pts
			BIGINT si = sort_indices[i];
			__m256d _dd0 = _mm256_permute4x64_pd(
				_mm256_castpd128_pd256(_mm_load_pd(dd + 2 * si)),
				0x44);

			// Combine kernel with complex source value to simplify inner loop
			__m256d _k0 = _mm256_load_pd(pKer1 + 0);
			__m256d _k2 = _mm256_load_pd(pKer1 + 4);

			__m256d _kk0 = _mm256_mul_pd(_dd0, _mm256_permute4x64_pd(_k0, 0x50));
			__m256d _kk1 = _mm256_mul_pd(_dd0, _mm256_permute4x64_pd(_k0, 0xfa));
			__m256d _kk2 = _mm256_mul_pd(_dd0, _mm256_permute4x64_pd(_k2, 0x50));
			__m256d _kk3 = _mm256_mul_pd(_dd0, _mm256_permute4x64_pd(_k2, 0xfa));

			// critical inner loop:
			for (int dy = 0; dy < ns; ++dy) {
				BIGINT j = size1 * (i2[i] - off2 + dy) + i1[i] - off1;   // should be in subgrid
				double* pDu = du + 2 * j;

				__m256d _kerval = _mm256_set1_pd(pKer2[dy]);

				__m256d _du0 = _mm256_loadu_pd(pDu + 0);
				__m256d _du1 = _mm256_loadu_pd(pDu + 4);
				__m256d _du2 = _mm256_loadu_pd(pDu + 8);
				__m256d _du3 = _mm256_loadu_pd(pDu + 12);

				_du0 = _mm256_fmadd_pd(_kerval, _kk0, _du0);
				_du1 = _mm256_fmadd_pd(_kerval, _kk1, _du1);
				_du2 = _mm256_fmadd_pd(_kerval, _kk2, _du2);
				_du3 = _mm256_fmadd_pd(_kerval, _kk3, _du3);

				_mm256_storeu_pd(pDu + 0, _du0);
				_mm256_storeu_pd(pDu + 4, _du1);
				_mm256_storeu_pd(pDu + 8, _du2);
				_mm256_storeu_pd(pDu + 12, _du3);
			}

			pKer1 += nsPadded;
			pKer2 += nsPadded;
		}
		break;
	case 12:
		for (BIGINT i = begin; i < end; i++) {           // loop over NU pts
			BIGINT si = sort_indices[i];
			__m256d _dd0 = _mm256_permute4x64_pd(
				_mm256_castpd128_pd256(_mm_load_pd(dd + 2 * si)),
				0x44);

			// Combine kernel with complex source value to simplify inner loop
			__m256d _k0 = _mm256_load_pd(pKer1 + 0);
			__m256d _k2 = _mm256_load_pd(pKer1 + 4);
			__m256d _k4 = _mm256_load_pd(pKer1 + 8);

			__m256d _kk0 = _mm256_mul_pd(_dd0, _mm256_permute4x64_pd(_k0, 0x50));
			__m256d _kk1 = _mm256_mul_pd(_dd0, _mm256_permute4x64_pd(_k0, 0xfa));
			__m256d _kk2 = _mm256_mul_pd(_dd0, _mm256_permute4x64_pd(_k2, 0x50));
			__m256d _kk3 = _mm256_mul_pd(_dd0, _mm256_permute4x64_pd(_k2, 0xfa));
			__m256d _kk4 = _mm256_mul_pd(_dd0, _mm256_permute4x64_pd(_k4, 0x50));
			__m256d _kk5 = _mm256_mul_pd(_dd0, _mm256_permute4x64_pd(_k4, 0xfa));

			// critical inner loop:
			for (int dy = 0; dy < ns; ++dy) {
				BIGINT j = size1 * (i2[i] - off2 + dy) + i1[i] - off1;   // should be in subgrid
				double* pDu = du + 2 * j;

				__m256d _kerval = _mm256_set1_pd(pKer2[dy]);

				__m256d _du0 = _mm256_loadu_pd(pDu + 0);
				__m256d _du1 = _mm256_loadu_pd(pDu + 4);
				__m256d _du2 = _mm256_loadu_pd(pDu + 8);
				__m256d _du3 = _mm256_loadu_pd(pDu + 12);
				__m256d _du4 = _mm256_loadu_pd(pDu + 16);
				__m256d _du5 = _mm256_loadu_pd(pDu + 20);

				_du0 = _mm256_fmadd_pd(_kerval, _kk0, _du0);
				_du1 = _mm256_fmadd_pd(_kerval, _kk1, _du1);
				_du2 = _mm256_fmadd_pd(_kerval, _kk2, _du2);
				_du3 = _mm256_fmadd_pd(_kerval, _kk3, _du3);
				_du4 = _mm256_fmadd_pd(_kerval, _kk4, _du4);
				_du5 = _mm256_fmadd_pd(_kerval, _kk5, _du5);

				_mm256_storeu_pd(pDu + 0, _du0);
				_mm256_storeu_pd(pDu + 4, _du1);
				_mm256_storeu_pd(pDu + 8, _du2);
				_mm256_storeu_pd(pDu + 12, _du3);
				_mm256_storeu_pd(pDu + 16, _du4);
				_mm256_storeu_pd(pDu + 20, _du5);
			}

			pKer1 += nsPadded;
			pKer2 += nsPadded;
		}
		break;
	default:
		for (BIGINT i = begin; i < end; i++) {           // loop over NU pts
			BIGINT si = sort_indices[i];
			double re0 = dd[2 * si];
			double im0 = dd[2 * si + 1];

			// Combine kernel with complex source value to simplify inner loop
			double ker1val[2 * MAX_NSPREAD];    // here 2* is because of complex
			for (int j = 0; j < ns; j++) {
				ker1val[2 * j] = re0 * pKer1[j];
				ker1val[2 * j + 1] = im0 * pKer1[j];
			}
			// critical inner loop:
			for (int dy = 0; dy < ns; ++dy) {
				BIGINT j = size1 * (i2[i] - off2 + dy) + i1[i] - off1;   // should be in subgrid
				double kerval = pKer2[dy];
				double* trg = du + 2 * j;
				for (int dx = 0; dx < 2 * ns; ++dx) {
					trg[dx] += kerval * ker1val[dx];
				}
			}

			pKer1 += nsPadded;
			pKer2 += nsPadded;
		}
		break;
	}
}

template<>
inline void spread_subproblem_2d<float>(BIGINT* sort_indices,
	BIGINT off1, BIGINT off2,
	BIGINT size1, BIGINT size2,
	float* du, float* dd,
	BIGINT* i1, BIGINT* i2,
	float* kernel_vals1, float* kernel_vals2,
	BIGINT begin, BIGINT end, const spread_opts& opts)
{
	int ns = opts.nspread;
	float ns2 = (float)ns / 2;          // half spread width
	int nsPadded = 4 * (1 + (ns - 1) / 4); // pad ns to mult of 4
	for (BIGINT i = 0; i < 2 * size1 * size2; ++i)
		du[i] = 0.0;

	float* pKer1 = kernel_vals1 + begin * nsPadded;
	float* pKer2 = kernel_vals2 + begin * nsPadded;

	__m256i _mask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, -1, -1);
	__m256i _broadcast2 = _mm256_set_epi32(1, 0, 1, 0, 1, 0, 1, 0);
	__m256i _spreadlo = _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0);
	__m256i _spreadhi = _mm256_set_epi32(7, 7, 6, 6, 5, 5, 4, 4);

	switch (nsPadded) {
	case 8:
		for (BIGINT i = begin; i < end; i++) {           // loop over NU pts
			// Combine kernel with complex source value
			BIGINT si = sort_indices[i];
			__m256 _d0 = _mm256_maskload_ps(dd + 2 * si, _mask);
			__m256 _dd0 = _mm256_permutevar8x32_ps(_d0, _broadcast2);

			__m256 _k0 = _mm256_loadu_ps(pKer1 + 0);

			__m256 _kk0 = _mm256_mul_ps(_dd0, _mm256_permutevar8x32_ps(_k0, _spreadlo));
			__m256 _kk1 = _mm256_mul_ps(_dd0, _mm256_permutevar8x32_ps(_k0, _spreadhi));

			// critical inner loop:
			for (int dy = 0; dy < ns; ++dy) {
				BIGINT j = size1 * (i2[i] - off2 + dy) + i1[i] - off1;   // should be in subgrid
				float* pDu = du + 2 * j;

				__m256 _kerval = _mm256_set1_ps(pKer2[dy]);

				__m256 _du0 = _mm256_loadu_ps(pDu + 0);
				__m256 _du1 = _mm256_loadu_ps(pDu + 8);

				_du0 = _mm256_fmadd_ps(_kerval, _kk0, _du0);
				_du1 = _mm256_fmadd_ps(_kerval, _kk1, _du1);

				_mm256_storeu_ps(pDu + 0, _du0);
				_mm256_storeu_ps(pDu + 8, _du1);
			}

			pKer1 += nsPadded;
			pKer2 += nsPadded;
		}
		break;
	case 12:
		for (BIGINT i = begin; i < end; i++) {           // loop over NU pts
			// Combine kernel with complex source value
			BIGINT si = sort_indices[i];
			__m256 _d0 = _mm256_maskload_ps(dd + 2 * si, _mask);
			__m256 _dd0 = _mm256_permutevar8x32_ps(_d0, _broadcast2);

			__m256 _k0 = _mm256_loadu_ps(pKer1 + 0);
			__m256 _k2 = _mm256_castps128_ps256(_mm_load_ps(pKer1 + 8));

			__m256 _kk0 = _mm256_mul_ps(_dd0, _mm256_permutevar8x32_ps(_k0, _spreadlo));
			__m256 _kk1 = _mm256_mul_ps(_dd0, _mm256_permutevar8x32_ps(_k0, _spreadhi));
			__m256 _kk2 = _mm256_mul_ps(_dd0, _mm256_permutevar8x32_ps(_k2, _spreadlo));

			// critical inner loop:
			for (int dy = 0; dy < ns; ++dy) {
				BIGINT j = size1 * (i2[i] - off2 + dy) + i1[i] - off1;   // should be in subgrid
				float* pDu = du + 2 * j;

				__m256 _kerval = _mm256_set1_ps(pKer2[dy]);

				__m256 _du0 = _mm256_loadu_ps(pDu + 0);
				__m256 _du1 = _mm256_loadu_ps(pDu + 8);
				__m256 _du2 = _mm256_loadu_ps(pDu + 16);

				_du0 = _mm256_fmadd_ps(_kerval, _kk0, _du0);
				_du1 = _mm256_fmadd_ps(_kerval, _kk1, _du1);
				_du2 = _mm256_fmadd_ps(_kerval, _kk2, _du2);

				_mm256_storeu_ps(pDu + 0, _du0);
				_mm256_storeu_ps(pDu + 8, _du1);
				_mm256_storeu_ps(pDu + 16, _du2);
			}

			pKer1 += nsPadded;
			pKer2 += nsPadded;
		}
		break;
	default:
		for (BIGINT i = begin; i < end; i++) {           // loop over NU pts
			BIGINT si = sort_indices[i];
			float re0 = dd[2 * si];
			float im0 = dd[2 * si + 1];

			// Combine kernel with complex source value to simplify inner loop
			float ker1val[2 * MAX_NSPREAD];    // here 2* is because of complex
			for (int j = 0; j < ns; j++) {
				ker1val[2 * j] = re0 * pKer1[j];
				ker1val[2 * j + 1] = im0 * pKer1[j];
			}
			// critical inner loop:
			for (int dy = 0; dy < ns; ++dy) {
				BIGINT j = size1 * (i2[i] - off2 + dy) + i1[i] - off1;   // should be in subgrid
				float kerval = pKer2[dy];
				float* trg = du + 2 * j;
				for (int dx = 0; dx < 2 * ns; ++dx) {
					trg[dx] += kerval * ker1val[dx];
				}
			}

			pKer1 += nsPadded;
			pKer2 += nsPadded;
		}
		break;
	}
}
#endif

template<class T>
void spread_subproblem_3d(BIGINT* sort_indices, 
	BIGINT off1, BIGINT off2, BIGINT off3, 
	BIGINT size1, BIGINT size2, BIGINT size3,
	T* du, T* dd,
	BIGINT* i1, BIGINT* i2, BIGINT* i3,
	T* kernel_vals1, T* kernel_vals2, T* kernel_vals3,
	BIGINT begin, BIGINT end, const spread_opts& opts)
	/* spreader from dd (NU) to du (uniform) in 3D without wrapping.
	   See above docs/notes for spread_subproblem_2d.
	   kx,ky,kz (size M) are NU locations in [off+ns/2,off+size-1-ns/2] in each dim.
	   dd (size M complex) are complex source strengths
	   du (size size1*size2*size3) is uniform complex output array
	 */
{
	int ns = opts.nspread;
	T ns2 = (T)ns / 2;          // half spread width
	int nsPadded = 4 * (1 + (ns - 1) / 4); // pad ns to mult of 4
	for (BIGINT i = 0; i < 2 * size1 * size2 * size3; ++i)
		du[i] = 0.0;

	T* pKer1 = kernel_vals1 + begin * nsPadded;
	T* pKer2 = kernel_vals2 + begin * nsPadded;
	T* pKer3 = kernel_vals3 + begin * nsPadded;

	for (BIGINT i = begin; i < end; i++) {           // loop over NU pts
		BIGINT si = sort_indices[i];
		T re0 = dd[2 * si];
		T im0 = dd[2 * si + 1];

		// Combine kernel with complex source value to simplify inner loop
		T ker1val[2 * MAX_NSPREAD];    // here 2* is because of complex
		for (int j = 0; j < ns; j++) {
			ker1val[2 * j] = re0 * pKer1[j];
			ker1val[2 * j + 1] = im0 * pKer1[j];
		}

		// critical inner loop:
		for (int dz = 0; dz < ns; ++dz) {
			BIGINT oz = size1 * size2 * (i3[i] - off3 + dz);        // offset due to z
			for (int dy = 0; dy < ns; ++dy) {
				BIGINT j = oz + size1 * (i2[i] - off2 + dy) + i1[i] - off1;   // should be in subgrid
				T kerval = pKer2[dy] * pKer3[dz];
				T* trg = du + 2 * j;
				for (int dx = 0; dx < 2 * ns; ++dx) {
					trg[dx] += kerval * ker1val[dx];
				}
			}
		}

		pKer1 += nsPadded;
		pKer2 += nsPadded;
		pKer3 += nsPadded;
	}
}

#ifdef __AVX2__
template<>
inline void spread_subproblem_3d<double>(BIGINT* sort_indices,
	BIGINT off1, BIGINT off2, BIGINT off3, 
	BIGINT size1, BIGINT size2, BIGINT size3,
	double* du, double* dd,
	BIGINT* i1, BIGINT* i2, BIGINT* i3,
	double* kernel_vals1, double* kernel_vals2, double* kernel_vals3,
	BIGINT begin, BIGINT end, const spread_opts& opts)
{
	int ns = opts.nspread;
	double ns2 = (double)ns / 2;          // half spread width
	int nsPadded = 4 * (1 + (ns - 1) / 4); // pad ns to mult of 4
	for (BIGINT i = 0; i < 2 * size1 * size2 * size3; ++i)
		du[i] = 0.0;

	double* pKer1 = kernel_vals1 + begin * nsPadded;
	double* pKer2 = kernel_vals2 + begin * nsPadded;
	double* pKer3 = kernel_vals3 + begin * nsPadded;

	switch (nsPadded) {
	case 8:
		for (BIGINT i = begin; i < end; i++) {           // loop over NU pts
			BIGINT si = sort_indices[i];
			__m256d _dd0 = _mm256_permute4x64_pd(
				_mm256_castpd128_pd256(_mm_load_pd(dd + 2 * si)),
				0x44);

			// Combine kernel with complex source value to simplify inner loop
			__m256d _k0 = _mm256_load_pd(pKer1 + 0);
			__m256d _k2 = _mm256_load_pd(pKer1 + 4);

			__m256d _kk0 = _mm256_mul_pd(_dd0, _mm256_permute4x64_pd(_k0, 0x50));
			__m256d _kk1 = _mm256_mul_pd(_dd0, _mm256_permute4x64_pd(_k0, 0xfa));
			__m256d _kk2 = _mm256_mul_pd(_dd0, _mm256_permute4x64_pd(_k2, 0x50));
			__m256d _kk3 = _mm256_mul_pd(_dd0, _mm256_permute4x64_pd(_k2, 0xfa));

			// critical inner loop:
			for (int dz = 0; dz < ns; ++dz) {
				BIGINT oz = size1 * size2 * (i3[i] - off3 + dz);        // offset due to z
				for (int dy = 0; dy < ns; ++dy) {
					BIGINT j = oz + size1 * (i2[i] - off2 + dy) + i1[i] - off1;   // should be in subgrid
					double* pDu = du + 2 * j;

					__m256d _kerval = _mm256_set1_pd(pKer2[dy] * pKer3[dz]);

					__m256d _du0 = _mm256_loadu_pd(pDu + 0);
					__m256d _du1 = _mm256_loadu_pd(pDu + 4);
					__m256d _du2 = _mm256_loadu_pd(pDu + 8);
					__m256d _du3 = _mm256_loadu_pd(pDu + 12);

					_du0 = _mm256_fmadd_pd(_kerval, _kk0, _du0);
					_du1 = _mm256_fmadd_pd(_kerval, _kk1, _du1);
					_du2 = _mm256_fmadd_pd(_kerval, _kk2, _du2);
					_du3 = _mm256_fmadd_pd(_kerval, _kk3, _du3);

					_mm256_storeu_pd(pDu + 0, _du0);
					_mm256_storeu_pd(pDu + 4, _du1);
					_mm256_storeu_pd(pDu + 8, _du2);
					_mm256_storeu_pd(pDu + 12, _du3);
				}
			}

			pKer1 += nsPadded;
			pKer2 += nsPadded;
			pKer3 += nsPadded;
		}
		break;

	case 12:
		for (BIGINT i = begin; i < end; i++) {           // loop over NU pts
			BIGINT si = sort_indices[i];
			__m256d _dd0 = _mm256_permute4x64_pd(
				_mm256_castpd128_pd256(_mm_load_pd(dd + 2 * si)),
				0x44);

			// Combine kernel with complex source value to simplify inner loop
			__m256d _k0 = _mm256_load_pd(pKer1 + 0);
			__m256d _k2 = _mm256_load_pd(pKer1 + 4);
			__m256d _k4 = _mm256_load_pd(pKer1 + 8);

			__m256d _kk0 = _mm256_mul_pd(_dd0, _mm256_permute4x64_pd(_k0, 0x50));
			__m256d _kk1 = _mm256_mul_pd(_dd0, _mm256_permute4x64_pd(_k0, 0xfa));
			__m256d _kk2 = _mm256_mul_pd(_dd0, _mm256_permute4x64_pd(_k2, 0x50));
			__m256d _kk3 = _mm256_mul_pd(_dd0, _mm256_permute4x64_pd(_k2, 0xfa));
			__m256d _kk4 = _mm256_mul_pd(_dd0, _mm256_permute4x64_pd(_k4, 0x50));
			__m256d _kk5 = _mm256_mul_pd(_dd0, _mm256_permute4x64_pd(_k4, 0xfa));

			// critical inner loop:
			for (int dz = 0; dz < ns; ++dz) {
				BIGINT oz = size1 * size2 * (i3[i] - off3 + dz);        // offset due to z
				for (int dy = 0; dy < ns; ++dy) {
					BIGINT j = oz + size1 * (i2[i] - off2 + dy) + i1[i] - off1;   // should be in subgrid
					double* pDu = du + 2 * j;

					__m256d _kerval = _mm256_set1_pd(pKer2[dy] * pKer3[dz]);

					__m256d _du0 = _mm256_loadu_pd(pDu + 0);
					__m256d _du1 = _mm256_loadu_pd(pDu + 4);
					__m256d _du2 = _mm256_loadu_pd(pDu + 8);
					__m256d _du3 = _mm256_loadu_pd(pDu + 12);
					__m256d _du4 = _mm256_loadu_pd(pDu + 16);
					__m256d _du5 = _mm256_loadu_pd(pDu + 20);

					_du0 = _mm256_fmadd_pd(_kerval, _kk0, _du0);
					_du1 = _mm256_fmadd_pd(_kerval, _kk1, _du1);
					_du2 = _mm256_fmadd_pd(_kerval, _kk2, _du2);
					_du3 = _mm256_fmadd_pd(_kerval, _kk3, _du3);
					_du4 = _mm256_fmadd_pd(_kerval, _kk4, _du4);
					_du5 = _mm256_fmadd_pd(_kerval, _kk5, _du5);

					_mm256_storeu_pd(pDu + 0, _du0);
					_mm256_storeu_pd(pDu + 4, _du1);
					_mm256_storeu_pd(pDu + 8, _du2);
					_mm256_storeu_pd(pDu + 12, _du3);
					_mm256_storeu_pd(pDu + 16, _du4);
					_mm256_storeu_pd(pDu + 20, _du5);
				}
			}

			pKer1 += nsPadded;
			pKer2 += nsPadded;
			pKer3 += nsPadded;
		}
		break;

	default:
		for (BIGINT i = begin; i < end; i++) {           // loop over NU pts
			BIGINT si = sort_indices[i];
			double re0 = dd[2 * si];
			double im0 = dd[2 * si + 1];

			// Combine kernel with complex source value to simplify inner loop
			double ker1val[2 * MAX_NSPREAD];    // here 2* is because of complex
			for (int j = 0; j < ns; j++) {
				ker1val[2 * j] = re0 * pKer1[j];
				ker1val[2 * j + 1] = im0 * pKer1[j];
			}

			// critical inner loop:
			for (int dz = 0; dz < ns; ++dz) {
				BIGINT oz = size1 * size2 * (i3[i] - off3 + dz);        // offset due to z
				for (int dy = 0; dy < ns; ++dy) {
					BIGINT j = oz + size1 * (i2[i] - off2 + dy) + i1[i] - off1;   // should be in subgrid
					double kerval = pKer2[dy] * pKer3[dz];
					double* trg = du + 2 * j;
					for (int dx = 0; dx < 2 * ns; ++dx) {
						trg[dx] += kerval * ker1val[dx];
					}
				}
			}

			pKer1 += nsPadded;
			pKer2 += nsPadded;
			pKer3 += nsPadded;
		}
		break;
	}
}

template<>
inline void spread_subproblem_3d<float>(BIGINT* sort_indices,
	BIGINT off1, BIGINT off2, BIGINT off3, 
	BIGINT size1, BIGINT size2, BIGINT size3,
	float* du, float* dd,
	BIGINT* i1, BIGINT* i2, BIGINT* i3,
	float* kernel_vals1, float* kernel_vals2, float* kernel_vals3,
	BIGINT begin, BIGINT end, const spread_opts& opts)
{
	int ns = opts.nspread;
	float ns2 = (float)ns / 2;          // half spread width
	int nsPadded = 4 * (1 + (ns - 1) / 4); // pad ns to mult of 4
	for (BIGINT i = 0; i < 2 * size1 * size2 * size3; ++i)
		du[i] = 0.0;

	float* pKer1 = kernel_vals1 + begin * nsPadded;
	float* pKer2 = kernel_vals2 + begin * nsPadded;
	float* pKer3 = kernel_vals3 + begin * nsPadded;

	__m256i _mask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, -1, -1);
	__m256i _broadcast2 = _mm256_set_epi32(1, 0, 1, 0, 1, 0, 1, 0);
	__m256i _spreadlo = _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0);
	__m256i _spreadhi = _mm256_set_epi32(7, 7, 6, 6, 5, 5, 4, 4);

	switch (nsPadded) {
	case 4:
		for (BIGINT i = begin; i < end; i++) {           // loop over NU pts
			// Combine kernel with complex source value
			BIGINT si = sort_indices[i];
			__m256 _d0 = _mm256_maskload_ps(dd + 2 * si, _mask);
			__m256 _dd0 = _mm256_permutevar8x32_ps(_d0, _broadcast2);

			__m256 _k0 = _mm256_castps128_ps256(_mm_loadu_ps(pKer1));

			__m256 _kk0 = _mm256_mul_ps(_dd0, _mm256_permutevar8x32_ps(_k0, _spreadlo));

			// critical inner loop:
			for (int dz = 0; dz < ns; ++dz) {
				BIGINT oz = size1 * size2 * (i3[i] - off3 + dz);        // offset due to z
				for (int dy = 0; dy < ns; ++dy) {
					BIGINT j = oz + size1 * (i2[i] - off2 + dy) + i1[i] - off1;   // should be in subgrid
					float* pDu = du + 2 * j;

					__m256 _kerval = _mm256_set1_ps(pKer2[dy] * pKer3[dz]);

					__m256 _du0 = _mm256_loadu_ps(pDu + 0);

					_du0 = _mm256_fmadd_ps(_kerval, _kk0, _du0);

					_mm256_storeu_ps(pDu + 0, _du0);
				}
			}

			pKer1 += nsPadded;
			pKer2 += nsPadded;
			pKer3 += nsPadded;
		}
		break;

	case 8:
		for (BIGINT i = begin; i < end; i++) {           // loop over NU pts
			// Combine kernel with complex source value
			BIGINT si = sort_indices[i];
			__m256 _d0 = _mm256_maskload_ps(dd + 2 * si, _mask);
			__m256 _dd0 = _mm256_permutevar8x32_ps(_d0, _broadcast2);

			__m256 _k0 = _mm256_load_ps(pKer1 + 0);

			__m256 _kk0 = _mm256_mul_ps(_dd0, _mm256_permutevar8x32_ps(_k0, _spreadlo));
			__m256 _kk1 = _mm256_mul_ps(_dd0, _mm256_permutevar8x32_ps(_k0, _spreadhi));

			// critical inner loop:
			for (int dz = 0; dz < ns; ++dz) {
				BIGINT oz = size1 * size2 * (i3[i] - off3 + dz);        // offset due to z
				for (int dy = 0; dy < ns; ++dy) {
					BIGINT j = oz + size1 * (i2[i] - off2 + dy) + i1[i] - off1;   // should be in subgrid
					float* pDu = du + 2 * j;

					__m256 _kerval = _mm256_set1_ps(pKer2[dy] * pKer3[dz]);

					__m256 _du0 = _mm256_loadu_ps(pDu + 0);
					__m256 _du1 = _mm256_loadu_ps(pDu + 8);

					_du0 = _mm256_fmadd_ps(_kerval, _kk0, _du0);
					_du1 = _mm256_fmadd_ps(_kerval, _kk1, _du1);

					_mm256_storeu_ps(pDu + 0, _du0);
					_mm256_storeu_ps(pDu + 8, _du1);
				}
			}

			pKer1 += nsPadded;
			pKer2 += nsPadded;
			pKer3 += nsPadded;
		}
		break;

	case 12:
		for (BIGINT i = begin; i < end; i++) {           // loop over NU pts
			// Combine kernel with complex source value
			BIGINT si = sort_indices[i];
			__m256 _d0 = _mm256_maskload_ps(dd + 2 * si, _mask);
			__m256 _dd0 = _mm256_permutevar8x32_ps(_d0, _broadcast2);

			__m256 _k0 = _mm256_loadu_ps(pKer1 + 0);
			__m256 _k2 = _mm256_castps128_ps256(_mm_loadu_ps(pKer1 + 8));

			__m256 _kk0 = _mm256_mul_ps(_dd0, _mm256_permutevar8x32_ps(_k0, _spreadlo));
			__m256 _kk1 = _mm256_mul_ps(_dd0, _mm256_permutevar8x32_ps(_k0, _spreadhi));
			__m256 _kk2 = _mm256_mul_ps(_dd0, _mm256_permutevar8x32_ps(_k2, _spreadlo));

			// critical inner loop:
			for (int dz = 0; dz < ns; ++dz) {
				BIGINT oz = size1 * size2 * (i3[i] - off3 + dz);        // offset due to z
				for (int dy = 0; dy < ns; ++dy) {
					BIGINT j = oz + size1 * (i2[i] - off2 + dy) + i1[i] - off1;   // should be in subgrid
					float* pDu = du + 2 * j;

					__m256 _kerval = _mm256_set1_ps(pKer2[dy] * pKer3[dz]);

					__m256 _du0 = _mm256_loadu_ps(pDu + 0);
					__m256 _du1 = _mm256_loadu_ps(pDu + 8);
					__m256 _du2 = _mm256_loadu_ps(pDu + 16);

					_du0 = _mm256_fmadd_ps(_kerval, _kk0, _du0);
					_du1 = _mm256_fmadd_ps(_kerval, _kk1, _du1);
					_du2 = _mm256_fmadd_ps(_kerval, _kk2, _du2);

					_mm256_storeu_ps(pDu + 0, _du0);
					_mm256_storeu_ps(pDu + 8, _du1);
					_mm256_storeu_ps(pDu + 16, _du2);
				}
			}

			pKer1 += nsPadded;
			pKer2 += nsPadded;
			pKer3 += nsPadded;
		}
		break;

	case 16:
		for (BIGINT i = begin; i < end; i++) {           // loop over NU pts
			// Combine kernel with complex source value
			BIGINT si = sort_indices[i];
			__m256 _d0 = _mm256_maskload_ps(dd + 2 * si, _mask);
			__m256 _dd0 = _mm256_permutevar8x32_ps(_d0, _broadcast2);

			__m256 _k0 = _mm256_load_ps(pKer1 + 0);
			__m256 _k2 = _mm256_load_ps(pKer1 + 8);

			__m256 _kk0 = _mm256_mul_ps(_dd0, _mm256_permutevar8x32_ps(_k0, _spreadlo));
			__m256 _kk1 = _mm256_mul_ps(_dd0, _mm256_permutevar8x32_ps(_k0, _spreadhi));
			__m256 _kk2 = _mm256_mul_ps(_dd0, _mm256_permutevar8x32_ps(_k2, _spreadlo));
			__m256 _kk3 = _mm256_mul_ps(_dd0, _mm256_permutevar8x32_ps(_k2, _spreadhi));

			// critical inner loop:
			for (int dz = 0; dz < ns; ++dz) {
				BIGINT oz = size1 * size2 * (i3[i] - off3 + dz);        // offset due to z
				for (int dy = 0; dy < ns; ++dy) {
					BIGINT j = oz + size1 * (i2[i] - off2 + dy) + i1[i] - off1;   // should be in subgrid
					float* pDu = du + 2 * j;

					__m256 _kerval = _mm256_set1_ps(pKer2[dy] * pKer3[dz]);

					__m256 _du0 = _mm256_loadu_ps(pDu + 0);
					__m256 _du1 = _mm256_loadu_ps(pDu + 8);
					__m256 _du2 = _mm256_loadu_ps(pDu + 16);
					__m256 _du3 = _mm256_loadu_ps(pDu + 24);

					_du0 = _mm256_fmadd_ps(_kerval, _kk0, _du0);
					_du1 = _mm256_fmadd_ps(_kerval, _kk1, _du1);
					_du2 = _mm256_fmadd_ps(_kerval, _kk2, _du2);
					_du3 = _mm256_fmadd_ps(_kerval, _kk3, _du3);

					_mm256_storeu_ps(pDu + 0, _du0);
					_mm256_storeu_ps(pDu + 8, _du1);
					_mm256_storeu_ps(pDu + 16, _du2);
					_mm256_storeu_ps(pDu + 24, _du3);
				}
			}

			pKer1 += nsPadded;
			pKer2 += nsPadded;
			pKer3 += nsPadded;
		}
		break;

	default:
		// Should never get here
		break;
	}
}
#endif
#endif
