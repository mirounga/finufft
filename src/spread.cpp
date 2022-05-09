#ifndef SPREAD_H
#define SPREAD_H

#include <spreadinterp.h>
#include <dataTypes.h>
#include <defs.h>
#include <utils.h>
#include <utils_precindep.h>

#include <immintrin.h>
#include <foldrescale.h>
#include <eval_kernel.h>

#include <stdlib.h>
#include <vector>
#include <math.h>
#include <stdio.h>

#include <tbb/tbb.h>
#include <tbb/scalable_allocator.h>

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

// declarations of purely internal functions...
void add_wrapped_subgrid(BIGINT offset1, BIGINT offset2, BIGINT offset3,
	BIGINT size1, BIGINT size2, BIGINT size3, BIGINT N1,
	BIGINT N2, BIGINT N3, FLT* data_uniform, FLT* du0);
void add_wrapped_subgrid_thread_safe(BIGINT offset1, BIGINT offset2, BIGINT offset3,
	BIGINT size1, BIGINT size2, BIGINT size3, BIGINT N1,
	BIGINT N2, BIGINT N3, FLT* data_uniform, FLT* du0);

template<class T>
void spread_subproblem_1d(BIGINT* sort_indices, BIGINT off1, BIGINT size1, T* du, T* dd,
	BIGINT* i1, 
	T* kernel_vals1, 
	BIGINT size, const spread_opts& opts)
	/* 1D spreader from nonuniform to uniform subproblem grid, without wrapping.
	   Inputs:
	   off1 - integer offset of left size of du subgrid from that of overall fine
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


	T* pKer1 = kernel_vals1;

	for (BIGINT i = 0; i < size; i++) {           // loop over NU pts
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
#ifdef __AVX512F__
template<>
inline void spread_subproblem_1d<double>(BIGINT* sort_indices, BIGINT off1, BIGINT size1, double* du, double* dd,
	BIGINT* i1,
	double* kernel_vals1,
	BIGINT size, const spread_opts& opts)
{
	int ns = opts.nspread;          // a.k.a. w
	double ns2 = (double)ns / 2;          // half spread width
	int nsPadded = 4 * (1 + (ns - 1) / 4); // pad ns to mult of 4
	for (BIGINT i = 0; i < 2 * size1; ++i)         // zero output
		du[i] = 0.0;

	double* pKer1 = kernel_vals1;

	switch (nsPadded) {
	case 8:
		for (BIGINT i = 0; i < size; i++) {           // loop over NU pts
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
		for (BIGINT i = 0; i < size; i++) {           // loop over NU pts
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
		for (BIGINT i = 0; i < size; i++) {           // loop over NU pts
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
		for (BIGINT i = 0; i < size; i++) {           // loop over NU pts
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
	BIGINT size, const spread_opts& opts)
{
	int ns = opts.nspread;          // a.k.a. w
	float ns2 = (float)ns / 2;          // half spread width
	int nsPadded = 4 * (1 + (ns - 1) / 4); // pad ns to mult of 4
	for (BIGINT i = 0; i < 2 * size1; ++i)         // zero output
		du[i] = 0.0;

	float* pKer1 = kernel_vals1;

	float* du_base = du - 2 * off1;

	BIGINT size2 = size & ~0x01ll;

	__m512i _broadcast2 = _mm512_set_epi32(1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0);
	__m512i _spreadlo = _mm512_set_epi32(7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0);
	__m512i _spreadhi = _mm512_set_epi32(15, 15, 14, 14, 13, 13, 12, 12, 11, 11, 10, 10, 9, 9, 8, 8);

	switch (nsPadded) {
	case 4:
		for (BIGINT i = 0; i < size; i++) {           // loop over NU pts
			BIGINT si = sort_indices[i];
			__m512 _d0 = _mm512_maskz_loadu_ps(0x03, dd + 2 * si);
			__m512 _dd0 = _mm512_permutexvar_ps(_broadcast2, _d0);

			// offset rel to subgrid, starts the output indices
			float* pDu = du + 2 * (i1[i] - off1);

			__m512 _k0 = _mm512_castps128_ps512(_mm_load_ps(pKer1 + 0));

			__m512 _kk0 = _mm512_permutexvar_ps(_spreadlo, _k0);

			__m512 _du0 = _mm512_castps256_ps512(_mm256_loadu_ps(pDu + 0));

			_du0 = _mm512_fmadd_ps(_dd0, _kk0, _du0);

			_mm256_storeu_ps(pDu + 0, _mm512_castps512_ps256(_du0));

			pKer1 += nsPadded;
		}
		break;
	case 8:
		for (BIGINT i = 0; i < size2; i += 2) {           // loop over NU pts
			__m512 _k0 = _mm512_load_ps(pKer1);

			// offset rel to subgrid, starts the output indices
			float* pDu0 = du_base + 2 * i1[i + 0];

			__m512 _du0 = _mm512_loadu_ps(pDu0);

			BIGINT si0 = sort_indices[i + 0];
			__m512 _d0 = _mm512_maskz_loadu_ps(0x03, dd + 2 * si0);
			__m512 _dd0 = _mm512_permutexvar_ps(_broadcast2, _d0);

			__m512 _kk0 = _mm512_permutexvar_ps(_spreadlo, _k0);

			_du0 = _mm512_fmadd_ps(_dd0, _kk0, _du0);

			_mm512_storeu_ps(pDu0, _du0);

			float* pDu1 = du_base + 2 * i1[i + 1];

			__m512 _du1 = _mm512_loadu_ps(pDu1);

			BIGINT si1 = sort_indices[i + 1];
			__m512 _d1 = _mm512_maskz_loadu_ps(0x03, dd + 2 * si1);
			__m512 _dd1 = _mm512_permutexvar_ps(_broadcast2, _d1);

			__m512 _kk1 = _mm512_permutexvar_ps(_spreadhi, _k0);

			_du1 = _mm512_fmadd_ps(_dd1, _kk1, _du1);

			_mm512_storeu_ps(pDu1, _du1);

			pKer1 += 16;
		}

		for (BIGINT i = size2; i < size; i++) {           // loop over NU pts
			BIGINT si = sort_indices[i];
			__m512 _d0 = _mm512_maskz_loadu_ps(0x03, dd + 2 * si);
			__m512 _dd0 = _mm512_permutexvar_ps(_broadcast2, _d0);

			// offset rel to subgrid, starts the output indices
			float* pDu = du_base + 2 * i1[i + 0];

			__m512 _du0 = _mm512_loadu_ps(pDu + 0);

			__m512 _k0 = _mm512_castps256_ps512(_mm256_load_ps(pKer1 + 0));

			__m512 _kk0 = _mm512_permutexvar_ps(_spreadlo, _k0);

			_du0 = _mm512_fmadd_ps(_dd0, _kk0, _du0);

			_mm512_storeu_ps(pDu + 0, _du0);

			pKer1 += 8;
		}
		break;
	case 12:
		for (BIGINT i = 0; i < size; i++) {           // loop over NU pts
			BIGINT si = sort_indices[i];
			__m512 _d0 = _mm512_maskz_loadu_ps(0x03, dd + 2 * si);
			__m512 _dd0 = _mm512_permutexvar_ps(_broadcast2, _d0);

			// offset rel to subgrid, starts the output indices
			float* pDu = du + 2 * (i1[i] - off1);

			__m512 _k0 = _mm512_maskz_loadu_ps(0x0fff, pKer1 + 0);

			__m512 _kk0 = _mm512_permutexvar_ps(_spreadlo, _k0);
			__m512 _kk1 = _mm512_permutexvar_ps(_spreadhi, _k0);

			__m512 _du0 = _mm512_loadu_ps(pDu + 0);
			__m512 _du1 = _mm512_castps256_ps512(_mm256_loadu_ps(pDu + 16));

			_du0 = _mm512_fmadd_ps(_dd0, _kk0, _du0);
			_du1 = _mm512_fmadd_ps(_dd0, _kk1, _du1);

			_mm512_storeu_ps(pDu + 0, _du0);
			_mm256_storeu_ps(pDu + 16, _mm512_castps512_ps256(_du1));

			pKer1 += nsPadded;
		}
		break;
	case 16:
		for (BIGINT i = 0; i < size; i++) {           // loop over NU pts
			BIGINT si = sort_indices[i];
			__m512 _d0 = _mm512_maskz_loadu_ps(0x03, dd + 2 * si);
			__m512 _dd0 = _mm512_permutexvar_ps(_broadcast2, _d0);

			// offset rel to subgrid, starts the output indices
			float* pDu = du + 2 * (i1[i] - off1);

			__m512 _k0 = _mm512_load_ps(pKer1 + 0);

			__m512 _kk0 = _mm512_permutexvar_ps(_spreadlo, _k0);
			__m512 _kk1 = _mm512_permutexvar_ps(_spreadhi, _k0);

			__m512 _du0 = _mm512_loadu_ps(pDu + 0);
			__m512 _du1 = _mm512_loadu_ps(pDu + 16);

			_du0 = _mm512_fmadd_ps(_dd0, _kk0, _du0);
			_du1 = _mm512_fmadd_ps(_dd0, _kk1, _du1);

			_mm512_storeu_ps(pDu + 0, _du0);
			_mm512_storeu_ps(pDu + 16, _du1);

			pKer1 += nsPadded;
		}
		break;
	default:
		// Should never get here
		break;
	}
}
#else
template<>
inline void spread_subproblem_1d<double>(BIGINT* sort_indices, BIGINT off1, BIGINT size1, double* du, double* dd,
	BIGINT* i1,
	double* kernel_vals1,
	BIGINT size, const spread_opts& opts)
{
	int ns = opts.nspread;          // a.k.a. w
	double ns2 = (double)ns / 2;          // half spread width
	int nsPadded = 4 * (1 + (ns - 1) / 4); // pad ns to mult of 4
	for (BIGINT i = 0; i < 2 * size1; ++i)         // zero output
		du[i] = 0.0;

	double* pKer1 = kernel_vals1;

	switch (nsPadded) {
	case 8:
		for (BIGINT i = 0; i < size; i++) {           // loop over NU pts
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
		for (BIGINT i = 0; i < size; i++) {           // loop over NU pts
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
		for (BIGINT i = 0; i < size; i++) {           // loop over NU pts
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
		for (BIGINT i = 0; i < size; i++) {           // loop over NU pts
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
	BIGINT size, const spread_opts& opts)
{
	int ns = opts.nspread;          // a.k.a. w
	float ns2 = (float)ns / 2;          // half spread width
	int nsPadded = 4 * (1 + (ns - 1) / 4); // pad ns to mult of 4
	for (BIGINT i = 0; i < 2 * size1; ++i)         // zero output
		du[i] = 0.0;

	float* pKer1 = kernel_vals1;

	__m256i _mask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, -1, -1);
	__m256i _broadcast2 = _mm256_set_epi32(1, 0, 1, 0, 1, 0, 1, 0);
	__m256i _spreadlo = _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0);
	__m256i _spreadhi = _mm256_set_epi32(7, 7, 6, 6, 5, 5, 4, 4);

	switch (nsPadded) {
	case 4:
		for (BIGINT i = 0; i < size; i++) {           // loop over NU pts
			BIGINT si = sort_indices[i];
			__m256 _d0 = _mm256_maskload_ps(dd + 2 * si, _mask);
			__m256 _dd0 = _mm256_permutevar8x32_ps(_d0, _broadcast2);

			// offset rel to subgrid, starts the output indices
			float* pDu = du + 2 * (i1[i] - off1);

			__m256 _k0 = _mm256_castps128_ps256(_mm_load_ps(pKer1 + 0));

			__m256 _kk0 = _mm256_permutevar8x32_ps(_k0, _spreadlo);

			__m256 _du0 = _mm256_loadu_ps(pDu + 0);

			_du0 = _mm256_fmadd_ps(_dd0, _kk0, _du0);

			_mm256_storeu_ps(pDu + 0, _du0);

			pKer1 += nsPadded;
		}
		break;
	case 8:
		for (BIGINT i = 0; i < size; i++) {           // loop over NU pts
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
	case 12:
		for (BIGINT i = 0; i < size; i++) {           // loop over NU pts
			BIGINT si = sort_indices[i];
			__m256 _d0 = _mm256_maskload_ps(dd + 2 * si, _mask);
			__m256 _dd0 = _mm256_permutevar8x32_ps(_d0, _broadcast2);

			// offset rel to subgrid, starts the output indices
			float* pDu = du + 2 * (i1[i] - off1);

			__m256 _k0 = _mm256_loadu_ps(pKer1 + 0);
			__m256 _k2 = _mm256_castps128_ps256(_mm_load_ps(pKer1 + 8));

			__m256 _kk0 = _mm256_permutevar8x32_ps(_k0, _spreadlo);
			__m256 _kk1 = _mm256_permutevar8x32_ps(_k0, _spreadhi);
			__m256 _kk2 = _mm256_permutevar8x32_ps(_k2, _spreadlo);

			__m256 _du0 = _mm256_loadu_ps(pDu + 0);
			__m256 _du1 = _mm256_loadu_ps(pDu + 8);
			__m256 _du2 = _mm256_loadu_ps(pDu + 16);

			_du0 = _mm256_fmadd_ps(_dd0, _kk0, _du0);
			_du1 = _mm256_fmadd_ps(_dd0, _kk1, _du1);
			_du2 = _mm256_fmadd_ps(_dd0, _kk2, _du2);

			_mm256_storeu_ps(pDu + 0, _du0);
			_mm256_storeu_ps(pDu + 8, _du1);
			_mm256_storeu_ps(pDu + 16, _du2);

			pKer1 += nsPadded;
		}
		break;
	case 16:
		for (BIGINT i = 0; i < size; i++) {           // loop over NU pts
			BIGINT si = sort_indices[i];
			__m256 _d0 = _mm256_maskload_ps(dd + 2 * si, _mask);
			__m256 _dd0 = _mm256_permutevar8x32_ps(_d0, _broadcast2);

			// offset rel to subgrid, starts the output indices
			float* pDu = du + 2 * (i1[i] - off1);

			__m256 _k0 = _mm256_load_ps(pKer1 + 0);
			__m256 _k2 = _mm256_load_ps(pKer1 + 8);

			__m256 _kk0 = _mm256_permutevar8x32_ps(_k0, _spreadlo);
			__m256 _kk1 = _mm256_permutevar8x32_ps(_k0, _spreadhi);
			__m256 _kk2 = _mm256_permutevar8x32_ps(_k2, _spreadlo);
			__m256 _kk3 = _mm256_permutevar8x32_ps(_k2, _spreadhi);

			__m256 _du0 = _mm256_loadu_ps(pDu + 0);
			__m256 _du1 = _mm256_loadu_ps(pDu + 8);
			__m256 _du2 = _mm256_loadu_ps(pDu + 16);
			__m256 _du3 = _mm256_loadu_ps(pDu + 24);

			_du0 = _mm256_fmadd_ps(_dd0, _kk0, _du0);
			_du1 = _mm256_fmadd_ps(_dd0, _kk1, _du1);
			_du2 = _mm256_fmadd_ps(_dd0, _kk2, _du2);
			_du3 = _mm256_fmadd_ps(_dd0, _kk3, _du3);

			_mm256_storeu_ps(pDu + 0, _du0);
			_mm256_storeu_ps(pDu + 8, _du1);
			_mm256_storeu_ps(pDu + 16, _du2);
			_mm256_storeu_ps(pDu + 24, _du3);

			pKer1 += nsPadded;
		}
		break;
	default:
		// Should never get here
		break;
	}
}
#endif
#endif

template<class T>
void spread_subproblem_2d(BIGINT* sort_indices,
	BIGINT off1, BIGINT off2,
	BIGINT size1, BIGINT size2,
	T* du, T* dd,
	BIGINT* i1, BIGINT* i2,
	T* kernel_vals1, T* kernel_vals2,
	BIGINT size, const spread_opts& opts)
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

	T* pKer1 = kernel_vals1;
	T* pKer2 = kernel_vals2;

	for (BIGINT i = 0; i < size; i++) {           // loop over NU pts
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
#ifdef __AVX512F__
template<>
inline void spread_subproblem_2d<double>(BIGINT* sort_indices,
	BIGINT off1, BIGINT off2,
	BIGINT size1, BIGINT size2,
	double* du, double* dd,
	BIGINT* i1, BIGINT* i2,
	double* kernel_vals1, double* kernel_vals2,
	BIGINT size, const spread_opts& opts)
{
	int ns = opts.nspread;
	double ns2 = (double)ns / 2;          // half spread width
	int nsPadded = 4 * (1 + (ns - 1) / 4); // pad ns to mult of 4
	for (BIGINT i = 0; i < 2 * size1 * size2; ++i)
		du[i] = 0.0;

	double* pKer1 = kernel_vals1 + 0 * nsPadded;
	double* pKer2 = kernel_vals2 + 0 * nsPadded;

	switch (nsPadded) {
	case 8:
		for (BIGINT i = 0; i < size; i++) {           // loop over NU pts
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
		for (BIGINT i = 0; i < size; i++) {           // loop over NU pts
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
		for (BIGINT i = 0; i < size; i++) {           // loop over NU pts
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
	float* du0, float* dd,
	BIGINT* i1, BIGINT* i2,
	float* kernel_vals1, float* kernel_vals2,
	BIGINT size, const spread_opts& opts)
{
	int ns = opts.nspread;
	float ns2 = (float)ns / 2;          // half spread width
	int nsPadded = 4 * (1 + (ns - 1) / 4); // pad ns to mult of 4

	float* du1 = (float*)malloc(sizeof(float) * 2 * (size1 * size2  + MAX_NSPREAD));
	float* du2 = (float*)malloc(sizeof(float) * 2 * (size1 * size2  + MAX_NSPREAD));
	float* du3 = (float*)malloc(sizeof(float) * 2 * (size1 * size2  + MAX_NSPREAD));

	for (BIGINT i = 0; i < 2 * size1 * size2; ++i)
		du0[i] = du1[i] = du2[i] = du3[i] = 0.0;

	float* pKer1 = kernel_vals1 + 0 * nsPadded;
	float* pKer2 = kernel_vals2 + 0 * nsPadded;

	__m512i _spread = _mm512_set_epi32(7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0);

	BIGINT size4 = size - (size - 0) % 4;

	switch (nsPadded) {
	case 8:
		// Unrolled loop
		for (BIGINT i = 0; i < size4; i += 4) {           // loop over NU pts
			// Combine kernel with complex source value
			BIGINT si0 = sort_indices[i + 0];
			BIGINT si1 = sort_indices[i + 1];
			BIGINT si2 = sort_indices[i + 2];
			BIGINT si3 = sort_indices[i + 3];

			__m128 _d0 = _mm_maskz_loadu_ps(0x3, dd + 2 * si0);
			__m128 _d1 = _mm_maskz_loadu_ps(0x3, dd + 2 * si1);
			__m128 _d2 = _mm_maskz_loadu_ps(0x3, dd + 2 * si2);
			__m128 _d3 = _mm_maskz_loadu_ps(0x3, dd + 2 * si3);

			__m512 _dd0 = _mm512_broadcast_f32x2(_d0);
			__m512 _dd1 = _mm512_broadcast_f32x2(_d1);
			__m512 _dd2 = _mm512_broadcast_f32x2(_d2);
			__m512 _dd3 = _mm512_broadcast_f32x2(_d3);

			__m512 _k0 = _mm512_maskz_loadu_ps(0x00ff, pKer1 + 0);
			__m512 _k1 = _mm512_maskz_loadu_ps(0x00ff, pKer1 + 8);
			__m512 _k2 = _mm512_maskz_loadu_ps(0x00ff, pKer1 + 16);
			__m512 _k3 = _mm512_maskz_loadu_ps(0x00ff, pKer1 + 24);

			__m512 _kk0 = _mm512_mul_ps(_dd0, _mm512_permutexvar_ps(_spread, _k0));
			__m512 _kk1 = _mm512_mul_ps(_dd1, _mm512_permutexvar_ps(_spread, _k1));
			__m512 _kk2 = _mm512_mul_ps(_dd2, _mm512_permutexvar_ps(_spread, _k2));
			__m512 _kk3 = _mm512_mul_ps(_dd3, _mm512_permutexvar_ps(_spread, _k3));

			float* pD0 = du0 + 2 * (size1 * (i2[i + 0] - off2) + i1[i + 0] - off1); // should be in subgrid
			float* pD1 = du1 + 2 * (size1 * (i2[i + 1] - off2) + i1[i + 1] - off1);
			float* pD2 = du2 + 2 * (size1 * (i2[i + 2] - off2) + i1[i + 2] - off1);
			float* pD3 = du3 + 2 * (size1 * (i2[i + 3] - off2) + i1[i + 3] - off1);

			// critical inner loop:
			for (int dy = 0; dy < ns; ++dy) {

				__m512 _ker0 = _mm512_set1_ps(pKer2[dy + 0]);
				__m512 _ker1 = _mm512_set1_ps(pKer2[dy + 8]);
				__m512 _ker2 = _mm512_set1_ps(pKer2[dy + 16]);
				__m512 _ker3 = _mm512_set1_ps(pKer2[dy + 24]);

				__m512 _du0 = _mm512_loadu_ps(pD0);
				__m512 _du1 = _mm512_loadu_ps(pD1);
				__m512 _du2 = _mm512_loadu_ps(pD2);
				__m512 _du3 = _mm512_loadu_ps(pD3);

				_du0 = _mm512_fmadd_ps(_ker0, _kk0, _du0);
				_du1 = _mm512_fmadd_ps(_ker1, _kk1, _du1);
				_du2 = _mm512_fmadd_ps(_ker2, _kk2, _du2);
				_du3 = _mm512_fmadd_ps(_ker3, _kk3, _du3);

				_mm512_storeu_ps(pD0, _du0);
				_mm512_storeu_ps(pD1, _du1);
				_mm512_storeu_ps(pD2, _du2);
				_mm512_storeu_ps(pD3, _du3);

				pD0 += 2 * size1;
				pD1 += 2 * size1;
				pD2 += 2 * size1;
				pD3 += 2 * size1;
			}

			pKer1 += 4 * nsPadded;
			pKer2 += 4 * nsPadded;
		}
		// Short tail
		for (BIGINT i = size4; i < size; i++) {           // loop over NU pts
			// Combine kernel with complex source value
			BIGINT si0 = sort_indices[i];
			__m128 _d0 = _mm_maskz_load_ps(0x3, dd + 2 * si0);
			__m512 _dd0 = _mm512_broadcast_f32x2(_d0);

			__m512 _k0 = _mm512_maskz_load_ps(0x00ff, pKer1 + 0);

			__m512 _kk0 = _mm512_mul_ps(_dd0, _mm512_permutexvar_ps(_spread, _k0));

			float* pD0 = du0 + 2 * (size1 * (i2[i] - off2) + i1[i] - off1); // should be in subgrid

			// critical inner loop:
			for (int dy = 0; dy < ns; ++dy) {

				__m512 _ker0 = _mm512_set1_ps(pKer2[dy]);

				__m512 _du0 = _mm512_loadu_ps(pD0);

				_du0 = _mm512_fmadd_ps(_ker0, _kk0, _du0);

				_mm512_storeu_ps(pD0, _du0);

				pD0 += 2 * size1;
			}

			pKer1 += nsPadded;
			pKer2 += nsPadded;
		}

		for (BIGINT i = 0; i < 2 * size1 * size2; ++i)
			du0[i] += du1[i] + du2[i] + du3[i];

		break;
	default:
		for (BIGINT i = 0; i < size; i++) {           // loop over NU pts
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
				float* trg = du0 + 2 * j;
				for (int dx = 0; dx < 2 * ns; ++dx) {
					trg[dx] += kerval * ker1val[dx];
				}
			}

			pKer1 += nsPadded;
			pKer2 += nsPadded;
		}
		break;
	}

	free(du1); free(du2); free(du3);
}
#else
template<>
inline void spread_subproblem_2d<double>(BIGINT* sort_indices,
	BIGINT off1, BIGINT off2,
	BIGINT size1, BIGINT size2,
	double* du, double* dd,
	BIGINT* i1, BIGINT* i2,
	double* kernel_vals1, double* kernel_vals2,
	BIGINT size, const spread_opts& opts)
{
	int ns = opts.nspread;
	double ns2 = (double)ns / 2;          // half spread width
	int nsPadded = 4 * (1 + (ns - 1) / 4); // pad ns to mult of 4
	for (BIGINT i = 0; i < 2 * size1 * size2; ++i)
		du[i] = 0.0;

	double* pKer1 = kernel_vals1;
	double* pKer2 = kernel_vals2;

	switch (nsPadded) {
	case 8:
		for (BIGINT i = 0; i < size; i++) {           // loop over NU pts
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
		for (BIGINT i = 0; i < size; i++) {           // loop over NU pts
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
		for (BIGINT i = 0; i < size; i++) {           // loop over NU pts
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
	BIGINT size, const spread_opts& opts)
{
	int ns = opts.nspread;
	float ns2 = (float)ns / 2;          // half spread width
	int nsPadded = 4 * (1 + (ns - 1) / 4); // pad ns to mult of 4
	for (BIGINT i = 0; i < 2 * size1 * size2; ++i)
		du[i] = 0.0;

	float* pKer1 = kernel_vals1 + 0 * nsPadded;
	float* pKer2 = kernel_vals2 + 0 * nsPadded;

	__m256i _mask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, -1, -1);
	__m256i _broadcast2 = _mm256_set_epi32(1, 0, 1, 0, 1, 0, 1, 0);
	__m256i _spreadlo = _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0);
	__m256i _spreadhi = _mm256_set_epi32(7, 7, 6, 6, 5, 5, 4, 4);

	switch (nsPadded) {
	case 4:
		for (BIGINT i = 0; i < size; i++) {           // loop over NU pts
			// Combine kernel with complex source value
			BIGINT si = sort_indices[i];
			__m256 _d0 = _mm256_maskload_ps(dd + 2 * si, _mask);
			__m256 _dd0 = _mm256_permutevar8x32_ps(_d0, _broadcast2);

			__m256 _k0 = _mm256_castps128_ps256(_mm_load_ps(pKer1 + 0));

			__m256 _kk0 = _mm256_mul_ps(_dd0, _mm256_permutevar8x32_ps(_k0, _spreadlo));

			// critical inner loop:
			for (int dy = 0; dy < ns; ++dy) {
				BIGINT j = size1 * (i2[i] - off2 + dy) + i1[i] - off1;   // should be in subgrid
				float* pDu = du + 2 * j;

				__m256 _kerval = _mm256_set1_ps(pKer2[dy]);

				__m256 _du0 = _mm256_loadu_ps(pDu + 0);

				_du0 = _mm256_fmadd_ps(_kerval, _kk0, _du0);

				_mm256_storeu_ps(pDu + 0, _du0);
			}

			pKer1 += nsPadded;
			pKer2 += nsPadded;
		}
		break;
	case 8:
		for (BIGINT i = 0; i < size; i++) {           // loop over NU pts
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
		for (BIGINT i = 0; i < size; i++) {           // loop over NU pts
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
		for (BIGINT i = 0; i < size; i++) {           // loop over NU pts
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
#endif

template<class T>
void spread_subproblem_3d(BIGINT* sort_indices, 
	BIGINT off1, BIGINT off2, BIGINT off3, 
	BIGINT size1, BIGINT size2, BIGINT size3,
	T* du, T* dd,
	BIGINT* i1, BIGINT* i2, BIGINT* i3,
	T* kernel_vals1, T* kernel_vals2, T* kernel_vals3,
	BIGINT size, const spread_opts& opts)
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

	T* pKer1 = kernel_vals1;
	T* pKer2 = kernel_vals2;
	T* pKer3 = kernel_vals3;

	for (BIGINT i = 0; i < size; i++) {           // loop over NU pts
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
	BIGINT size, const spread_opts& opts)
{
	int ns = opts.nspread;
	double ns2 = (double)ns / 2;          // half spread width
	int nsPadded = 4 * (1 + (ns - 1) / 4); // pad ns to mult of 4
	for (BIGINT i = 0; i < 2 * size1 * size2 * size3; ++i)
		du[i] = 0.0;

	double* pKer1 = kernel_vals1;
	double* pKer2 = kernel_vals2;
	double* pKer3 = kernel_vals3;

	switch (nsPadded) {
	case 8:
		for (BIGINT i = 0; i < size; i++) {           // loop over NU pts
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
		for (BIGINT i = 0; i < size; i++) {           // loop over NU pts
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
		for (BIGINT i = 0; i < size; i++) {           // loop over NU pts
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
	BIGINT size, const spread_opts& opts)
{
	int ns = opts.nspread;
	float ns2 = (float)ns / 2;          // half spread width
	int nsPadded = 4 * (1 + (ns - 1) / 4); // pad ns to mult of 4
	for (BIGINT i = 0; i < 2 * size1 * size2 * size3; ++i)
		du[i] = 0.0;

	float* pKer1 = kernel_vals1;
	float* pKer2 = kernel_vals2;
	float* pKer3 = kernel_vals3;

	__m256i _mask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, -1, -1);
	__m256i _broadcast2 = _mm256_set_epi32(1, 0, 1, 0, 1, 0, 1, 0);
	__m256i _spreadlo = _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0);
	__m256i _spreadhi = _mm256_set_epi32(7, 7, 6, 6, 5, 5, 4, 4);

	switch (nsPadded) {
	case 4:
		for (BIGINT i = 0; i < size; i++) {           // loop over NU pts
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
		for (BIGINT i = 0; i < size; i++) {           // loop over NU pts
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
		for (BIGINT i = 0; i < size; i++) {           // loop over NU pts
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
		for (BIGINT i = 0; i < size; i++) {           // loop over NU pts
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

void combined_eval_spread_1d(BIGINT* sort_indices,
	FLT* x1, BIGINT* i1,
	const BIGINT& offset1, const BIGINT& size1,
	FLT* du0, FLT* data_nonuniform, BIGINT M, const spread_opts& opts)
{
	int ns = opts.nspread;          // abbrev. for w, kernel width
	int nsPadded = 4 * (1 + (ns - 1) / 4); // pad ns to mult of 4

	FLT* kernel_vals1 = (FLT*)scalable_aligned_malloc(nsPadded * M * sizeof(FLT), 64);

	evaluate_kernel(kernel_vals1, x1, M, opts);

	spread_subproblem_1d<FLT>(sort_indices, offset1, size1, du0, data_nonuniform,
		i1, kernel_vals1,
		M, opts);

	scalable_aligned_free(kernel_vals1);
}

void combined_eval_spread_2d(BIGINT* sort_indices,
	FLT* x1, FLT* x2, BIGINT* i1, BIGINT* i2,
	const BIGINT& offset1, const BIGINT& offset2, const BIGINT& size1, const BIGINT& size2,
	FLT* du0, FLT* data_nonuniform, BIGINT M, const spread_opts& opts)
{
	int ns = opts.nspread;          // abbrev. for w, kernel width
	int nsPadded = 4 * (1 + (ns - 1) / 4); // pad ns to mult of 4

	FLT* kernel_vals1 = (FLT*)scalable_aligned_malloc(nsPadded * M * sizeof(FLT), 64);

	FLT* kernel_vals2 = (FLT*)scalable_aligned_malloc(nsPadded * M * sizeof(FLT), 64);

	evaluate_kernel(kernel_vals1, x1, M, opts);
	evaluate_kernel(kernel_vals2, x2, M, opts);

	spread_subproblem_2d<FLT>(sort_indices, offset1, offset2, size1, size2,
		du0, data_nonuniform,
		i1, i2, kernel_vals1, kernel_vals2,
		M, opts);

	scalable_aligned_free(kernel_vals2);
	scalable_aligned_free(kernel_vals1);
}

void combined_eval_spread_3d(BIGINT* sort_indices,
	FLT* x1, FLT* x2, FLT* x3, BIGINT* i1, BIGINT* i2, BIGINT* i3,
	const BIGINT& offset1, const BIGINT& offset2, const BIGINT& offset3, const BIGINT& size1, const BIGINT& size2, const BIGINT& size3,
	FLT* du0, FLT* data_nonuniform, BIGINT M, const spread_opts& opts)
{
	int ns = opts.nspread;          // abbrev. for w, kernel width
	int nsPadded = 4 * (1 + (ns - 1) / 4); // pad ns to mult of 4

	FLT* kernel_vals1 = (FLT*)scalable_aligned_malloc(nsPadded * M * sizeof(FLT), 64);
	FLT* kernel_vals2 = (FLT*)scalable_aligned_malloc(nsPadded * M * sizeof(FLT), 64);
	FLT* kernel_vals3 = (FLT*)scalable_aligned_malloc(nsPadded * M * sizeof(FLT), 64);

	evaluate_kernel(kernel_vals1, x1, M, opts);
	evaluate_kernel(kernel_vals2, x2, M, opts);
	evaluate_kernel(kernel_vals3, x3, M, opts);

	spread_subproblem_3d<FLT>(sort_indices, offset1, offset2, offset3, size1, size2, size3,
		du0, data_nonuniform,
		i1, i2, i3, kernel_vals1, kernel_vals2, kernel_vals3,
		M, opts);

	scalable_aligned_free(kernel_vals3);
	scalable_aligned_free(kernel_vals2);
	scalable_aligned_free(kernel_vals1);
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

		tbb::parallel_for(tbb::blocked_range<BIGINT>(0, M, 10000),
			[&](const tbb::blocked_range<BIGINT>& r) {
				BIGINT* i1 = NULL, * i2 = NULL, * i3 = NULL;
				FLT* x1 = NULL, * x2 = NULL, * x3 = NULL;

				// get the subgrid which will include padding by roughly nspread/2
				BIGINT offset1 = 0, offset2 = 0, offset3 = 0;
				BIGINT size1 = 1, size2 = 1, size3 = 1; // get_subgrid set

				switch (ndims) {
				case 3:
					i3 = (BIGINT*)scalable_aligned_malloc(sizeof(BIGINT) * r.size(), 64);
					x3 = (FLT*)scalable_aligned_malloc(sizeof(FLT) * r.size(), 64);
					foldrescale<FLT>(sort_indices + r.begin(), kz, i3, x3, N3, r.size(), opts);
					get_subgrid(offset3, size3, i3, r.size(), ns);
					// Fall through
				case 2:
					i2 = (BIGINT*)scalable_aligned_malloc(sizeof(BIGINT) * r.size(), 64);
					x2 = (FLT*)scalable_aligned_malloc(sizeof(FLT) * r.size(), 64);
					foldrescale<FLT>(sort_indices + r.begin(), ky, i2, x2, N2, r.size(), opts);
					get_subgrid(offset2, size2, i2, r.size(), ns);
					// Fall through
				case 1:
					i1 = (BIGINT*)scalable_aligned_malloc(sizeof(BIGINT) * r.size(), 64);
					x1 = (FLT*)scalable_aligned_malloc(sizeof(FLT) * r.size(), 64);
					foldrescale<FLT>(sort_indices + r.begin(), kx, i1, x1, N1, r.size(), opts);
					get_subgrid(offset1, size1, i1, r.size(), ns);
					break;
				}

				if (ndims == 1) {
					// x1 in [-w/2,-w/2+1], up to rounding
					// However if N1*epsmach>O(1) then can cause O(1) errors in x1, hence ppoly
					// kernel evaluation will fall outside their designed domains, >>1 errors.
					// This can only happen if the overall error would be O(1) anyway. Clip x1??
					for (BIGINT i = 0; i < r.size(); i++) {           // loop over NU pts
						if (x1[i] < -ns2) x1[i] = -ns2;
						if (x1[i] > -ns2 + 1) x1[i] = -ns2 + 1;   // ***
					}
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
						combined_eval_spread_1d(sort_indices + r.begin(),
							x1, i1,
							offset1, size1,
							du0, data_nonuniform,
							r.size(), opts);
						break;
					case 2:
						combined_eval_spread_2d(sort_indices + r.begin(),
							x1, x2, i1, i2,
							offset1, offset2, size1, size2,
							du0, data_nonuniform,
							r.size(), opts);
						break;
					case 3:
						combined_eval_spread_3d(sort_indices + r.begin(),
							x1, x2, x3, i1, i2, i3,
							offset1, offset2, offset3, size1, size2, size3,
							du0, data_nonuniform,
							r.size(), opts);
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

				switch (ndims) {
				case 3:
					scalable_aligned_free(x3);
					scalable_aligned_free(i3);
					// Fall through
				case 2:
					scalable_aligned_free(x2);
					scalable_aligned_free(i2);
					// Fall through
				case 1:
					scalable_aligned_free(x1);
					scalable_aligned_free(i1);
				}
			}); // end main loop over subprobs

		if (opts.debug) printf("\tt1 fancy spread: \t%.3g s (%lld subprobs)\n", timer.elapsedsec(), (long long)nb);
	}   // end of choice of which t1 spread type to use
	return 0;
}

void add_wrapped_subgrid(BIGINT offset1, BIGINT offset2, BIGINT offset3,
	BIGINT size1, BIGINT size2, BIGINT size3, BIGINT N1,
	BIGINT N2, BIGINT N3, FLT* data_uniform, FLT* du0)
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
