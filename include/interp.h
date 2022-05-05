#ifndef INTERP_H
#define INTERP_H

#include <immintrin.h>

template<class T>
void interp_line(BIGINT* sort_indices, T* data_nonuniform, T* du_padded,
	T* kernel_vals1,
	BIGINT* i1,
	BIGINT N1,
	int ns, BIGINT size)
	// 1D interpolate complex values from du_padded array to data_nonuniform, using real weights
	// ker[0] through ker[ns-1]. out must be size 2 (real,imag), and du_padded
	// of size 2*N1 (alternating real,imag). i1 is the left-most index in [0,N1)
	// Periodic wrapping in the du_padded array is applied, assuming N1>=ns.
	// dx is index into ker array, j index in complex du_padded (data_uniform) array.
	// Barnett 6/15/17
{
	const int nsPadded = 4 * (1 + (ns - 1) / 4); // pad ns to mult of 4

	// main loop over NU targs, interp each from U
	for (BIGINT i = 0; i < size; i++)
	{
		T* ker = kernel_vals1 + (i * nsPadded);

		T out[] = { 0.0, 0.0 };

		BIGINT j = i1[i];

		for (int dx = 0; dx < ns; ++dx) {
			out[0] += du_padded[2 * j] * ker[dx];
			out[1] += du_padded[2 * j + 1] * ker[dx];
			++j;
		}

		// Copy result buffer to output array
		BIGINT si = sort_indices[i];
		data_nonuniform[2 * si] = out[0];
		data_nonuniform[2 * si + 1] = out[1];
	} // end loop over targets in chunk
}

#ifdef __AVX2__
template<>
inline void interp_line<double>(BIGINT* sort_indices, double* data_nonuniform, double* du_padded,
	double* kernel_vals1,
	BIGINT* i1,
	BIGINT N1,
	int ns, BIGINT size)
{
	const int nsPadded = 4 * (1 + (ns - 1) / 4); // pad ns to mult of 4

	// main loop over NU targs, interp each from U
	double* pKer = kernel_vals1;

	switch (nsPadded) {
	case 8:
		for (BIGINT i = 0; i < size; i++)
		{
			double* pDu = du_padded + 2 * i1[i];

			__m256d _k0 = _mm256_load_pd(pKer + 0);
			__m256d _k2 = _mm256_load_pd(pKer + 4);

			__m256d _kk0 = _mm256_permute4x64_pd(_k0, 0x50);
			__m256d _kk1 = _mm256_permute4x64_pd(_k0, 0xfa);
			__m256d _kk2 = _mm256_permute4x64_pd(_k2, 0x50);
			__m256d _kk3 = _mm256_permute4x64_pd(_k2, 0xfa);

			__m256d _du0 = _mm256_loadu_pd(pDu + 0);
			__m256d _du1 = _mm256_loadu_pd(pDu + 4);
			__m256d _du2 = _mm256_loadu_pd(pDu + 8);
			__m256d _du3 = _mm256_loadu_pd(pDu + 12);

			__m256d _out0 = _mm256_mul_pd(_kk0, _du0);
			__m256d _out1 = _mm256_mul_pd(_kk1, _du1);
			__m256d _out2 = _mm256_mul_pd(_kk2, _du2);
			__m256d _out3 = _mm256_mul_pd(_kk3, _du3);

			_out0 = _mm256_add_pd(_out0, _out1);
			_out2 = _mm256_add_pd(_out2, _out3);
			_out0 = _mm256_add_pd(_out0, _out2);

			__m128d _out = _mm_add_pd(
				_mm256_castpd256_pd128(_out0),
				_mm256_extractf128_pd(_out0, 1));

			// Copy result buffer to output array
			BIGINT si0 = sort_indices[i];

			_mm_store_pd(data_nonuniform + 2 * si0, _out);

			pKer += nsPadded;
		}
		break;
	case 12:
		for (BIGINT i = 0; i < size; i++)
		{
			double* pDu = du_padded + 2 * i1[i];

			__m256d _k0 = _mm256_load_pd(pKer + 0);
			__m256d _k2 = _mm256_load_pd(pKer + 4);
			__m256d _k4 = _mm256_load_pd(pKer + 8);

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

			__m256d _out0 = _mm256_mul_pd(_kk0, _du0);
			__m256d _out1 = _mm256_mul_pd(_kk1, _du1);
			__m256d _out2 = _mm256_mul_pd(_kk2, _du2);
			__m256d _out3 = _mm256_mul_pd(_kk3, _du3);
			__m256d _out4 = _mm256_mul_pd(_kk4, _du4);
			__m256d _out5 = _mm256_mul_pd(_kk5, _du5);

			_out0 = _mm256_add_pd(_out0, _out1);
			_out2 = _mm256_add_pd(_out2, _out3);
			_out4 = _mm256_add_pd(_out4, _out5);
			_out0 = _mm256_add_pd(_out0, _out2);
			_out0 = _mm256_add_pd(_out0, _out4);

			__m128d _out = _mm_add_pd(
				_mm256_castpd256_pd128(_out0),
				_mm256_extractf128_pd(_out0, 1));

			// Copy result buffer to output array
			BIGINT si0 = sort_indices[i];

			_mm_store_pd(data_nonuniform + 2 * si0, _out);

			pKer += nsPadded;
		}
		break;
	case 16:
		for (BIGINT i = 0; i < size; i++)
		{
			double* pDu = du_padded + 2 * i1[i];

			__m256d _k0 = _mm256_load_pd(pKer + 0);
			__m256d _k2 = _mm256_load_pd(pKer + 4);

			__m256d _kk0 = _mm256_permute4x64_pd(_k0, 0x50);
			__m256d _kk1 = _mm256_permute4x64_pd(_k0, 0xfa);
			__m256d _kk2 = _mm256_permute4x64_pd(_k2, 0x50);
			__m256d _kk3 = _mm256_permute4x64_pd(_k2, 0xfa);

			__m256d _du0 = _mm256_loadu_pd(pDu + 0);
			__m256d _du1 = _mm256_loadu_pd(pDu + 4);
			__m256d _du2 = _mm256_loadu_pd(pDu + 8);
			__m256d _du3 = _mm256_loadu_pd(pDu + 12);

			__m256d _out0 = _mm256_mul_pd(_kk0, _du0);
			__m256d _out1 = _mm256_mul_pd(_kk1, _du1);
			__m256d _out2 = _mm256_mul_pd(_kk2, _du2);
			__m256d _out3 = _mm256_mul_pd(_kk3, _du3);

			_k0 = _mm256_load_pd(pKer + 8);
			_k2 = _mm256_load_pd(pKer + 12);

			_kk0 = _mm256_permute4x64_pd(_k0, 0x50);
			_kk1 = _mm256_permute4x64_pd(_k0, 0xfa);
			_kk2 = _mm256_permute4x64_pd(_k2, 0x50);
			_kk3 = _mm256_permute4x64_pd(_k2, 0xfa);

			__m256d _du4 = _mm256_loadu_pd(pDu + 16);
			__m256d _du5 = _mm256_loadu_pd(pDu + 20);
			__m256d _du6 = _mm256_loadu_pd(pDu + 24);
			__m256d _du7 = _mm256_loadu_pd(pDu + 28);

			_out0 = _mm256_fmadd_pd(_kk0, _du4, _out0);
			_out1 = _mm256_fmadd_pd(_kk1, _du5, _out1);
			_out2 = _mm256_fmadd_pd(_kk2, _du6, _out2);
			_out3 = _mm256_fmadd_pd(_kk3, _du7, _out3);

			_out0 = _mm256_add_pd(_out0, _out1);
			_out2 = _mm256_add_pd(_out2, _out3);
			_out0 = _mm256_add_pd(_out0, _out2);

			__m128d _out = _mm_add_pd(
				_mm256_castpd256_pd128(_out0),
				_mm256_extractf128_pd(_out0, 1));

			// Copy result buffer to output array
			BIGINT si0 = sort_indices[i];

			_mm_store_pd(data_nonuniform + 2 * si0, _out);

			pKer += nsPadded;
		}
		break;
	default:
		for (BIGINT i = 0; i < size; i++)
		{
			__m128d _out0 = _mm_setzero_pd();

			BIGINT j0 = i1[i];

			for (int dx = 0; dx < ns; ++dx) {
				__m128d _ker0 = _mm_loaddup_pd(pKer + dx);

				__m128d _du0 = _mm_load_pd(du_padded + 2 * j0);

				_out0 = _mm_fmadd_pd(_ker0, _du0, _out0);

				++j0;
			}

			// Copy result buffer to output array
			BIGINT si0 = sort_indices[i];

			_mm_store_pd(data_nonuniform + 2 * si0, _out0);

			pKer += nsPadded;
		}
		break;
	}
}

template<>
inline void interp_line<float>(BIGINT* sort_indices, float* data_nonuniform, float* du_padded,
	float* kernel_vals1,
	BIGINT* i1,
	BIGINT N1,
	int ns, BIGINT size)
{
	const int nsPadded = 4 * (1 + (ns - 1) / 4); // pad ns to mult of 4

	// main loop over NU targs, interp each from U
	float* pKer = kernel_vals1;

	__m256i _mask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, -1, -1);
	__m256i _spreadlo = _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0);
	__m256i _spreadhi = _mm256_set_epi32(7, 7, 6, 6, 5, 5, 4, 4);

	BIGINT size2 = size & ~0x01;

	switch (nsPadded) {
	case 4:
		// Unrolled loop
		for (BIGINT i = 0; i < size2; i += 2)
		{
			float* pDu0 = du_padded + 2 * i1[i + 0];
			float* pDu1 = du_padded + 2 * i1[i + 1];

			__m256 _k0 = _mm256_castps128_ps256(_mm_load_ps(pKer + 0));

			__m256 _kk0 = _mm256_permutevar8x32_ps(_k0, _spreadlo);
			__m256 _du0 = _mm256_load_ps(pDu0 + 0);
			__m256 _out0 = _mm256_mul_ps(_kk0, _du0);

			__m256 _k1 = _mm256_castps128_ps256(_mm_load_ps(pKer + 4));

			__m256 _kk1 = _mm256_permutevar8x32_ps(_k1, _spreadlo);
			__m256 _du1 = _mm256_loadu_ps(pDu1 + 0);
			__m256 _out1 = _mm256_mul_ps(_kk1, _du1);

			_out0 = _mm256_add_ps(_out0,
				_mm256_permute2f128_ps(_out0, _out0, 0x01));

			_out1 = _mm256_add_ps(_out1,
				_mm256_permute2f128_ps(_out1, _out1, 0x01));

			_out0 = _mm256_add_ps(_out0,
				_mm256_permute_ps(_out0, 0x8e));

			_out1 = _mm256_add_ps(_out1,
				_mm256_permute_ps(_out1, 0x8e));

			// Copy result buffer to output array
			BIGINT si0 = sort_indices[i + 0];
			BIGINT si1 = sort_indices[i + 1];

			_mm256_maskstore_ps(data_nonuniform + 2 * si0, _mask, _out0);
			_mm256_maskstore_ps(data_nonuniform + 2 * si1, _mask, _out1);

			pKer += 8;
		}
		// Short tail
		for (BIGINT i = size2; i < size; i++)
		{
			float* pDu = du_padded + 2 * i1[i];

			__m256 _k0 = _mm256_castps128_ps256(_mm_load_ps(pKer + 0));

			__m256 _kk0 = _mm256_permutevar8x32_ps(_k0, _spreadlo);
			__m256 _du0 = _mm256_loadu_ps(pDu + 0);
			__m256 _out0 = _mm256_mul_ps(_kk0, _du0);

			_out0 = _mm256_add_ps(_out0,
				_mm256_permute2f128_ps(_out0, _out0, 0x01));

			_out0 = _mm256_add_ps(_out0,
				_mm256_permute_ps(_out0, 0x8e));

			// Copy result buffer to output array
			BIGINT si0 = sort_indices[i];

			_mm256_maskstore_ps(data_nonuniform + 2 * si0, _mask, _out0);

			pKer += nsPadded;
		}
		break;
	case 8:
		// Unrolled loop
		for (BIGINT i = 0; i < size2; i += 2)
		{
			float* pDu0 = du_padded + 2 * i1[i + 0];
			float* pDu1 = du_padded + 2 * i1[i + 1];

			__m256 _k0 = _mm256_load_ps(pKer + 0);

			__m256 _kk0 = _mm256_permutevar8x32_ps(_k0, _spreadlo);
			__m256 _du0 = _mm256_load_ps(pDu0 + 0);
			__m256 _out0 = _mm256_mul_ps(_kk0, _du0);

			__m256 _k1 = _mm256_load_ps(pKer + 8);

			__m256 _kk1 = _mm256_permutevar8x32_ps(_k1, _spreadlo);
			__m256 _du1 = _mm256_loadu_ps(pDu1 + 0);
			__m256 _out1 = _mm256_mul_ps(_kk1, _du1);

			_kk0 = _mm256_permutevar8x32_ps(_k0, _spreadhi);
			_du0 = _mm256_loadu_ps(pDu0 + 8);
			_out0 = _mm256_fmadd_ps(_kk0, _du0, _out0);

			_kk1 = _mm256_permutevar8x32_ps(_k1, _spreadhi);
			_du1 = _mm256_loadu_ps(pDu1 + 8);
			_out1 = _mm256_fmadd_ps(_kk1, _du1, _out1);

			_out0 = _mm256_add_ps(_out0,
				_mm256_permute2f128_ps(_out0, _out0, 0x01));

			_out1 = _mm256_add_ps(_out1,
				_mm256_permute2f128_ps(_out1, _out1, 0x01));

			_out0 = _mm256_add_ps(_out0,
				_mm256_permute_ps(_out0, 0x8e));

			_out1 = _mm256_add_ps(_out1,
				_mm256_permute_ps(_out1, 0x8e));

			// Copy result buffer to output array
			BIGINT si0 = sort_indices[i + 0];
			BIGINT si1 = sort_indices[i + 1];

			_mm256_maskstore_ps(data_nonuniform + 2 * si0, _mask, _out0);
			_mm256_maskstore_ps(data_nonuniform + 2 * si1, _mask, _out1);

			pKer += 16;
		}
		// Short tail
		for (BIGINT i = size2; i < size; i++)
		{
			float* pDu = du_padded + 2 * i1[i];

			__m256 _k0 = _mm256_load_ps(pKer + 0);

			__m256 _kk0 = _mm256_permutevar8x32_ps(_k0, _spreadlo);
			__m256 _du0 = _mm256_loadu_ps(pDu + 0);
			__m256 _out0 = _mm256_mul_ps(_kk0, _du0);

			_kk0 = _mm256_permutevar8x32_ps(_k0, _spreadhi);
			_du0 = _mm256_loadu_ps(pDu + 8);
			_out0 = _mm256_fmadd_ps(_kk0, _du0, _out0);

			_out0 = _mm256_add_ps(_out0,
				_mm256_permute2f128_ps(_out0, _out0, 0x01));

			_out0 = _mm256_add_ps(_out0,
				_mm256_permute_ps(_out0, 0x8e));

			// Copy result buffer to output array
			BIGINT si0 = sort_indices[i];

			_mm256_maskstore_ps(data_nonuniform + 2 * si0, _mask, _out0);

			pKer += nsPadded;
		}
		break;
	default:
		for (BIGINT i = 0; i < size; i++)
		{
			__m256 _out0 = _mm256_setzero_ps();

			BIGINT j0 = i1[i];

			for (int dx = 0; dx < ns; ++dx) {
				__m256 _ker0 = _mm256_set1_ps(pKer[dx]);

				__m256 _du0 = _mm256_maskload_ps(du_padded + 2 * j0, _mask);

				_out0 = _mm256_fmadd_ps(_ker0, _du0, _out0);

				++j0;
			}

			// Copy result buffer to output array
			BIGINT si0 = sort_indices[i];

			_mm256_maskstore_ps(data_nonuniform + 2 * si0, _mask, _out0);

			pKer += nsPadded;
		}
		break;
	}
}
#endif

template<class T>
void interp_square(BIGINT* sort_indices, T* data_nonuniform, T* du_padded,
	T* kernel_vals1, T* kernel_vals2,
	BIGINT* i1, BIGINT* i2,
	BIGINT N1, BIGINT N2,
	int ns, BIGINT size)
	// 2D interpolate complex values from du_padded (uniform grid data) array to out value,
	// using ns*ns square of real weights
	// in ker. out must be size 2 (real,imag), and du_padded
	// of size 2*N1*N2 (alternating real,imag). i1 is the left-most index in [0,N1)
	// and i2 the bottom index in [0,N2).
	// Periodic wrapping in the du_padded array is applied, assuming N1,N2>=ns.
	// dx,dy indices into ker array, j index in complex du_padded array.
	// Barnett 6/16/17
{
	const int nsPadded = 4 * (1 + (ns - 1) / 4); // pad ns to mult of 4

	const BIGINT paddedN1 = N1 + 2 * MAX_NSPREAD;

	for (BIGINT i = 0; i < size; i++)
	{
		T* ker1 = kernel_vals1 + (i * nsPadded);
		T* ker2 = kernel_vals2 + (i * nsPadded);

		T out[] = { 0.0, 0.0 };

		// no wrapping: avoid ptrs
		for (int dy = 0; dy < ns; dy++) {
			BIGINT j = paddedN1 * (i2[i] + dy) + i1[i];
			for (int dx = 0; dx < ns; dx++) {
				T k = ker1[dx] * ker2[dy];
				out[0] += du_padded[2 * j] * k;
				out[1] += du_padded[2 * j + 1] * k;
				++j;
			}
		}

		// Copy result buffer to output array
		BIGINT si = sort_indices[i];
		data_nonuniform[2 * si] = out[0];
		data_nonuniform[2 * si + 1] = out[1];
	}
}

#ifdef __AVX2__
#ifdef __AVX512F__
template<>
inline void interp_square<double>(BIGINT* sort_indices, double* data_nonuniform, double* du_padded,
	double* kernel_vals1, double* kernel_vals2,
	BIGINT* i1, BIGINT* i2,
	BIGINT N1, BIGINT N2,
	int ns, BIGINT size)
{
	const int nsPadded = 4 * (1 + (ns - 1) / 4); // pad ns to mult of 4

	const BIGINT paddedN1 = N1 + 2 * MAX_NSPREAD;

	double* pKer1 = kernel_vals1;
	double* pKer2 = kernel_vals2;

	switch (nsPadded) {
	case 8:
		for (BIGINT i = 0; i < size; i++)
		{
			__m256d _out0 = _mm256_setzero_pd();
			__m256d _out1 = _mm256_setzero_pd();
			__m256d _out2 = _mm256_setzero_pd();
			__m256d _out3 = _mm256_setzero_pd();

			for (int dy = 0; dy < ns; dy++) {
				double* pDu = du_padded + 2 * (paddedN1 * (i2[i] + dy) + i1[i]);

				__m256d _ker2 = _mm256_set1_pd(pKer2[dy]);

				__m256d _k0 = _mm256_mul_pd(_ker2, _mm256_load_pd(pKer1 + 0));
				__m256d _k2 = _mm256_mul_pd(_ker2, _mm256_load_pd(pKer1 + 4));

				__m256d _kk0 = _mm256_permute4x64_pd(_k0, 0x50);
				__m256d _kk1 = _mm256_permute4x64_pd(_k0, 0xfa);
				__m256d _kk2 = _mm256_permute4x64_pd(_k2, 0x50);
				__m256d _kk3 = _mm256_permute4x64_pd(_k2, 0xfa);

				__m256d _du0 = _mm256_loadu_pd(pDu + 0);
				__m256d _du1 = _mm256_loadu_pd(pDu + 4);
				__m256d _du2 = _mm256_loadu_pd(pDu + 8);
				__m256d _du3 = _mm256_loadu_pd(pDu + 12);

				_out0 = _mm256_fmadd_pd(_kk0, _du0, _out0);
				_out1 = _mm256_fmadd_pd(_kk1, _du1, _out1);
				_out2 = _mm256_fmadd_pd(_kk2, _du2, _out2);
				_out3 = _mm256_fmadd_pd(_kk3, _du3, _out3);
			}

			_out0 = _mm256_add_pd(_out0, _out1);
			_out2 = _mm256_add_pd(_out2, _out3);
			_out0 = _mm256_add_pd(_out0, _out2);

			__m128d _out = _mm_add_pd(
				_mm256_castpd256_pd128(_out0),
				_mm256_extractf128_pd(_out0, 1));

			// Copy result buffer to output array
			BIGINT si0 = sort_indices[i];

			_mm_store_pd(data_nonuniform + 2 * si0, _out);

			pKer1 += nsPadded;
			pKer2 += nsPadded;
		}
		break;
	case 12:
		for (BIGINT i = 0; i < size; i++)
		{
			__m256d _out0 = _mm256_setzero_pd();
			__m256d _out1 = _mm256_setzero_pd();
			__m256d _out2 = _mm256_setzero_pd();
			__m256d _out3 = _mm256_setzero_pd();
			__m256d _out4 = _mm256_setzero_pd();
			__m256d _out5 = _mm256_setzero_pd();

			for (int dy = 0; dy < ns; dy++) {
				double* pDu = du_padded + 2 * (paddedN1 * (i2[i] + dy) + i1[i]);

				__m256d _ker2 = _mm256_set1_pd(pKer2[dy]);

				__m256d _k0 = _mm256_mul_pd(_ker2, _mm256_load_pd(pKer1 + 0));
				__m256d _k2 = _mm256_mul_pd(_ker2, _mm256_load_pd(pKer1 + 4));
				__m256d _k4 = _mm256_mul_pd(_ker2, _mm256_load_pd(pKer1 + 8));

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

				_out0 = _mm256_fmadd_pd(_kk0, _du0, _out0);
				_out1 = _mm256_fmadd_pd(_kk1, _du1, _out1);
				_out2 = _mm256_fmadd_pd(_kk2, _du2, _out2);
				_out3 = _mm256_fmadd_pd(_kk3, _du3, _out3);
				_out4 = _mm256_fmadd_pd(_kk4, _du4, _out4);
				_out5 = _mm256_fmadd_pd(_kk5, _du5, _out5);
			}

			_out0 = _mm256_add_pd(_out0, _out1);
			_out2 = _mm256_add_pd(_out2, _out3);
			_out4 = _mm256_add_pd(_out4, _out5);
			_out0 = _mm256_add_pd(_out0, _out2);
			_out0 = _mm256_add_pd(_out0, _out4);

			__m128d _out = _mm_add_pd(
				_mm256_castpd256_pd128(_out0),
				_mm256_extractf128_pd(_out0, 1));

			// Copy result buffer to output array
			BIGINT si0 = sort_indices[i];

			_mm_store_pd(data_nonuniform + 2 * si0, _out);

			pKer1 += nsPadded;
			pKer2 += nsPadded;
		}
		break;
	default:
		for (BIGINT i = 0; i < size; i++)
		{
			__m128d _out = _mm_setzero_pd();

			// no wrapping: avoid ptrs
			for (int dy = 0; dy < ns; dy++) {
				double* pDu = du_padded + 2 * (paddedN1 * (i2[i] + dy) + i1[i]);

				for (int dx = 0; dx < ns; dx++) {
					__m128d _kk = _mm_set1_pd(pKer1[dx] * pKer2[dy]);

					__m128d _du = _mm_load_pd(pDu);

					_out = _mm_fmadd_pd(_kk, _du, _out);

					pDu += 2;
				}
			}

			// Copy result buffer to output array
			BIGINT si0 = sort_indices[i];

			_mm_store_pd(data_nonuniform + 2 * si0, _out);

			pKer1 += nsPadded;
			pKer2 += nsPadded;
		}
		break;
	}
}
template<>
inline void interp_square<float>(BIGINT* sort_indices, float* data_nonuniform, float* du_padded,
	float* kernel_vals1, float* kernel_vals2,
	BIGINT* i1, BIGINT* i2,
	BIGINT N1, BIGINT N2,
	int ns, BIGINT size)
{
	const int nsPadded = 4 * (1 + (ns - 1) / 4); // pad ns to mult of 4

	const BIGINT paddedN1 = N1 + 2 * MAX_NSPREAD;

	float* pKer1 = kernel_vals1;
	float* pKer2 = kernel_vals2;

	__m512i _spreadlo = _mm512_set_epi32(7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0);
	__m512i _spreadhi = _mm512_set_epi32(15, 15, 14, 14, 13, 13, 12, 12, 11, 11, 10, 10, 9, 9, 8, 8);

	BIGINT size4 = size - size % 4;

	switch (nsPadded) {
	case 8:
		// Unrolled loop
		for (BIGINT i = 0; i < size4; i += 4)
		{
			__m512 _k0 = _mm512_loadu_ps(pKer1 + 0);
			__m512 _k2 = _mm512_loadu_ps(pKer1 + 16);

			__m512 _ka0 = _mm512_permutexvar_ps(_spreadlo, _k0);
			__m512 _ka1 = _mm512_permutexvar_ps(_spreadhi, _k0);
			__m512 _ka2 = _mm512_permutexvar_ps(_spreadlo, _k2);
			__m512 _ka3 = _mm512_permutexvar_ps(_spreadhi, _k2);

			__m512 _out0 = _mm512_setzero_ps();
			__m512 _out1 = _mm512_setzero_ps();
			__m512 _out2 = _mm512_setzero_ps();
			__m512 _out3 = _mm512_setzero_ps();

			float* pDu0 = du_padded + 2 * (paddedN1 * i2[i + 0] + i1[i + 0]);
			float* pDu1 = du_padded + 2 * (paddedN1 * i2[i + 1] + i1[i + 1]);
			float* pDu2 = du_padded + 2 * (paddedN1 * i2[i + 2] + i1[i + 2]);
			float* pDu3 = du_padded + 2 * (paddedN1 * i2[i + 3] + i1[i + 3]);

			for (int dy = 0; dy < ns; dy++) {
				__m512 _ker0 = _mm512_set1_ps(pKer2[dy + 0]);
				__m512 _ker1 = _mm512_set1_ps(pKer2[dy + 8]);
				__m512 _ker2 = _mm512_set1_ps(pKer2[dy + 16]);
				__m512 _ker3 = _mm512_set1_ps(pKer2[dy + 24]);

				__m512 _ma0 = _mm512_mul_ps(_ker0, _ka0);
				__m512 _ma1 = _mm512_mul_ps(_ker1, _ka1);
				__m512 _ma2 = _mm512_mul_ps(_ker2, _ka2);
				__m512 _ma3 = _mm512_mul_ps(_ker3, _ka3);

				__m512 _du0 = _mm512_loadu_ps(pDu0);
				__m512 _du1 = _mm512_loadu_ps(pDu1);
				__m512 _du2 = _mm512_loadu_ps(pDu2);
				__m512 _du3 = _mm512_loadu_ps(pDu3);

				_out0 = _mm512_fmadd_ps(_ma0, _du0, _out0);
				_out1 = _mm512_fmadd_ps(_ma1, _du1, _out1);
				_out2 = _mm512_fmadd_ps(_ma2, _du2, _out2);
				_out3 = _mm512_fmadd_ps(_ma3, _du3, _out3);

				pDu0 += 2 * paddedN1;
				pDu1 += 2 * paddedN1;
				pDu2 += 2 * paddedN1;
				pDu3 += 2 * paddedN1;
			}

			_out0 = _mm512_add_ps(_out0, _mm512_shuffle_f32x4(_out0, _out0, 0x8e));
			_out1 = _mm512_add_ps(_out1, _mm512_shuffle_f32x4(_out1, _out1, 0x8e));
			_out2 = _mm512_add_ps(_out2, _mm512_shuffle_f32x4(_out2, _out2, 0x8e));
			_out3 = _mm512_add_ps(_out3, _mm512_shuffle_f32x4(_out3, _out3, 0x8e));

			_out0 = _mm512_add_ps(_out0, _mm512_shuffle_f32x4(_out0, _out0, 0xb1));
			_out1 = _mm512_add_ps(_out1, _mm512_shuffle_f32x4(_out1, _out1, 0xb1));
			_out2 = _mm512_add_ps(_out2, _mm512_shuffle_f32x4(_out2, _out2, 0xb1));
			_out3 = _mm512_add_ps(_out3, _mm512_shuffle_f32x4(_out3, _out3, 0xb1));

			_out0 = _mm512_add_ps(_out0, _mm512_permute_ps(_out0, 0x8e));
			_out1 = _mm512_add_ps(_out1, _mm512_permute_ps(_out1, 0x8e));
			_out2 = _mm512_add_ps(_out2, _mm512_permute_ps(_out2, 0x8e));
			_out3 = _mm512_add_ps(_out3, _mm512_permute_ps(_out3, 0x8e));

			// Copy result buffer to output array
			BIGINT si0 = sort_indices[i + 0];
			BIGINT si1 = sort_indices[i + 1];
			BIGINT si2 = sort_indices[i + 2];
			BIGINT si3 = sort_indices[i + 3];

			_mm512_mask_store_ps(data_nonuniform + 2 * si0, 0x0003, _out0);
			_mm512_mask_store_ps(data_nonuniform + 2 * si1, 0x0003, _out1);
			_mm512_mask_store_ps(data_nonuniform + 2 * si2, 0x0003, _out2);
			_mm512_mask_store_ps(data_nonuniform + 2 * si3, 0x0003, _out3);

			pKer1 += 32;
			pKer2 += 32;
		}
		// Short tail
		for (BIGINT i = size4; i < size; i++)
		{
			__m512 _k0 = _mm512_maskz_load_ps(0x00ff, pKer1 + 0);

			__m512 _ka0 = _mm512_permutexvar_ps(_spreadlo, _k0);

			__m512 _out0 = _mm512_setzero_ps();

			for (int dy = 0; dy < ns; dy++) {
				float* pDu0 = du_padded + 2 * (paddedN1 * (i2[i] + dy) + i1[i]);

				__m512 _ker0 = _mm512_set1_ps(pKer2[dy]);

				__m512 _ma0 = _mm512_mul_ps(_ker0, _ka0);

				__m512 _du0 = _mm512_loadu_ps(pDu0 + 0);

				_out0 = _mm512_fmadd_ps(_ma0, _du0, _out0);
			}

			_out0 = _mm512_add_ps(_out0, _mm512_shuffle_f32x4(_out0, _out0, 0x8e));

			_out0 = _mm512_add_ps(_out0, _mm512_shuffle_f32x4(_out0, _out0, 0xb1));

			_out0 = _mm512_add_ps(_out0, _mm512_permute_ps(_out0, 0x8e));

			// Copy result buffer to output array
			BIGINT si0 = sort_indices[i + 0];

			_mm512_mask_store_ps(data_nonuniform + 2 * si0, 0x0003, _out0);

			pKer1 += 8;
			pKer2 += 8;
		}
		break;
	case 12:
		for (BIGINT i = 0; i < size; i++)
		{
			__m512 _k0 = _mm512_maskz_load_ps(0x0fff, pKer1 + 0);
			__m512 _k1 = _mm512_maskz_load_ps(0x0fff, pKer1 + 0);
			__m512 _k2 = _mm512_maskz_load_ps(0x0fff, pKer1 + 0);
			__m512 _k3 = _mm512_maskz_load_ps(0x0fff, pKer1 + 0);

			__m512 _ka0 = _mm512_permutexvar_ps(_spreadlo, _k0);
			__m512 _kb0 = _mm512_permutexvar_ps(_spreadhi, _k0);

			__m512 _out0 = _mm512_setzero_ps();
			__m512 _out1 = _mm512_setzero_ps();
			__m512 _out2 = _mm512_setzero_ps();
			__m512 _out3 = _mm512_setzero_ps();

			for (int dy = 0; dy < ns; dy++) {
				float* pDu0 = du_padded + 2 * (paddedN1 * (i2[i] + dy) + i1[i]);

				__m512 _ker0 = _mm512_set1_ps(pKer2[dy]);

				{
					__m512 _ma0 = _mm512_mul_ps(_ker0, _ka0);

					__m512 _du0 = _mm512_loadu_ps(pDu0 + 0);

					_out0 = _mm512_fmadd_ps(_ma0, _du0, _out0);
				}

				{
					__m512 _mb0 = _mm512_mul_ps(_ker0, _kb0);

					__m512 _du0 = _mm512_maskz_load_ps(0x00ff, pDu0 + 16);

					_out0 = _mm512_fmadd_ps(_mb0, _du0, _out0);
				}
			}

			_out0 = _mm512_add_ps(_out0, _mm512_shuffle_f32x4(_out0, _out0, 0x8e));

			_out0 = _mm512_add_ps(_out0, _mm512_shuffle_f32x4(_out0, _out0, 0xb1));

			_out0 = _mm512_add_ps(_out0, _mm512_permute_ps(_out0, 0x8e));

			// Copy result buffer to output array
			BIGINT si0 = sort_indices[i + 0];

			_mm512_mask_store_ps(data_nonuniform + 2 * si0, 0x0003, _out0);

			pKer1 += 12;
			pKer2 += 12;
		}
		break;
	default:
		for (BIGINT i = 0; i < size; i++)
		{
			__m256 _out0 = _mm256_setzero_ps();

			// no wrapping: avoid ptrs
			for (int dy = 0; dy < ns; dy++) {
				float* pDu = du_padded + 2 * (paddedN1 * (i2[i] + dy) + i1[i]);

				for (int dx = 0; dx < ns; dx++) {
					__m256 _kk = _mm256_set1_ps(pKer1[dx] * pKer2[dy]);

					__m256 _du = _mm256_maskz_load_ps(0x03, pDu);

					_out0 = _mm256_fmadd_ps(_kk, _du, _out0);

					pDu += 2;
				}
			}

			// Copy result buffer to output array
			BIGINT si0 = sort_indices[i];

			_mm256_mask_store_ps(data_nonuniform + 2 * si0, 0x03, _out0);

			pKer1 += nsPadded;
			pKer2 += nsPadded;
		}
		break;
	}
}
#else
template<>
inline void interp_square<double>(BIGINT* sort_indices, double* data_nonuniform, double* du_padded,
	double* kernel_vals1, double* kernel_vals2,
	BIGINT* i1, BIGINT* i2,
	BIGINT N1, BIGINT N2,
	int ns, BIGINT size)
{
	const int nsPadded = 4 * (1 + (ns - 1) / 4); // pad ns to mult of 4

	const BIGINT paddedN1 = N1 + 2 * MAX_NSPREAD;

	double* pKer1 = kernel_vals1;
	double* pKer2 = kernel_vals2;

	switch (nsPadded) {
	case 8:
		for (BIGINT i = 0; i < size; i++)
		{
			__m256d _out0 = _mm256_setzero_pd();
			__m256d _out1 = _mm256_setzero_pd();
			__m256d _out2 = _mm256_setzero_pd();
			__m256d _out3 = _mm256_setzero_pd();

			for (int dy = 0; dy < ns; dy++) {
				double* pDu = du_padded + 2 * (paddedN1 * (i2[i] + dy) + i1[i]);

				__m256d _ker2 = _mm256_set1_pd(pKer2[dy]);

				__m256d _k0 = _mm256_mul_pd(_ker2, _mm256_load_pd(pKer1 + 0));
				__m256d _k2 = _mm256_mul_pd(_ker2, _mm256_load_pd(pKer1 + 4));

				__m256d _kk0 = _mm256_permute4x64_pd(_k0, 0x50);
				__m256d _kk1 = _mm256_permute4x64_pd(_k0, 0xfa);
				__m256d _kk2 = _mm256_permute4x64_pd(_k2, 0x50);
				__m256d _kk3 = _mm256_permute4x64_pd(_k2, 0xfa);

				__m256d _du0 = _mm256_loadu_pd(pDu + 0);
				__m256d _du1 = _mm256_loadu_pd(pDu + 4);
				__m256d _du2 = _mm256_loadu_pd(pDu + 8);
				__m256d _du3 = _mm256_loadu_pd(pDu + 12);

				_out0 = _mm256_fmadd_pd(_kk0, _du0, _out0);
				_out1 = _mm256_fmadd_pd(_kk1, _du1, _out1);
				_out2 = _mm256_fmadd_pd(_kk2, _du2, _out2);
				_out3 = _mm256_fmadd_pd(_kk3, _du3, _out3);
			}

			_out0 = _mm256_add_pd(_out0, _out1);
			_out2 = _mm256_add_pd(_out2, _out3);
			_out0 = _mm256_add_pd(_out0, _out2);

			__m128d _out = _mm_add_pd(
				_mm256_castpd256_pd128(_out0),
				_mm256_extractf128_pd(_out0, 1));

			// Copy result buffer to output array
			BIGINT si0 = sort_indices[i];

			_mm_store_pd(data_nonuniform + 2 * si0, _out);

			pKer1 += nsPadded;
			pKer2 += nsPadded;
		}
		break;
	case 12:
		for (BIGINT i = 0; i < size; i++)
		{
			__m256d _out0 = _mm256_setzero_pd();
			__m256d _out1 = _mm256_setzero_pd();
			__m256d _out2 = _mm256_setzero_pd();
			__m256d _out3 = _mm256_setzero_pd();
			__m256d _out4 = _mm256_setzero_pd();
			__m256d _out5 = _mm256_setzero_pd();

			for (int dy = 0; dy < ns; dy++) {
				double* pDu = du_padded + 2 * (paddedN1 * (i2[i] + dy) + i1[i]);

				__m256d _ker2 = _mm256_set1_pd(pKer2[dy]);

				__m256d _k0 = _mm256_mul_pd(_ker2, _mm256_load_pd(pKer1 + 0));
				__m256d _k2 = _mm256_mul_pd(_ker2, _mm256_load_pd(pKer1 + 4));
				__m256d _k4 = _mm256_mul_pd(_ker2, _mm256_load_pd(pKer1 + 8));

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

				_out0 = _mm256_fmadd_pd(_kk0, _du0, _out0);
				_out1 = _mm256_fmadd_pd(_kk1, _du1, _out1);
				_out2 = _mm256_fmadd_pd(_kk2, _du2, _out2);
				_out3 = _mm256_fmadd_pd(_kk3, _du3, _out3);
				_out4 = _mm256_fmadd_pd(_kk4, _du4, _out4);
				_out5 = _mm256_fmadd_pd(_kk5, _du5, _out5);
			}

			_out0 = _mm256_add_pd(_out0, _out1);
			_out2 = _mm256_add_pd(_out2, _out3);
			_out4 = _mm256_add_pd(_out4, _out5);
			_out0 = _mm256_add_pd(_out0, _out2);
			_out0 = _mm256_add_pd(_out0, _out4);

			__m128d _out = _mm_add_pd(
				_mm256_castpd256_pd128(_out0),
				_mm256_extractf128_pd(_out0, 1));

			// Copy result buffer to output array
			BIGINT si0 = sort_indices[i];

			_mm_store_pd(data_nonuniform + 2 * si0, _out);

			pKer1 += nsPadded;
			pKer2 += nsPadded;
		}
		break;
	default:
		for (BIGINT i = 0; i < size; i++)
		{
			__m128d _out = _mm_setzero_pd();

			// no wrapping: avoid ptrs
			for (int dy = 0; dy < ns; dy++) {
				double* pDu = du_padded + 2 * (paddedN1 * (i2[i] + dy) + i1[i]);

				for (int dx = 0; dx < ns; dx++) {
					__m128d _kk = _mm_set1_pd(pKer1[dx] * pKer2[dy]);

					__m128d _du = _mm_load_pd(pDu);

					_out = _mm_fmadd_pd(_kk, _du, _out);

					pDu += 2;
				}
			}

			// Copy result buffer to output array
			BIGINT si0 = sort_indices[i];

			_mm_store_pd(data_nonuniform + 2 * si0, _out);

			pKer1 += nsPadded;
			pKer2 += nsPadded;
		}
		break;
	}
}
template<>
inline void interp_square<float>(BIGINT* sort_indices, float* data_nonuniform, float* du_padded,
	float* kernel_vals1, float* kernel_vals2,
	BIGINT* i1, BIGINT* i2,
	BIGINT N1, BIGINT N2,
	int ns, BIGINT size)
{
	const int nsPadded = 4 * (1 + (ns - 1) / 4); // pad ns to mult of 4

	const BIGINT paddedN1 = N1 + 2 * MAX_NSPREAD;

	float* pKer1 = kernel_vals1;
	float* pKer2 = kernel_vals2;

	__m256i _mask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, -1, -1);
	__m256i _spreadlo = _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0);
	__m256i _spreadhi = _mm256_set_epi32(7, 7, 6, 6, 5, 5, 4, 4);

	switch (nsPadded) {
	case 4:
		for (BIGINT i = 0; i < size; i++)
		{
			__m256 _out0 = _mm256_setzero_ps();

			__m256 _ker1 = _mm256_castps128_ps256(_mm_load_ps(pKer1 + 0));
			__m256 _kk0 = _mm256_permutevar8x32_ps(_ker1, _spreadlo);

			for (int dy = 0; dy < ns; dy++) {
				float* pDu0 = du_padded + 2 * (paddedN1 * (i2[i] + dy) + i1[i]);

				__m256 _ker2 = _mm256_set1_ps(pKer2[dy]);
				__m256 _k0 = _mm256_mul_ps(_ker2, _kk0);

				__m256 _du0 = _mm256_loadu_ps(pDu0 + 0);
				_out0 = _mm256_fmadd_ps(_k0, _du0, _out0);
			}

			_out0 = _mm256_add_ps(_out0,
				_mm256_permute2f128_ps(_out0, _out0, 0x01));

			_out0 = _mm256_add_ps(_out0,
				_mm256_permute_ps(_out0, 0x8e));

			// Copy result buffer to output array
			BIGINT si0 = sort_indices[i + 0];

			_mm256_maskstore_ps(data_nonuniform + 2 * si0, _mask, _out0);

			pKer1 += 4;
			pKer2 += 4;
		}
		break;
	case 8:
		for (BIGINT i = 0; i < size; i++)
		{
			__m256 _out0 = _mm256_setzero_ps();
			__m256 _out1 = _mm256_setzero_ps();

			for (int dy = 0; dy < ns; dy++) {
				float* pDu0 = du_padded + 2 * (paddedN1 * (i2[i] + dy) + i1[i]);

				__m256 _ker2 = _mm256_set1_ps(pKer2[dy]);
				__m256 _k0 = _mm256_mul_ps(_ker2, _mm256_load_ps(pKer1 + 0));

				__m256 _kk0 = _mm256_permutevar8x32_ps(_k0, _spreadlo);
				__m256 _du0 = _mm256_loadu_ps(pDu0 + 0);
				_out0 = _mm256_fmadd_ps(_kk0, _du0, _out0);

				__m256 _kk1 = _mm256_permutevar8x32_ps(_k0, _spreadhi);
				__m256 _du1 = _mm256_loadu_ps(pDu0 + 8);
				_out1 = _mm256_fmadd_ps(_kk1, _du1, _out1); 
			}

			_out0 = _mm256_add_ps(_out0, _out1);

			_out0 = _mm256_add_ps(_out0,
				_mm256_permute2f128_ps(_out0, _out0, 0x01));

			_out0 = _mm256_add_ps(_out0,
				_mm256_permute_ps(_out0, 0x8e));

			// Copy result buffer to output array
			BIGINT si0 = sort_indices[i + 0];

			_mm256_maskstore_ps(data_nonuniform + 2 * si0, _mask, _out0);

			pKer1 += 8;
			pKer2 += 8;
		}
		break;
	case 12:
		for (BIGINT i = 0; i < size; i++)
		{
			__m256 _out0 = _mm256_setzero_ps();
			__m256 _out1 = _mm256_setzero_ps();
			__m256 _out2 = _mm256_setzero_ps();

			for (int dy = 0; dy < ns; dy++) {
				float* pDu0 = du_padded + 2 * (paddedN1 * (i2[i] + dy) + i1[i]);

				__m256 _ker2 = _mm256_set1_ps(pKer2[dy]);
				__m256 _k0 = _mm256_mul_ps(_ker2, _mm256_loadu_ps(pKer1 + 0));

				__m256 _kk0 = _mm256_permutevar8x32_ps(_k0, _spreadlo);
				__m256 _du0 = _mm256_loadu_ps(pDu0 + 0);
				_out0 = _mm256_fmadd_ps(_kk0, _du0, _out0);

				__m256 _kk1 = _mm256_permutevar8x32_ps(_k0, _spreadhi);
				__m256 _du1 = _mm256_loadu_ps(pDu0 + 8);
				_out1 = _mm256_fmadd_ps(_kk1, _du1, _out1);

				__m256 _k2 = _mm256_mul_ps(_ker2, _mm256_castps128_ps256(_mm_load_ps(pKer1 + 8)));

				__m256 _kk2 = _mm256_permutevar8x32_ps(_k2, _spreadlo);
				__m256 _du2 = _mm256_loadu_ps(pDu0 + 16);
				_out2 = _mm256_fmadd_ps(_kk2, _du2, _out2);
			}

			_out0 = _mm256_add_ps(_out0, _out1);
			_out0 = _mm256_add_ps(_out0, _out2);

			_out0 = _mm256_add_ps(_out0,
				_mm256_permute2f128_ps(_out0, _out0, 0x01));

			_out0 = _mm256_add_ps(_out0,
				_mm256_permute_ps(_out0, 0x8e));

			// Copy result buffer to output array
			BIGINT si0 = sort_indices[i + 0];

			_mm256_maskstore_ps(data_nonuniform + 2 * si0, _mask, _out0);

			pKer1 += 12;
			pKer2 += 12;
		}
		break;
	default:
		for (BIGINT i = 0; i < size; i++)
		{
			__m256 _out0 = _mm256_setzero_ps();

			// no wrapping: avoid ptrs
			for (int dy = 0; dy < ns; dy++) {
				float* pDu = du_padded + 2 * (paddedN1 * (i2[i] + dy) + i1[i]);

				for (int dx = 0; dx < ns; dx++) {
					__m256 _kk = _mm256_set1_ps(pKer1[dx] * pKer2[dy]);

					__m256 _du = _mm256_maskload_ps(pDu, _mask);

					_out0 = _mm256_fmadd_ps(_kk, _du, _out0);

					pDu += 2;
				}
			}

			// Copy result buffer to output array
			BIGINT si0 = sort_indices[i];

			_mm256_maskstore_ps(data_nonuniform + 2 * si0, _mask, _out0);

			pKer1 += nsPadded;
			pKer2 += nsPadded;
		}
		break;
	}
}
#endif
#endif

template<class T>
void interp_cube(BIGINT* sort_indices, T* data_nonuniform, T* du_padded,
	T* kernel_vals1, T* kernel_vals2, T* kernel_vals3,
	BIGINT* i1, BIGINT* i2, BIGINT* i3,
	BIGINT N1, BIGINT N2, BIGINT N3,
	int ns, BIGINT size)
	// 3D interpolate complex values from du_padded (uniform grid data) array to data_nonuniform value,
	// using ns*ns*ns cube of real weights
	// in ker. out must be size 2 (real,imag), and du_padded
	// of size 2*N1*N2*N3 (alternating real,imag). i1 is the left-most index in
	// [0,N1), i2 the bottom index in [0,N2), i3 lowest in [0,N3).
	// Periodic wrapping in the du_padded array is applied, assuming N1,N2,N3>=ns.
	// dx,dy,dz indices into ker array, j index in complex du_padded array.
	// Barnett 6/16/17
{
	const int nsPadded = 4 * (1 + (ns - 1) / 4); // pad ns to mult of 4

	const BIGINT paddedN1 = N1 + 2 * MAX_NSPREAD;
	const BIGINT paddedN2 = N2 + 2 * MAX_NSPREAD;

	for (BIGINT i = 0; i < size; i++)
	{
		T* ker1 = kernel_vals1 + (i * nsPadded);
		T* ker2 = kernel_vals2 + (i * nsPadded);
		T* ker3 = kernel_vals3 + (i * nsPadded);

		T out[] = { 0.0, 0.0 };

		for (int dz = 0; dz < ns; dz++) {
			BIGINT oz = paddedN1 * paddedN2 * (i3[i] + dz);        // offset due to z
			for (int dy = 0; dy < ns; dy++) {
				BIGINT j = oz + paddedN1 * (i2[i] + dy) + i1[i];
				T ker23 = ker2[dy] * ker3[dz];
				for (int dx = 0; dx < ns; dx++) {
					T k = ker1[dx] * ker23;
					out[0] += du_padded[2 * j] * k;
					out[1] += du_padded[2 * j + 1] * k;
					++j;
				}
			}
		}

		// Copy result buffer to output array
		BIGINT si = sort_indices[i];
		data_nonuniform[2 * si] = out[0];
		data_nonuniform[2 * si + 1] = out[1];
	}
}

#ifdef __AVX2__
template<>
inline void interp_cube<double>(BIGINT* sort_indices, double* data_nonuniform, double* du_padded,
	double* kernel_vals1, double* kernel_vals2, double* kernel_vals3,
	BIGINT* i1, BIGINT* i2, BIGINT* i3,
	BIGINT N1, BIGINT N2, BIGINT N3,
	int ns, BIGINT size)
{
	const int nsPadded = 4 * (1 + (ns - 1) / 4); // pad ns to mult of 4

	const BIGINT paddedN1 = N1 + 2 * MAX_NSPREAD;
	const BIGINT paddedN2 = N2 + 2 * MAX_NSPREAD;

	double* pKer1 = kernel_vals1;
	double* pKer2 = kernel_vals2;
	double* pKer3 = kernel_vals3;

	switch (nsPadded) {
	case 4:
		for (BIGINT i = 0; i < size; i++)
		{
			__m256d _out0 = _mm256_setzero_pd();
			__m256d _out1 = _mm256_setzero_pd();

			for (int dz = 0; dz < ns; dz++) {
				BIGINT oz = paddedN1 * paddedN2 * (i3[i] + dz);        // offset due to z
				for (int dy = 0; dy < ns; dy++) {
					double* pDu = du_padded + 2 * (oz + paddedN1 * (i2[i] + dy) + i1[i]);

					__m256d _ker23 = _mm256_set1_pd(pKer2[dy] * pKer3[dz]);

					__m256d _k0 = _mm256_mul_pd(_ker23, _mm256_load_pd(pKer1 + 0));

					__m256d _kk0 = _mm256_permute4x64_pd(_k0, 0x50);
					__m256d _kk1 = _mm256_permute4x64_pd(_k0, 0xfa);

					__m256d _du0 = _mm256_loadu_pd(pDu + 0);
					__m256d _du1 = _mm256_loadu_pd(pDu + 4);

					_out0 = _mm256_fmadd_pd(_kk0, _du0, _out0);
					_out1 = _mm256_fmadd_pd(_kk1, _du1, _out1);
				}
			}

			_out0 = _mm256_add_pd(_out0, _out1);

			__m128d _out = _mm_add_pd(
				_mm256_castpd256_pd128(_out0),
				_mm256_extractf128_pd(_out0, 1));

			// Copy result buffer to output array
			BIGINT si0 = sort_indices[i];

			_mm_store_pd(data_nonuniform + 2 * si0, _out);

			pKer1 += nsPadded;
			pKer2 += nsPadded;
			pKer3 += nsPadded;
		}
		break;
	case 8:
		for (BIGINT i = 0; i < size; i++)
		{
			__m256d _out0 = _mm256_setzero_pd();
			__m256d _out1 = _mm256_setzero_pd();
			__m256d _out2 = _mm256_setzero_pd();
			__m256d _out3 = _mm256_setzero_pd();

			for (int dz = 0; dz < ns; dz++) {
				BIGINT oz = paddedN1 * paddedN2 * (i3[i] + dz);        // offset due to z
				for (int dy = 0; dy < ns; dy++) {
					double* pDu = du_padded + 2 * (oz + paddedN1 * (i2[i] + dy) + i1[i]);

					__m256d _ker23 = _mm256_set1_pd(pKer2[dy] * pKer3[dz]);

					__m256d _k0 = _mm256_mul_pd(_ker23, _mm256_load_pd(pKer1 + 0));
					__m256d _k2 = _mm256_mul_pd(_ker23, _mm256_load_pd(pKer1 + 4));

					__m256d _kk0 = _mm256_permute4x64_pd(_k0, 0x50);
					__m256d _kk1 = _mm256_permute4x64_pd(_k0, 0xfa);
					__m256d _kk2 = _mm256_permute4x64_pd(_k2, 0x50);
					__m256d _kk3 = _mm256_permute4x64_pd(_k2, 0xfa);

					__m256d _du0 = _mm256_loadu_pd(pDu + 0);
					__m256d _du1 = _mm256_loadu_pd(pDu + 4);
					__m256d _du2 = _mm256_loadu_pd(pDu + 8);
					__m256d _du3 = _mm256_loadu_pd(pDu + 12);

					_out0 = _mm256_fmadd_pd(_kk0, _du0, _out0);
					_out1 = _mm256_fmadd_pd(_kk1, _du1, _out1);
					_out2 = _mm256_fmadd_pd(_kk2, _du2, _out2);
					_out3 = _mm256_fmadd_pd(_kk3, _du3, _out3);
				}
			}

			_out0 = _mm256_add_pd(_out0, _out1);
			_out2 = _mm256_add_pd(_out2, _out3);
			_out0 = _mm256_add_pd(_out0, _out2);

			__m128d _out = _mm_add_pd(
				_mm256_castpd256_pd128(_out0),
				_mm256_extractf128_pd(_out0, 1));

			// Copy result buffer to output array
			BIGINT si0 = sort_indices[i];

			_mm_store_pd(data_nonuniform + 2 * si0, _out);

			pKer1 += nsPadded;
			pKer2 += nsPadded;
			pKer3 += nsPadded;
		}
		break;
	case 12:
		for (BIGINT i = 0; i < size; i++)
		{
			__m256d _out0 = _mm256_setzero_pd();
			__m256d _out1 = _mm256_setzero_pd();
			__m256d _out2 = _mm256_setzero_pd();
			__m256d _out3 = _mm256_setzero_pd();

			for (int dz = 0; dz < ns; dz++) {
				BIGINT oz = paddedN1 * paddedN2 * (i3[i] + dz);        // offset due to z
				for (int dy = 0; dy < ns; dy++) {
					double* pDu = du_padded + 2 * (oz + paddedN1 * (i2[i] + dy) + i1[i]);

					__m256d _ker23 = _mm256_set1_pd(pKer2[dy] * pKer3[dz]);

					{
						__m256d _k0 = _mm256_mul_pd(_ker23, _mm256_load_pd(pKer1 + 0));
						__m256d _k2 = _mm256_mul_pd(_ker23, _mm256_load_pd(pKer1 + 4));

						__m256d _kk0 = _mm256_permute4x64_pd(_k0, 0x50);
						__m256d _kk1 = _mm256_permute4x64_pd(_k0, 0xfa);
						__m256d _kk2 = _mm256_permute4x64_pd(_k2, 0x50);
						__m256d _kk3 = _mm256_permute4x64_pd(_k2, 0xfa);

						__m256d _du0 = _mm256_loadu_pd(pDu + 0);
						__m256d _du1 = _mm256_loadu_pd(pDu + 4);
						__m256d _du2 = _mm256_loadu_pd(pDu + 8);
						__m256d _du3 = _mm256_loadu_pd(pDu + 12);

						_out0 = _mm256_fmadd_pd(_kk0, _du0, _out0);
						_out1 = _mm256_fmadd_pd(_kk1, _du1, _out1);
						_out2 = _mm256_fmadd_pd(_kk2, _du2, _out2);
						_out3 = _mm256_fmadd_pd(_kk3, _du3, _out3);
					}

					{
						__m256d _k4 = _mm256_mul_pd(_ker23, _mm256_load_pd(pKer1 + 8));

						__m256d _kk4 = _mm256_permute4x64_pd(_k4, 0x50);
						__m256d _kk5 = _mm256_permute4x64_pd(_k4, 0xfa);

						__m256d _du4 = _mm256_loadu_pd(pDu + 16);
						__m256d _du5 = _mm256_loadu_pd(pDu + 20);

						_out0 = _mm256_fmadd_pd(_kk4, _du4, _out0);
						_out1 = _mm256_fmadd_pd(_kk5, _du5, _out1);
					}
				}
			}

			_out2 = _mm256_add_pd(_out2, _out3);
			_out0 = _mm256_add_pd(_out0, _out1);
			_out0 = _mm256_add_pd(_out0, _out2);

			__m128d _out = _mm_add_pd(
				_mm256_castpd256_pd128(_out0),
				_mm256_extractf128_pd(_out0, 1));

			// Copy result buffer to output array
			BIGINT si0 = sort_indices[i];

			_mm_store_pd(data_nonuniform + 2 * si0, _out);

			pKer1 += nsPadded;
			pKer2 += nsPadded;
			pKer3 += nsPadded;
		}
		break;
	case 16:
		for (BIGINT i = 0; i < size; i++)
		{
			__m256d _out0 = _mm256_setzero_pd();
			__m256d _out1 = _mm256_setzero_pd();
			__m256d _out2 = _mm256_setzero_pd();
			__m256d _out3 = _mm256_setzero_pd();

			for (int dz = 0; dz < ns; dz++) {
				BIGINT oz = paddedN1 * paddedN2 * (i3[i] + dz);        // offset due to z
				for (int dy = 0; dy < ns; dy++) {
					double* pDu = du_padded + 2 * (oz + paddedN1 * (i2[i] + dy) + i1[i]);

					__m256d _ker23 = _mm256_set1_pd(pKer2[dy] * pKer3[dz]);

					{
						__m256d _k0 = _mm256_mul_pd(_ker23, _mm256_load_pd(pKer1 + 0));
						__m256d _k2 = _mm256_mul_pd(_ker23, _mm256_load_pd(pKer1 + 4));

						__m256d _kk0 = _mm256_permute4x64_pd(_k0, 0x50);
						__m256d _kk1 = _mm256_permute4x64_pd(_k0, 0xfa);
						__m256d _kk2 = _mm256_permute4x64_pd(_k2, 0x50);
						__m256d _kk3 = _mm256_permute4x64_pd(_k2, 0xfa);

						__m256d _du0 = _mm256_loadu_pd(pDu + 0);
						__m256d _du1 = _mm256_loadu_pd(pDu + 4);
						__m256d _du2 = _mm256_loadu_pd(pDu + 8);
						__m256d _du3 = _mm256_loadu_pd(pDu + 12);

						_out0 = _mm256_fmadd_pd(_kk0, _du0, _out0);
						_out1 = _mm256_fmadd_pd(_kk1, _du1, _out1);
						_out2 = _mm256_fmadd_pd(_kk2, _du2, _out2);
						_out3 = _mm256_fmadd_pd(_kk3, _du3, _out3);
					}

					{
						__m256d _k4 = _mm256_mul_pd(_ker23, _mm256_load_pd(pKer1 + 8));
						__m256d _k6 = _mm256_mul_pd(_ker23, _mm256_load_pd(pKer1 + 12));

						__m256d _kk4 = _mm256_permute4x64_pd(_k4, 0x50);
						__m256d _kk5 = _mm256_permute4x64_pd(_k4, 0xfa);
						__m256d _kk6 = _mm256_permute4x64_pd(_k6, 0x50);
						__m256d _kk7 = _mm256_permute4x64_pd(_k6, 0xfa);

						__m256d _du4 = _mm256_loadu_pd(pDu + 16);
						__m256d _du5 = _mm256_loadu_pd(pDu + 20);
						__m256d _du6 = _mm256_loadu_pd(pDu + 24);
						__m256d _du7 = _mm256_loadu_pd(pDu + 28);

						_out0 = _mm256_fmadd_pd(_kk4, _du4, _out0);
						_out1 = _mm256_fmadd_pd(_kk5, _du5, _out1);
						_out2 = _mm256_fmadd_pd(_kk6, _du6, _out2);
						_out3 = _mm256_fmadd_pd(_kk7, _du7, _out3);
					}
				}
			}

			_out0 = _mm256_add_pd(_out0, _out1);
			_out2 = _mm256_add_pd(_out2, _out3);
			_out0 = _mm256_add_pd(_out0, _out2);

			__m128d _out = _mm_add_pd(
				_mm256_castpd256_pd128(_out0),
				_mm256_extractf128_pd(_out0, 1));

			// Copy result buffer to output array
			BIGINT si0 = sort_indices[i];

			_mm_store_pd(data_nonuniform + 2 * si0, _out);

			pKer1 += nsPadded;
			pKer2 += nsPadded;
			pKer3 += nsPadded;
		}
		break;
	default: // Should never get here
		break;
	}
}

template<>
inline void interp_cube<float>(BIGINT* sort_indices, float* data_nonuniform, float* du_padded,
	float* kernel_vals1, float* kernel_vals2, float* kernel_vals3,
	BIGINT* i1, BIGINT* i2, BIGINT* i3,
	BIGINT N1, BIGINT N2, BIGINT N3,
	int ns, BIGINT size)
{
	const int nsPadded = 4 * (1 + (ns - 1) / 4); // pad ns to mult of 4

	const BIGINT paddedN1 = N1 + 2 * MAX_NSPREAD;
	const BIGINT paddedN2 = N2 + 2 * MAX_NSPREAD;

	float* pKer1 = kernel_vals1;
	float* pKer2 = kernel_vals2;
	float* pKer3 = kernel_vals3;
	
	__m256i _mask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, -1, -1);
	__m256i _spreadlo = _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0);
	__m256i _spreadhi = _mm256_set_epi32(7, 7, 6, 6, 5, 5, 4, 4);

	switch (nsPadded) {
	case 4:
		for (BIGINT i = 0; i < size; i++)
		{
			__m256 _out0 = _mm256_setzero_ps();

			for (int dz = 0; dz < ns; dz++) {
				BIGINT oz = paddedN1 * paddedN2 * (i3[i] + dz);        // offset due to z
				for (int dy = 0; dy < ns; dy++) {
					float* pDu = du_padded + 2 * (oz + paddedN1 * (i2[i] + dy) + i1[i]);

					__m256 _ker23 = _mm256_set1_ps(pKer2[dy] * pKer3[dz]);

					__m256 _k0 = _mm256_mul_ps(_ker23, _mm256_castps128_ps256(_mm_load_ps(pKer1)));

					__m256 _kk0 = _mm256_permutevar8x32_ps(_k0, _spreadlo);
					__m256 _du0 = _mm256_loadu_ps(pDu + 0);
					_out0 = _mm256_fmadd_ps(_kk0, _du0, _out0);
				}
			}

			_out0 = _mm256_add_ps(_out0,
				_mm256_permute2f128_ps(_out0, _out0, 0x01));

			_out0 = _mm256_add_ps(_out0,
				_mm256_permute_ps(_out0, 0x8e));

			// Copy result buffer to output array
			BIGINT si0 = sort_indices[i];
			_mm256_maskstore_ps(data_nonuniform + 2 * si0, _mask, _out0);

			pKer1 += nsPadded;
			pKer2 += nsPadded;
			pKer3 += nsPadded;
		}
		break;
	case 8:
		for (BIGINT i = 0; i < size; i++)
		{
			__m256 _out0 = _mm256_setzero_ps();

			for (int dz = 0; dz < ns; dz++) {
				BIGINT oz = paddedN1 * paddedN2 * (i3[i] + dz);        // offset due to z
				for (int dy = 0; dy < ns; dy++) {
					float* pDu = du_padded + 2 * (oz + paddedN1 * (i2[i] + dy) + i1[i]);

					__m256 _ker23 = _mm256_set1_ps(pKer2[dy] * pKer3[dz]);

					__m256 _k0 = _mm256_mul_ps(_ker23, _mm256_loadu_ps(pKer1 + 0));

					__m256 _kk0 = _mm256_permutevar8x32_ps(_k0, _spreadlo);
					__m256 _du0 = _mm256_loadu_ps(pDu + 0);
					_out0 = _mm256_fmadd_ps(_kk0, _du0, _out0);

					__m256 _kk1 = _mm256_permutevar8x32_ps(_k0, _spreadhi);
					__m256 _du1 = _mm256_loadu_ps(pDu + 8);
					_out0 = _mm256_fmadd_ps(_kk1, _du1, _out0);
				}
			}

			_out0 = _mm256_add_ps(_out0,
				_mm256_permute2f128_ps(_out0, _out0, 0x01));

			_out0 = _mm256_add_ps(_out0,
				_mm256_permute_ps(_out0, 0x8e));

			// Copy result buffer to output array
			BIGINT si0 = sort_indices[i];
			_mm256_maskstore_ps(data_nonuniform + 2 * si0, _mask, _out0);

			pKer1 += nsPadded;
			pKer2 += nsPadded;
			pKer3 += nsPadded;
		}
		break;
	case 12:
		for (BIGINT i = 0; i < size; i++)
		{
			__m256 _out0 = _mm256_setzero_ps();

			for (int dz = 0; dz < ns; dz++) {
				BIGINT oz = paddedN1 * paddedN2 * (i3[i] + dz);        // offset due to z
				for (int dy = 0; dy < ns; dy++) {
					float* pDu = du_padded + 2 * (oz + paddedN1 * (i2[i] + dy) + i1[i]);

					__m256 _ker23 = _mm256_set1_ps(pKer2[dy] * pKer3[dz]);

					__m256 _k0 = _mm256_mul_ps(_ker23, _mm256_loadu_ps(pKer1 + 0));

					__m256 _kk0 = _mm256_permutevar8x32_ps(_k0, _spreadlo);
					__m256 _du0 = _mm256_loadu_ps(pDu + 0);
					_out0 = _mm256_fmadd_ps(_kk0, _du0, _out0);

					__m256 _kk1 = _mm256_permutevar8x32_ps(_k0, _spreadhi);
					__m256 _du1 = _mm256_loadu_ps(pDu + 8);
					_out0 = _mm256_fmadd_ps(_kk1, _du1, _out0);

					__m256 _k2 = _mm256_mul_ps(_ker23, _mm256_castps128_ps256(_mm_load_ps(pKer1 + 8)));

					__m256 _kk2 = _mm256_permutevar8x32_ps(_k2, _spreadlo);
					__m256 _du2 = _mm256_loadu_ps(pDu + 16);
					_out0 = _mm256_fmadd_ps(_kk2, _du2, _out0);
				}
			}

			_out0 = _mm256_add_ps(_out0,
				_mm256_permute2f128_ps(_out0, _out0, 0x01));

			_out0 = _mm256_add_ps(_out0,
				_mm256_permute_ps(_out0, 0x8e));

			// Copy result buffer to output array
			BIGINT si0 = sort_indices[i];
			_mm256_maskstore_ps(data_nonuniform + 2 * si0, _mask, _out0);

			pKer1 += nsPadded;
			pKer2 += nsPadded;
			pKer3 += nsPadded;
		}
		break;
	case 16:
		for (BIGINT i = 0; i < size; i++)
		{
			__m256 _out0 = _mm256_setzero_ps();

			for (int dz = 0; dz < ns; dz++) {
				BIGINT oz = paddedN1 * paddedN2 * (i3[i] + dz);        // offset due to z
				for (int dy = 0; dy < ns; dy++) {
					float* pDu = du_padded + 2 * (oz + paddedN1 * (i2[i] + dy) + i1[i]);

					__m256 _ker23 = _mm256_set1_ps(pKer2[dy] * pKer3[dz]);

					__m256 _k0 = _mm256_mul_ps(_ker23, _mm256_loadu_ps(pKer1 + 0));

					__m256 _kk0 = _mm256_permutevar8x32_ps(_k0, _spreadlo);
					__m256 _du0 = _mm256_loadu_ps(pDu + 0);
					_out0 = _mm256_fmadd_ps(_kk0, _du0, _out0);

					__m256 _kk1 = _mm256_permutevar8x32_ps(_k0, _spreadhi);
					__m256 _du1 = _mm256_loadu_ps(pDu + 8);
					_out0 = _mm256_fmadd_ps(_kk1, _du1, _out0);

					__m256 _k2 = _mm256_mul_ps(_ker23, _mm256_loadu_ps(pKer1 + 8));

					__m256 _kk2 = _mm256_permutevar8x32_ps(_k2, _spreadlo);
					__m256 _du2 = _mm256_loadu_ps(pDu + 16);
					_out0 = _mm256_fmadd_ps(_kk2, _du2, _out0);

					__m256 _kk3 = _mm256_permutevar8x32_ps(_k2, _spreadhi);
					__m256 _du3 = _mm256_loadu_ps(pDu + 24);
					_out0 = _mm256_fmadd_ps(_kk3, _du3, _out0);
				}
			}

			_out0 = _mm256_add_ps(_out0,
				_mm256_permute2f128_ps(_out0, _out0, 0x01));

			_out0 = _mm256_add_ps(_out0,
				_mm256_permute_ps(_out0, 0x8e));

			// Copy result buffer to output array
			BIGINT si0 = sort_indices[i];
			_mm256_maskstore_ps(data_nonuniform + 2 * si0, _mask, _out0);

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