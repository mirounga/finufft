#ifndef INTERP_H
#define INTERP_H

template<class T>
void interp_line(BIGINT* sort_indices, T* data_nonuniform, T* du_padded,
	T* kernel_vals1,
	BIGINT* i1,
	BIGINT N1,
	int ns, BIGINT begin, BIGINT end)
	// 1D interpolate complex values from du_padded array to data_nonuniform, using real weights
	// ker[0] through ker[ns-1]. out must be size 2 (real,imag), and du_padded
	// of size 2*N1 (alternating real,imag). i1 is the left-most index in [0,N1)
	// Periodic wrapping in the du_padded array is applied, assuming N1>=ns.
	// dx is index into ker array, j index in complex du_padded (data_uniform) array.
	// Barnett 6/15/17
{
	const int nsPadded = 4 * (1 + (ns - 1) / 4); // pad ns to mult of 4

	// main loop over NU targs, interp each from U
	for (BIGINT i = begin; i < end; i++)
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
#include <immintrin.h>

template<>
inline void interp_line<double>(BIGINT* sort_indices, double* data_nonuniform, double* du_padded,
	double* kernel_vals1,
	BIGINT* i1,
	BIGINT N1,
	int ns, BIGINT begin, BIGINT end)
{
	const int nsPadded = 4 * (1 + (ns - 1) / 4); // pad ns to mult of 4

	// main loop over NU targs, interp each from U
	double* pKer = kernel_vals1 + begin * nsPadded;

	if (nsPadded == 8) {
		for (BIGINT i = begin; i < end; i++)
		{
			double* pDu = du_padded + 2 * i1[i];

			__m256d _k0 = _mm256_load_pd(pKer + 0);
			__m256d _k2 = _mm256_load_pd(pKer + 4);

			__m256d _kk0 = _mm256_permute4x64_pd(_k0, 0x50);
			__m256d _kk1 = _mm256_permute4x64_pd(_k0, 0xfa);
			__m256d _kk2 = _mm256_permute4x64_pd(_k2, 0x50);
			__m256d _kk3 = _mm256_permute4x64_pd(_k2, 0xfa);

			__m256d _du0 = _mm256_load_pd(pDu + 0);
			__m256d _du1 = _mm256_load_pd(pDu + 4);
			__m256d _du2 = _mm256_load_pd(pDu + 8);
			__m256d _du3 = _mm256_load_pd(pDu + 12);

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
	}
	else {
		for (BIGINT i = begin; i < end; i++)
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
	}
}

template<>
inline void interp_line<float>(BIGINT* sort_indices, float* data_nonuniform, float* du_padded,
	float* kernel_vals1,
	BIGINT* i1,
	BIGINT N1,
	int ns, BIGINT begin, BIGINT end)
{
	const int nsPadded = 4 * (1 + (ns - 1) / 4); // pad ns to mult of 4

	// main loop over NU targs, interp each from U
	float* pKer = kernel_vals1 + begin * nsPadded;

	__m256i _mask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, -1, -1);
	__m256i _spreadlo = _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0);
	__m256i _spreadhi = _mm256_set_epi32(7, 7, 6, 6, 5, 5, 4, 4);

	BIGINT end2 = begin + (end - begin) & ~0x01;

	switch (nsPadded) {
	case 8:
		// Unrolled loop
		for (BIGINT i = begin; i < end2; i += 2)
		{
			float* pDu0 = du_padded + 2 * i1[i + 0];
			float* pDu1 = du_padded + 2 * i1[i + 1];

			__m256 _k0 = _mm256_load_ps(pKer + 0);

			__m256 _kk0 = _mm256_permutevar8x32_ps(_k0, _spreadlo);
			__m256 _du0 = _mm256_load_ps(pDu0 + 0);
			__m256 _out0 = _mm256_mul_ps(_kk0, _du0);

			__m256 _k1 = _mm256_load_ps(pKer + 8);

			__m256 _kk1 = _mm256_permutevar8x32_ps(_k1, _spreadlo);
			__m256 _du1 = _mm256_load_ps(pDu1 + 0);
			__m256 _out1 = _mm256_mul_ps(_kk1, _du1);

			_kk0 = _mm256_permutevar8x32_ps(_k0, _spreadhi);
			_du0 = _mm256_load_ps(pDu0 + 8);
			_out0 = _mm256_fmadd_ps(_kk0, _du0, _out0);

			_kk1 = _mm256_permutevar8x32_ps(_k1, _spreadhi);
			_du1 = _mm256_load_ps(pDu1 + 8);
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
		for (BIGINT i = end2; i < end; i++)
		{
			float* pDu = du_padded + 2 * i1[i];

			__m256 _k0 = _mm256_load_ps(pKer + 0);

			__m256 _kk0 = _mm256_permutevar8x32_ps(_k0, _spreadlo);
			__m256 _du0 = _mm256_load_ps(pDu + 0);
			__m256 _out0 = _mm256_mul_ps(_kk0, _du0);

			_kk0 = _mm256_permutevar8x32_ps(_k0, _spreadhi);
			_du0 = _mm256_load_ps(pDu + 8);
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
		for (BIGINT i = begin; i < end; i++)
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
	int ns, BIGINT begin, BIGINT end)
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

	for (BIGINT i = begin; i < end; i++)
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
template<>
inline void interp_square<double>(BIGINT* sort_indices, double* data_nonuniform, double* du_padded,
	double* kernel_vals1, double* kernel_vals2,
	BIGINT* i1, BIGINT* i2,
	BIGINT N1, BIGINT N2,
	int ns, BIGINT begin, BIGINT end)
{
	const int nsPadded = 4 * (1 + (ns - 1) / 4); // pad ns to mult of 4

	const BIGINT paddedN1 = N1 + 2 * MAX_NSPREAD;

	double* pKer1 = kernel_vals1 + begin * nsPadded;
	double* pKer2 = kernel_vals2 + begin * nsPadded;

	for (BIGINT i = begin; i < end; i++)
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
}
template<>
inline void interp_square<float>(BIGINT* sort_indices, float* data_nonuniform, float* du_padded,
	float* kernel_vals1, float* kernel_vals2,
	BIGINT* i1, BIGINT* i2,
	BIGINT N1, BIGINT N2,
	int ns, BIGINT begin, BIGINT end)
{
	const int nsPadded = 4 * (1 + (ns - 1) / 4); // pad ns to mult of 4

	const BIGINT paddedN1 = N1 + 2 * MAX_NSPREAD;

	float* pKer1 = kernel_vals1 + begin * nsPadded;
	float* pKer2 = kernel_vals2 + begin * nsPadded;

	__m256i _mask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, -1, -1);
	__m256i _spreadlo = _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0);
	__m256i _spreadhi = _mm256_set_epi32(7, 7, 6, 6, 5, 5, 4, 4);

	for (BIGINT i = begin; i < end; i++)
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
}
#endif

template<class T>
void interp_cube(BIGINT* sort_indices, T* data_nonuniform, T* du_padded,
	T* kernel_vals1, T* kernel_vals2, T* kernel_vals3,
	BIGINT* i1, BIGINT* i2, BIGINT* i3,
	BIGINT N1, BIGINT N2, BIGINT N3,
	int ns, BIGINT begin, BIGINT end)
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

	for (BIGINT i = begin; i < end; i++)
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
	int ns, BIGINT begin, BIGINT end)
{
	const int nsPadded = 4 * (1 + (ns - 1) / 4); // pad ns to mult of 4

	const BIGINT paddedN1 = N1 + 2 * MAX_NSPREAD;
	const BIGINT paddedN2 = N2 + 2 * MAX_NSPREAD;

	double* pKer1 = kernel_vals1 + begin * nsPadded;
	double* pKer2 = kernel_vals2 + begin * nsPadded;
	double* pKer3 = kernel_vals3 + begin * nsPadded;

	switch (nsPadded) {
	case 8:
		for (BIGINT i = begin; i < end; i++)
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

					__m256d _du0 = _mm256_load_pd(pDu + 0);
					__m256d _du1 = _mm256_load_pd(pDu + 4);
					__m256d _du2 = _mm256_load_pd(pDu + 8);
					__m256d _du3 = _mm256_load_pd(pDu + 12);

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
		for (BIGINT i = begin; i < end; i++)
		{
			__m256d _out0 = _mm256_setzero_pd();
			__m256d _out1 = _mm256_setzero_pd();
			__m256d _out2 = _mm256_setzero_pd();
			__m256d _out3 = _mm256_setzero_pd();
			__m256d _out4 = _mm256_setzero_pd();
			__m256d _out5 = _mm256_setzero_pd();

			for (int dz = 0; dz < ns; dz++) {
				BIGINT oz = paddedN1 * paddedN2 * (i3[i] + dz);        // offset due to z
				for (int dy = 0; dy < ns; dy++) {
					double* pDu = du_padded + 2 * (oz + paddedN1 * (i2[i] + dy) + i1[i]);

					__m256d _ker23 = _mm256_set1_pd(pKer2[dy] * pKer3[dz]);

					__m256d _k0 = _mm256_mul_pd(_ker23, _mm256_load_pd(pKer1 + 0));
					__m256d _k2 = _mm256_mul_pd(_ker23, _mm256_load_pd(pKer1 + 4));
					__m256d _k4 = _mm256_mul_pd(_ker23, _mm256_load_pd(pKer1 + 8));

					__m256d _kk0 = _mm256_permute4x64_pd(_k0, 0x50);
					__m256d _kk1 = _mm256_permute4x64_pd(_k0, 0xfa);
					__m256d _kk2 = _mm256_permute4x64_pd(_k2, 0x50);
					__m256d _kk3 = _mm256_permute4x64_pd(_k2, 0xfa);
					__m256d _kk4 = _mm256_permute4x64_pd(_k4, 0x50);
					__m256d _kk5 = _mm256_permute4x64_pd(_k4, 0xfa);

					__m256d _du0 = _mm256_load_pd(pDu + 0);
					__m256d _du1 = _mm256_load_pd(pDu + 4);
					__m256d _du2 = _mm256_load_pd(pDu + 8);
					__m256d _du3 = _mm256_load_pd(pDu + 12);
					__m256d _du4 = _mm256_load_pd(pDu + 16);
					__m256d _du5 = _mm256_load_pd(pDu + 20);

					_out0 = _mm256_fmadd_pd(_kk0, _du0, _out0);
					_out1 = _mm256_fmadd_pd(_kk1, _du1, _out1);
					_out2 = _mm256_fmadd_pd(_kk2, _du2, _out2);
					_out3 = _mm256_fmadd_pd(_kk3, _du3, _out3);
					_out4 = _mm256_fmadd_pd(_kk4, _du4, _out4);
					_out5 = _mm256_fmadd_pd(_kk5, _du5, _out5);
				}
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
			pKer3 += nsPadded;
		}
		break;
	case 16:
		for (BIGINT i = begin; i < end; i++)
		{
			__m128d _out = _mm_setzero_pd();

			for (int dz = 0; dz < ns; dz++) {
				BIGINT oz = paddedN1 * paddedN2 * (i3[i] + dz);        // offset due to z
				for (int dy = 0; dy < ns; dy++) {
					double* pDu = du_padded + 2 * (oz + paddedN1 * (i2[i] + dy) + i1[i]);

					double ker23 = pKer2[dy] * pKer3[dz];
					for (int dx = 0; dx < ns; dx++) {
						__m128d _kk = _mm_set1_pd(pKer1[dx] * ker23);

						__m128d _du = _mm_load_pd(pDu);

						_out = _mm_fmadd_pd(_kk, _du, _out);

						pDu += 2;
					}
				}
			}

			// Copy result buffer to output array
			BIGINT si0 = sort_indices[i];

			_mm_store_pd(data_nonuniform + 2 * si0, _out);

			pKer1 += nsPadded;
			pKer2 += nsPadded;
			pKer3 += nsPadded;
		}
		break;
	default:
		for (BIGINT i = begin; i < end; i++)
		{
			__m128d _out = _mm_setzero_pd();

			for (int dz = 0; dz < ns; dz++) {
				BIGINT oz = paddedN1 * paddedN2 * (i3[i] + dz);        // offset due to z
				for (int dy = 0; dy < ns; dy++) {
					double* pDu = du_padded + 2 * (oz + paddedN1 * (i2[i] + dy) + i1[i]);

					double ker23 = pKer2[dy] * pKer3[dz];
					for (int dx = 0; dx < ns; dx++) {
						__m128d _kk = _mm_set1_pd(pKer1[dx] * ker23);

						__m128d _du = _mm_load_pd(pDu);

						_out = _mm_fmadd_pd(_kk, _du, _out);

						pDu += 2;
					}
				}
			}

			// Copy result buffer to output array
			BIGINT si0 = sort_indices[i];

			_mm_store_pd(data_nonuniform + 2 * si0, _out);

			pKer1 += nsPadded;
			pKer2 += nsPadded;
			pKer3 += nsPadded;
		}
		break;
	}
}

template<>
inline void interp_cube<float>(BIGINT* sort_indices, float* data_nonuniform, float* du_padded,
	float* kernel_vals1, float* kernel_vals2, float* kernel_vals3,
	BIGINT* i1, BIGINT* i2, BIGINT* i3,
	BIGINT N1, BIGINT N2, BIGINT N3,
	int ns, BIGINT begin, BIGINT end)
{
	const int nsPadded = 4 * (1 + (ns - 1) / 4); // pad ns to mult of 4

	const BIGINT paddedN1 = N1 + 2 * MAX_NSPREAD;
	const BIGINT paddedN2 = N2 + 2 * MAX_NSPREAD;

	float* pKer1 = kernel_vals1 + begin * nsPadded;
	float* pKer2 = kernel_vals2 + begin * nsPadded;
	float* pKer3 = kernel_vals3 + begin * nsPadded;

	__m256i _mask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, -1, -1);
	__m256i _spreadlo = _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0);
	__m256i _spreadhi = _mm256_set_epi32(7, 7, 6, 6, 5, 5, 4, 4);

	BIGINT end2 = begin + (end - begin) & ~0x01;

	switch (nsPadded) {
	case 12:
		// Unrolled loop
		for (BIGINT i = begin; i < end2; i += 2)
		{
			__m256 _out0 = _mm256_setzero_ps();
			__m256 _out1 = _mm256_setzero_ps();

			for (int dz = 0; dz < ns; dz++) {
				BIGINT oz0 = paddedN1 * paddedN2 * (i3[i + 0] + dz);        // offset due to z
				BIGINT oz1 = paddedN1 * paddedN2 * (i3[i + 1] + dz);
				for (int dy = 0; dy < ns; dy++) {
					float* pDu0 = du_padded + 2 * (oz0 + paddedN1 * (i2[i + 0] + dy) + i1[i + 0]);
					float* pDu1 = du_padded + 2 * (oz1 + paddedN1 * (i2[i + 1] + dy) + i1[i + 1]);

					__m256 _ker23a = _mm256_set1_ps(pKer2[dy] * pKer3[dz]);
					__m256 _k0 = _mm256_mul_ps(_ker23a, _mm256_load_ps(pKer1 + 0));

					__m256 _kk0 = _mm256_permutevar8x32_ps(_k0, _spreadlo);
					__m256 _du0 = _mm256_load_ps(pDu0 + 0);
					_out0 = _mm256_fmadd_ps(_kk0, _du0, _out0);

					__m256 _ker23b = _mm256_set1_ps(pKer2[dy + 12] * pKer3[dz + 12]);
					__m256 _k1 = _mm256_mul_ps(_ker23a, _mm256_loadu_ps(pKer1 + 12));

					__m256 _kk1 = _mm256_permutevar8x32_ps(_k1, _spreadlo);
					__m256 _du1 = _mm256_load_ps(pDu1 + 0);
					_out1 = _mm256_fmadd_ps(_kk1, _du1, _out1);

					_kk0 = _mm256_permutevar8x32_ps(_k0, _spreadhi);
					_du0 = _mm256_load_ps(pDu0 + 8);
					_out0 = _mm256_fmadd_ps(_kk0, _du0, _out0);

					_kk1 = _mm256_permutevar8x32_ps(_k0, _spreadhi);
					_du1 = _mm256_load_ps(pDu1 + 8);
					_out1 = _mm256_fmadd_ps(_kk1, _du1, _out1);

					_k0 = _mm256_mul_ps(_ker23a, _mm256_castps128_ps256(_mm_load_ps(pKer1 + 8)));
					_k1 = _mm256_mul_ps(_ker23a, _mm256_castps128_ps256(_mm_load_ps(pKer1 + 20)));

					_kk0 = _mm256_permutevar8x32_ps(_k0, _spreadlo);
					_du0 = _mm256_load_ps(pDu0 + 16);
					_out0 = _mm256_fmadd_ps(_kk0, _du0, _out0);

					_kk1 = _mm256_permutevar8x32_ps(_k1, _spreadlo);
					_du1 = _mm256_load_ps(pDu1 + 40);
					_out1 = _mm256_fmadd_ps(_kk1, _du1, _out1);
				}
			}

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

			pKer1 += 24;
			pKer2 += 24;
			pKer3 += 24;
		}
		// Short tail
		for (BIGINT i = end2; i < end; i++)
		{
			__m256 _out0 = _mm256_setzero_ps();

			for (int dz = 0; dz < ns; dz++) {
				BIGINT oz = paddedN1 * paddedN2 * (i3[i] + dz);        // offset due to z
				for (int dy = 0; dy < ns; dy++) {
					float* pDu0 = du_padded + 2 * (oz + paddedN1 * (i2[i] + dy) + i1[i]);

					__m256 _ker23a = _mm256_set1_ps(pKer2[dy] * pKer3[dz]);

					__m256 _k0 = _mm256_mul_ps(_ker23a, _mm256_load_ps(pKer1 + 0));

					__m256 _kk0 = _mm256_permutevar8x32_ps(_k0, _spreadlo);
					__m256 _du0 = _mm256_load_ps(pDu0 + 0);
					_out0 = _mm256_fmadd_ps(_kk0, _du0, _out0);

					_kk0 = _mm256_permutevar8x32_ps(_k0, _spreadhi);
					_du0 = _mm256_load_ps(pDu0 + 8);
					_out0 = _mm256_fmadd_ps(_kk0, _du0, _out0);

					_k0 = _mm256_mul_ps(_ker23a, _mm256_castps128_ps256(_mm_load_ps(pKer1 + 8)));

					_kk0 = _mm256_permutevar8x32_ps(_k0, _spreadlo);
					_du0 = _mm256_load_ps(pDu0 + 16);
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

			pKer1 += 12;
			pKer2 += 12;
			pKer3 += 12;
		}
		break;
	default:
		for (BIGINT i = begin; i < end; i++)
		{
			__m256 _out0 = _mm256_setzero_ps();

			for (int dz = 0; dz < ns; dz++) {
				BIGINT oz = paddedN1 * paddedN2 * (i3[i] + dz);        // offset due to z
				for (int dy = 0; dy < ns; dy++) {
					float* pDu = du_padded + 2 * (oz + paddedN1 * (i2[i] + dy) + i1[i]);

					float ker23 = pKer2[dy] * pKer3[dz];
					for (int dx = 0; dx < ns; dx++) {
						__m256 _kk = _mm256_set1_ps(pKer1[dx] * ker23);

						__m256 _du = _mm256_maskload_ps(pDu, _mask);

						_out0 = _mm256_fmadd_ps(_kk, _du, _out0);

						pDu += 2;
					}
				}
			}

			// Copy result buffer to output array
			BIGINT si0 = sort_indices[i];

			_mm256_maskstore_ps(data_nonuniform + 2 * si0, _mask, _out0);

			pKer1 += nsPadded;
			pKer2 += nsPadded;
			pKer3 += nsPadded;
		}
		break;
	}
}
#endif

#endif