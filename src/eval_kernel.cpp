#include <cassert>

#include <defs.h>
#include <spreadinterp.h>

#ifdef __AVX2__
#include <immintrin.h>

template<int ns, int np>
void eval_kernel_bulk_generic(float* c, float* kernel_vals, float* x1, const BIGINT size)
{
	// pad ns to mult of 4
	int nsPadded = 4 * (1 + (ns - 1) / 4);

	if (nsPadded == 8)
	{
		__m256 _two = _mm256_set1_ps(2.0f);
		__m256 _ns_m_1 = _mm256_set1_ps(ns - 1.0f);

		BIGINT size4 = size & ~0x3ll;

		float* ker = kernel_vals;

		// main loop
		for (BIGINT i = 0; i < size4; i += 4)
		{
			__m256 _x_a = _mm256_set1_ps(x1[i + 0]);
			__m256 _x_b = _mm256_set1_ps(x1[i + 1]);
			__m256 _x_c = _mm256_set1_ps(x1[i + 2]);
			__m256 _x_d = _mm256_set1_ps(x1[i + 3]);

			// scale so local grid offset z in [-1,1]
			__m256 _z_a = _mm256_fmadd_ps(_x_a, _two, _ns_m_1);
			__m256 _z_b = _mm256_fmadd_ps(_x_b, _two, _ns_m_1);
			__m256 _z_c = _mm256_fmadd_ps(_x_c, _two, _ns_m_1);
			__m256 _z_d = _mm256_fmadd_ps(_x_d, _two, _ns_m_1);

			int j = np - 1;

			float* pCoeff = c + 8 * j;

			__m256 _k_a = _mm256_load_ps(pCoeff);
			__m256 _k_b = _k_a, _k_c = _k_a, _k_d = _k_a;

			for (; j-- > 0;)
			{
				pCoeff -= 8;

				__m256 _c = _mm256_load_ps(pCoeff);

				_k_a = _mm256_fmadd_ps(_k_a, _z_a, _c);
				_k_b = _mm256_fmadd_ps(_k_b, _z_b, _c);
				_k_c = _mm256_fmadd_ps(_k_c, _z_c, _c);
				_k_d = _mm256_fmadd_ps(_k_d, _z_d, _c);
			}

			_mm256_store_ps(ker + 0, _k_a);
			_mm256_store_ps(ker + 8, _k_b);
			_mm256_store_ps(ker + 16, _k_c);
			_mm256_store_ps(ker + 24, _k_d);
			ker += 32;
		}

		// short tail
		for (BIGINT i = size4; i < size; i++)
		{
			__m256 _x_a = _mm256_set1_ps(x1[i]);
			// scale so local grid offset z in [-1,1]
			__m256 _z_a = _mm256_fmadd_ps(_x_a, _two, _ns_m_1);

			int j = np - 1;

			float* pCoeff = c + 8 * j;

			__m256 _k_a = _mm256_load_ps(pCoeff);

			for (; j-- > 0;)
			{
				pCoeff -= 8;

				__m256 _c = _mm256_load_ps(pCoeff);

				_k_a = _mm256_fmadd_ps(_k_a, _z_a, _c);
			}

			_mm256_store_ps(ker, _k_a);
			ker += 8;
		}
	}
	else if (nsPadded == 12)
	{
		__m256 _two = _mm256_set1_ps(2.0f);
		__m256 _ns_m_1 = _mm256_set1_ps(ns - 1.0f);

		BIGINT size4 = size & ~0x3ll;

		float* ker = kernel_vals;

		// main loop
		for (BIGINT i = 0; i < size4; i += 4)
		{
			__m256 _x_a = _mm256_set1_ps(x1[i + 0]);
			__m256 _x_b = _mm256_set1_ps(x1[i + 1]);
			__m256 _x_c = _mm256_set1_ps(x1[i + 2]);
			__m256 _x_d = _mm256_set1_ps(x1[i + 3]);

			// scale so local grid offset z in [-1,1]
			__m256 _z_a = _mm256_fmadd_ps(_x_a, _two, _ns_m_1);
			__m256 _z_b = _mm256_fmadd_ps(_x_b, _two, _ns_m_1);
			__m256 _z_c = _mm256_fmadd_ps(_x_c, _two, _ns_m_1);
			__m256 _z_d = _mm256_fmadd_ps(_x_d, _two, _ns_m_1);

			int j = np - 1;

			float* pCoeff = c + 12 * j;

			__m256 _k_a0 = _mm256_loadu_ps(pCoeff);
			__m256 _k_b0 = _k_a0, _k_c0 = _k_a0, _k_d0 = _k_a0;

			__m128 _k_a1 = _mm_loadu_ps(pCoeff + 8);
			__m128 _k_b1 = _k_a1, _k_c1 = _k_a1, _k_d1 = _k_a1;

			for (; j-- > 0;)
			{
				pCoeff -= 12;

				__m256 _c0 = _mm256_loadu_ps(pCoeff);

				_k_a0 = _mm256_fmadd_ps(_k_a0, _z_a, _c0);
				_k_b0 = _mm256_fmadd_ps(_k_b0, _z_b, _c0);
				_k_c0 = _mm256_fmadd_ps(_k_c0, _z_c, _c0);
				_k_d0 = _mm256_fmadd_ps(_k_d0, _z_d, _c0);

				__m128 _c1 = _mm_load_ps(pCoeff + 8);

				_k_a1 = _mm_fmadd_ps(_k_a1, _mm256_castps256_ps128(_z_a), _c1);
				_k_b1 = _mm_fmadd_ps(_k_b1, _mm256_castps256_ps128(_z_b), _c1);
				_k_c1 = _mm_fmadd_ps(_k_c1, _mm256_castps256_ps128(_z_c), _c1);
				_k_d1 = _mm_fmadd_ps(_k_d1, _mm256_castps256_ps128(_z_d), _c1);
			}

			_mm256_storeu_ps(ker + 0, _k_a0);
			_mm_store_ps(ker + 8, _k_a1);
			_mm256_storeu_ps(ker + 12, _k_b0);
			_mm_store_ps(ker + 20, _k_b1);
			_mm256_storeu_ps(ker + 24, _k_c0);
			_mm_store_ps(ker + 32, _k_c1);
			_mm256_storeu_ps(ker + 36, _k_d0);
			_mm_store_ps(ker + 44, _k_d1);
			ker += 48;
		}

		// short tail
		for (BIGINT i = size4; i < size; i++)
		{
			__m256 _x_a = _mm256_set1_ps(x1[i]);
			// scale so local grid offset z in [-1,1]
			__m256 _z_a = _mm256_fmadd_ps(_x_a, _two, _ns_m_1);

			int j = np - 1;

			float* pCoeff = c + 12 * j;

			__m256 _k_a0 = _mm256_load_ps(pCoeff);
			__m128 _k_a1 = _mm_load_ps(pCoeff + 8);

			for (; j-- > 0;)
			{
				pCoeff -= 12;

				__m256 _c0 = _mm256_loadu_ps(pCoeff);
				__m128 _c1 = _mm_load_ps(pCoeff + 8);

				_k_a0 = _mm256_fmadd_ps(_k_a0, _z_a, _c0);
				_k_a1 = _mm_fmadd_ps(_k_a1, _mm256_castps256_ps128(_z_a), _c1);
			}

			_mm256_storeu_ps(ker, _k_a0);
			_mm_store_ps(ker + 8, _k_a1);
			ker += 12;
		}
	}
	else
	{
		float* ker = kernel_vals;

		for (BIGINT i = 0; i < size; i++)
		{
			float x = x1[i];
			float z = 2.0f * x + ns - 1.0f;         // scale so local grid offset z in [-1,1]

			for (int i = 0; i < nsPadded; i++)
			{
				int j = np - 1;
				float k = c[nsPadded * j + i];

				for (; j-- > 0;)
				{
					k = c[nsPadded * j + i] + z * k;
				}

				ker[i] = k;
			}

			ker += nsPadded;

		}
	}
}

template<int ns, int np>
void eval_kernel_bulk_generic(double* c, double* kernel_vals, double* x1, const BIGINT size)
{
	// pad ns to mult of 4
	const int nsPadded = 4 * (1 + (ns - 1) / 4);

	if (nsPadded == 4)
	{
		__m256d _two = _mm256_set1_pd(2.0);
		__m256d _ns_m_1 = _mm256_set1_pd(ns - 1.0);

		BIGINT size4 = size & ~0x11ll;

		double* ker = kernel_vals;

		// main loop
		for (BIGINT i = 0; i < size4; i += 4)
		{
			__m256d _x_a = _mm256_set1_pd(x1[i + 0]);
			__m256d _x_b = _mm256_set1_pd(x1[i + 1]);
			__m256d _x_c = _mm256_set1_pd(x1[i + 2]);
			__m256d _x_d = _mm256_set1_pd(x1[i + 3]);

			// scale so local grid offset z in [-1,1]
			__m256d _z_a = _mm256_fmadd_pd(_x_a, _two, _ns_m_1);
			__m256d _z_b = _mm256_fmadd_pd(_x_b, _two, _ns_m_1);
			__m256d _z_c = _mm256_fmadd_pd(_x_c, _two, _ns_m_1);
			__m256d _z_d = _mm256_fmadd_pd(_x_d, _two, _ns_m_1);

			int j = np - 1;

			__m256d _k_a0 = _mm256_load_pd(c + 0 + nsPadded * j);
			__m256d _k_b0 = _k_a0, _k_c0 = _k_a0, _k_d0 = _k_a0;

			for (; j-- > 0;)
			{
				__m256d _c0 = _mm256_load_pd(c + 0 + nsPadded * j);

				_k_a0 = _mm256_fmadd_pd(_k_a0, _z_a, _c0);
				_k_b0 = _mm256_fmadd_pd(_k_b0, _z_b, _c0);
				_k_c0 = _mm256_fmadd_pd(_k_c0, _z_c, _c0);
				_k_d0 = _mm256_fmadd_pd(_k_d0, _z_d, _c0);
			}

			_mm256_store_pd(ker + 0, _k_a0);
			ker += nsPadded;

			_mm256_store_pd(ker + 0, _k_b0);
			ker += nsPadded;

			_mm256_store_pd(ker + 0, _k_c0);
			ker += nsPadded;

			_mm256_store_pd(ker + 0, _k_d0);
			ker += nsPadded;
		}

		// short tail
		for (BIGINT i = size4; i < size; i++)
		{
			__m256d _x_a = _mm256_set1_pd(x1[i]);
			// scale so local grid offset z in [-1,1]
			__m256d _z_a = _mm256_fmadd_pd(_x_a, _two, _ns_m_1);

			int j = np;

			__m256d _k_a0 = _mm256_load_pd(c + 0 + nsPadded * j);

			for (; j-- > 0;)
			{
				__m256d _c0 = _mm256_load_pd(c + 0 + nsPadded * j);

				_k_a0 = _mm256_fmadd_pd(_k_a0, _z_a, _c0);
			}

			_mm256_store_pd(ker + 0, _k_a0);
			ker += nsPadded;
		}
	}
	else if (nsPadded == 8)
	{
		__m256d _two = _mm256_set1_pd(2.0);
		__m256d _ns_m_1 = _mm256_set1_pd(ns - 1.0);

		BIGINT size2 = size & ~0x1ll;

		double* ker = kernel_vals;

		// main loop
		for (BIGINT i = 0; i < size2; i += 2)
		{
			__m256d _x_a = _mm256_set1_pd(x1[i + 0]);
			__m256d _x_b = _mm256_set1_pd(x1[i + 1]);

			// scale so local grid offset z in [-1,1]
			__m256d _z_a = _mm256_fmadd_pd(_x_a, _two, _ns_m_1);
			__m256d _z_b = _mm256_fmadd_pd(_x_b, _two, _ns_m_1);

			int j = np - 1;

			__m256d _k_a0 = _mm256_load_pd(c + 0 + nsPadded * j);
			__m256d _k_a1 = _mm256_load_pd(c + 4 + nsPadded * j);
			__m256d _k_b0 = _k_a0, _k_b1 = _k_a1;

			for (; j-- > 0;)
			{
				__m256d _c0 = _mm256_load_pd(c + 0 + nsPadded * j);
				__m256d _c1 = _mm256_load_pd(c + 4 + nsPadded * j);

				_k_a0 = _mm256_fmadd_pd(_k_a0, _z_a, _c0);
				_k_a1 = _mm256_fmadd_pd(_k_a1, _z_a, _c1);
				_k_b0 = _mm256_fmadd_pd(_k_b0, _z_b, _c0);
				_k_b1 = _mm256_fmadd_pd(_k_b1, _z_b, _c1);
			}

			_mm256_store_pd(ker + 0, _k_a0);
			_mm256_store_pd(ker + 4, _k_a1);
			ker += nsPadded;

			_mm256_store_pd(ker + 0, _k_b0);
			_mm256_store_pd(ker + 4, _k_b1);
			ker += nsPadded;
		}

		// short tail
		for (BIGINT i = size2; i < size; i++)
		{
			__m256d _x_a = _mm256_set1_pd(x1[i]);
			// scale so local grid offset z in [-1,1]
			__m256d _z_a = _mm256_fmadd_pd(_x_a, _two, _ns_m_1);

			int j = np - 1;

			__m256d _k_a0 = _mm256_load_pd(c + 0 + nsPadded * j);
			__m256d _k_a1 = _mm256_load_pd(c + 4 + nsPadded * j);

			for (; j-- > 0;)
			{
				__m256d _c0 = _mm256_load_pd(c + 0 + nsPadded * j);
				__m256d _c1 = _mm256_load_pd(c + 4 + nsPadded * j);

				_k_a0 = _mm256_fmadd_pd(_k_a0, _z_a, _c0);
				_k_a1 = _mm256_fmadd_pd(_k_a1, _z_a, _c1);
			}

			_mm256_store_pd(ker + 0, _k_a0);
			_mm256_store_pd(ker + 4, _k_a1);
			ker += nsPadded;
		}
	}
	else if (nsPadded == 12)
	{
		__m256d _two = _mm256_set1_pd(2.0);
		__m256d _ns_m_1 = _mm256_set1_pd(ns - 1.0);

		BIGINT size2 = size & ~0x1ll;

		double* ker = kernel_vals;

		// main loop
		for (BIGINT i = 0; i < size2; i += 2)
		{
			__m256d _x_a = _mm256_set1_pd(x1[i + 0]);
			__m256d _x_b = _mm256_set1_pd(x1[i + 1]);

			// scale so local grid offset z in [-1,1]
			__m256d _z_a = _mm256_fmadd_pd(_x_a, _two, _ns_m_1);
			__m256d _z_b = _mm256_fmadd_pd(_x_b, _two, _ns_m_1);

			int j = np - 1;

			__m256d _k_a0 = _mm256_load_pd(c + 0 + nsPadded * j);
			__m256d _k_a1 = _mm256_load_pd(c + 4 + nsPadded * j);
			__m256d _k_a2 = _mm256_load_pd(c + 4 + nsPadded * j);
			__m256d _k_b0 = _k_a0, _k_b1 = _k_a1, _k_b2 = _k_a2;

			for (; j-- > 0;)
			{
				__m256d _c0 = _mm256_load_pd(c + 0 + nsPadded * j);
				__m256d _c1 = _mm256_load_pd(c + 4 + nsPadded * j);
				__m256d _c2 = _mm256_load_pd(c + 8 + nsPadded * j);

				_k_a0 = _mm256_fmadd_pd(_k_a0, _z_a, _c0);
				_k_a1 = _mm256_fmadd_pd(_k_a1, _z_a, _c1);
				_k_a2 = _mm256_fmadd_pd(_k_a2, _z_a, _c2);
				_k_b0 = _mm256_fmadd_pd(_k_b0, _z_b, _c0);
				_k_b1 = _mm256_fmadd_pd(_k_b1, _z_b, _c1);
				_k_b2 = _mm256_fmadd_pd(_k_b2, _z_b, _c2);
			}

			_mm256_store_pd(ker + 0, _k_a0);
			_mm256_store_pd(ker + 4, _k_a1);
			_mm256_store_pd(ker + 8, _k_a2);
			ker += nsPadded;

			_mm256_store_pd(ker + 0, _k_b0);
			_mm256_store_pd(ker + 4, _k_b1);
			_mm256_store_pd(ker + 8, _k_b2);
			ker += nsPadded;
		}

		// short tail
		for (BIGINT i = size2; i < size; i++)
		{
			__m256d _x_a = _mm256_set1_pd(x1[i]);
			// scale so local grid offset z in [-1,1]
			__m256d _z_a = _mm256_fmadd_pd(_x_a, _two, _ns_m_1);

			int j = np - 1;

			__m256d _k_a0 = _mm256_load_pd(c + 0 + nsPadded * j);
			__m256d _k_a1 = _mm256_load_pd(c + 4 + nsPadded * j);
			__m256d _k_a2 = _mm256_load_pd(c + 8 + nsPadded * j);

			for (; j-- > 0;)
			{
				__m256d _c0 = _mm256_load_pd(c + 0 + nsPadded * j);
				__m256d _c1 = _mm256_load_pd(c + 4 + nsPadded * j);
				__m256d _c2 = _mm256_load_pd(c + 8 + nsPadded * j);

				_k_a0 = _mm256_fmadd_pd(_k_a0, _z_a, _c0);
				_k_a1 = _mm256_fmadd_pd(_k_a1, _z_a, _c1);
				_k_a2 = _mm256_fmadd_pd(_k_a2, _z_a, _c2);
			}

			_mm256_store_pd(ker + 0, _k_a0);
			_mm256_store_pd(ker + 4, _k_a1);
			_mm256_store_pd(ker + 8, _k_a2);
			ker += nsPadded;
		}
	}
	else if (nsPadded == 16)
	{
		__m256d _two = _mm256_set1_pd(2.0);
		__m256d _ns_m_1 = _mm256_set1_pd(ns - 1.0);

		double* ker = kernel_vals;

		// main loop
		for (BIGINT i = 0; i < size; i++)
		{
			__m256d _x_a = _mm256_set1_pd(x1[i]);
			// scale so local grid offset z in [-1,1]
			__m256d _z_a = _mm256_fmadd_pd(_x_a, _two, _ns_m_1);

			int j = np - 1;

			__m256d _k_a0 = _mm256_load_pd(c + 0 + nsPadded * j);
			__m256d _k_a1 = _mm256_load_pd(c + 4 + nsPadded * j);
			__m256d _k_a2 = _mm256_load_pd(c + 8 + nsPadded * j);
			__m256d _k_a3 = _mm256_load_pd(c + 12 + nsPadded * j);

			for (; j-- > 0;)
			{
				__m256d _c0 = _mm256_load_pd(c + 0 + nsPadded * j);
				__m256d _c1 = _mm256_load_pd(c + 4 + nsPadded * j);
				__m256d _c2 = _mm256_load_pd(c + 8 + nsPadded * j);
				__m256d _c3 = _mm256_load_pd(c + 12 + nsPadded * j);

				_k_a0 = _mm256_fmadd_pd(_k_a0, _z_a, _c0);
				_k_a1 = _mm256_fmadd_pd(_k_a1, _z_a, _c1);
				_k_a2 = _mm256_fmadd_pd(_k_a2, _z_a, _c2);
				_k_a3 = _mm256_fmadd_pd(_k_a3, _z_a, _c3);
			}

			_mm256_store_pd(ker + 0, _k_a0);
			_mm256_store_pd(ker + 4, _k_a1);
			_mm256_store_pd(ker + 8, _k_a2);
			_mm256_store_pd(ker + 12, _k_a3);
			ker += nsPadded;
		}
	}
	else
	{
		double* ker = kernel_vals;

		for (BIGINT i = 0; i < size; i++)
		{
			double x = x1[i];
			double z = 2.0 * x + ns - 1.0;         // scale so local grid offset z in [-1,1]

			for (int i = 0; i < nsPadded; i++)
			{
				int j = np - 1;
				double k = c[nsPadded * j + i];

				for (; j-- > 0;)
				{
					k = c[nsPadded * j + i] + z * k;
				}

				ker[i] = k;
			}

			ker += nsPadded;
		}
	}
}
#ifdef __AVX512F__
template<>
inline void eval_kernel_bulk_generic<7, 10>(float* c, float* kernel_vals, float* x1, const BIGINT size)
{
	__m512 _two = _mm512_set1_ps(2.0f);
	__m512 _ns_m_1 = _mm512_set1_ps(6.0f);

	BIGINT size16 = size - size % 16;

	float* ker = kernel_vals;

	__m512 _c0 = _mm512_broadcast_f32x8(_mm256_load_ps(c + 0));
	__m512 _c1 = _mm512_broadcast_f32x8(_mm256_load_ps(c + 8));
	__m512 _c2 = _mm512_broadcast_f32x8(_mm256_load_ps(c + 16));
	__m512 _c3 = _mm512_broadcast_f32x8(_mm256_load_ps(c + 24));
	__m512 _c4 = _mm512_broadcast_f32x8(_mm256_load_ps(c + 32));
	__m512 _c5 = _mm512_broadcast_f32x8(_mm256_load_ps(c + 40));
	__m512 _c6 = _mm512_broadcast_f32x8(_mm256_load_ps(c + 48));
	__m512 _c7 = _mm512_broadcast_f32x8(_mm256_load_ps(c + 56));
	__m512 _c8 = _mm512_broadcast_f32x8(_mm256_load_ps(c + 64));
	__m512 _c9 = _mm512_broadcast_f32x8(_mm256_load_ps(c + 72));

	// main loop
	for (BIGINT i = 0; i < size16; i += 16)
	{
		__m512 _x_ab = _mm512_insertf32x8(_mm512_set1_ps(x1[i + 0]), _mm256_set1_ps(x1[i + 1]), 1);
		__m512 _x_cd = _mm512_insertf32x8(_mm512_set1_ps(x1[i + 2]), _mm256_set1_ps(x1[i + 3]), 1);
		__m512 _x_ef = _mm512_insertf32x8(_mm512_set1_ps(x1[i + 4]), _mm256_set1_ps(x1[i + 5]), 1);
		__m512 _x_gh = _mm512_insertf32x8(_mm512_set1_ps(x1[i + 6]), _mm256_set1_ps(x1[i + 7]), 1);
		__m512 _x_ij = _mm512_insertf32x8(_mm512_set1_ps(x1[i + 8]), _mm256_set1_ps(x1[i + 9]), 1);
		__m512 _x_kl = _mm512_insertf32x8(_mm512_set1_ps(x1[i + 10]), _mm256_set1_ps(x1[i + 11]), 1);
		__m512 _x_mn = _mm512_insertf32x8(_mm512_set1_ps(x1[i + 12]), _mm256_set1_ps(x1[i + 13]), 1);
		__m512 _x_op = _mm512_insertf32x8(_mm512_set1_ps(x1[i + 14]), _mm256_set1_ps(x1[i + 15]), 1);

		// scale so local grid offset z in [-1,1]
		__m512 _z_ab = _mm512_fmadd_ps(_x_ab, _two, _ns_m_1);
		__m512 _z_cd = _mm512_fmadd_ps(_x_cd, _two, _ns_m_1);
		__m512 _z_ef = _mm512_fmadd_ps(_x_ef, _two, _ns_m_1);
		__m512 _z_gh = _mm512_fmadd_ps(_x_gh, _two, _ns_m_1);
		__m512 _z_ij = _mm512_fmadd_ps(_x_ij, _two, _ns_m_1);
		__m512 _z_kl = _mm512_fmadd_ps(_x_kl, _two, _ns_m_1);
		__m512 _z_mn = _mm512_fmadd_ps(_x_mn, _two, _ns_m_1);
		__m512 _z_op = _mm512_fmadd_ps(_x_op, _two, _ns_m_1);

		__m512 _k_ab = _mm512_fmadd_ps(_c9, _z_ab, _c8);
		__m512 _k_cd = _mm512_fmadd_ps(_c9, _z_cd, _c8);
		__m512 _k_ef = _mm512_fmadd_ps(_c9, _z_ef, _c8);
		__m512 _k_gh = _mm512_fmadd_ps(_c9, _z_gh, _c8);
		__m512 _k_ij = _mm512_fmadd_ps(_c9, _z_ij, _c8);
		__m512 _k_kl = _mm512_fmadd_ps(_c9, _z_kl, _c8);
		__m512 _k_mn = _mm512_fmadd_ps(_c9, _z_mn, _c8);
		__m512 _k_op = _mm512_fmadd_ps(_c9, _z_op, _c8);

		_k_ab = _mm512_fmadd_ps(_k_ab, _z_ab, _c7);
		_k_cd = _mm512_fmadd_ps(_k_cd, _z_cd, _c7);
		_k_ef = _mm512_fmadd_ps(_k_ef, _z_ef, _c7);
		_k_gh = _mm512_fmadd_ps(_k_gh, _z_gh, _c7);
		_k_ij = _mm512_fmadd_ps(_k_ij, _z_ij, _c7);
		_k_kl = _mm512_fmadd_ps(_k_kl, _z_kl, _c7);
		_k_mn = _mm512_fmadd_ps(_k_mn, _z_mn, _c7);
		_k_op = _mm512_fmadd_ps(_k_op, _z_op, _c7);

		_k_ab = _mm512_fmadd_ps(_k_ab, _z_ab, _c6);
		_k_cd = _mm512_fmadd_ps(_k_cd, _z_cd, _c6);
		_k_ef = _mm512_fmadd_ps(_k_ef, _z_ef, _c6);
		_k_gh = _mm512_fmadd_ps(_k_gh, _z_gh, _c6);
		_k_ij = _mm512_fmadd_ps(_k_ij, _z_ij, _c6);
		_k_kl = _mm512_fmadd_ps(_k_kl, _z_kl, _c6);
		_k_mn = _mm512_fmadd_ps(_k_mn, _z_mn, _c6);
		_k_op = _mm512_fmadd_ps(_k_op, _z_op, _c6);

		_k_ab = _mm512_fmadd_ps(_k_ab, _z_ab, _c5);
		_k_cd = _mm512_fmadd_ps(_k_cd, _z_cd, _c5);
		_k_ef = _mm512_fmadd_ps(_k_ef, _z_ef, _c5);
		_k_gh = _mm512_fmadd_ps(_k_gh, _z_gh, _c5);
		_k_ij = _mm512_fmadd_ps(_k_ij, _z_ij, _c5);
		_k_kl = _mm512_fmadd_ps(_k_kl, _z_kl, _c5);
		_k_mn = _mm512_fmadd_ps(_k_mn, _z_mn, _c5);
		_k_op = _mm512_fmadd_ps(_k_op, _z_op, _c5);

		_k_ab = _mm512_fmadd_ps(_k_ab, _z_ab, _c4);
		_k_cd = _mm512_fmadd_ps(_k_cd, _z_cd, _c4);
		_k_ef = _mm512_fmadd_ps(_k_ef, _z_ef, _c4);
		_k_gh = _mm512_fmadd_ps(_k_gh, _z_gh, _c4);
		_k_ij = _mm512_fmadd_ps(_k_ij, _z_ij, _c4);
		_k_kl = _mm512_fmadd_ps(_k_kl, _z_kl, _c4);
		_k_mn = _mm512_fmadd_ps(_k_mn, _z_mn, _c4);
		_k_op = _mm512_fmadd_ps(_k_op, _z_op, _c4);

		_k_ab = _mm512_fmadd_ps(_k_ab, _z_ab, _c3);
		_k_cd = _mm512_fmadd_ps(_k_cd, _z_cd, _c3);
		_k_ef = _mm512_fmadd_ps(_k_ef, _z_ef, _c3);
		_k_gh = _mm512_fmadd_ps(_k_gh, _z_gh, _c3);
		_k_ij = _mm512_fmadd_ps(_k_ij, _z_ij, _c3);
		_k_kl = _mm512_fmadd_ps(_k_kl, _z_kl, _c3);
		_k_mn = _mm512_fmadd_ps(_k_mn, _z_mn, _c3);
		_k_op = _mm512_fmadd_ps(_k_op, _z_op, _c3);

		_k_ab = _mm512_fmadd_ps(_k_ab, _z_ab, _c2);
		_k_cd = _mm512_fmadd_ps(_k_cd, _z_cd, _c2);
		_k_ef = _mm512_fmadd_ps(_k_ef, _z_ef, _c2);
		_k_gh = _mm512_fmadd_ps(_k_gh, _z_gh, _c2);
		_k_ij = _mm512_fmadd_ps(_k_ij, _z_ij, _c2);
		_k_kl = _mm512_fmadd_ps(_k_kl, _z_kl, _c2);
		_k_mn = _mm512_fmadd_ps(_k_mn, _z_mn, _c2);
		_k_op = _mm512_fmadd_ps(_k_op, _z_op, _c2);

		_k_ab = _mm512_fmadd_ps(_k_ab, _z_ab, _c1);
		_k_cd = _mm512_fmadd_ps(_k_cd, _z_cd, _c1);
		_k_ef = _mm512_fmadd_ps(_k_ef, _z_ef, _c1);
		_k_gh = _mm512_fmadd_ps(_k_gh, _z_gh, _c1);
		_k_ij = _mm512_fmadd_ps(_k_ij, _z_ij, _c1);
		_k_kl = _mm512_fmadd_ps(_k_kl, _z_kl, _c1);
		_k_mn = _mm512_fmadd_ps(_k_mn, _z_mn, _c1);
		_k_op = _mm512_fmadd_ps(_k_op, _z_op, _c1);

		_k_ab = _mm512_fmadd_ps(_k_ab, _z_ab, _c0);
		_k_cd = _mm512_fmadd_ps(_k_cd, _z_cd, _c0);
		_k_ef = _mm512_fmadd_ps(_k_ef, _z_ef, _c0);
		_k_gh = _mm512_fmadd_ps(_k_gh, _z_gh, _c0);
		_k_ij = _mm512_fmadd_ps(_k_ij, _z_ij, _c0);
		_k_kl = _mm512_fmadd_ps(_k_kl, _z_kl, _c0);
		_k_mn = _mm512_fmadd_ps(_k_mn, _z_mn, _c0);
		_k_op = _mm512_fmadd_ps(_k_op, _z_op, _c0);

		_mm512_store_ps(ker + 0, _k_ab);
		_mm512_store_ps(ker + 16, _k_cd);
		_mm512_store_ps(ker + 32, _k_ef);
		_mm512_store_ps(ker + 48, _k_gh);
		_mm512_store_ps(ker + 64, _k_ij);
		_mm512_store_ps(ker + 80, _k_kl);
		_mm512_store_ps(ker + 96, _k_mn);
		_mm512_store_ps(ker + 112, _k_op);

		ker += 128;
	}

	// short tail
	for (BIGINT i = size16; i < size; i++)
	{
		__m512 _x_a = _mm512_set1_ps(x1[i]);
		// scale so local grid offset z in [-1,1]
		__m512 _z_a = _mm512_fmadd_ps(_x_a, _two, _ns_m_1);

		__m512 _k_a = _mm512_fmadd_ps(_c9, _z_a, _c8);

		_k_a = _mm512_fmadd_ps(_k_a, _z_a, _c7);
		_k_a = _mm512_fmadd_ps(_k_a, _z_a, _c6);
		_k_a = _mm512_fmadd_ps(_k_a, _z_a, _c5);
		_k_a = _mm512_fmadd_ps(_k_a, _z_a, _c4);
		_k_a = _mm512_fmadd_ps(_k_a, _z_a, _c3);
		_k_a = _mm512_fmadd_ps(_k_a, _z_a, _c2);
		_k_a = _mm512_fmadd_ps(_k_a, _z_a, _c1);
		_k_a = _mm512_fmadd_ps(_k_a, _z_a, _c0);

		_mm256_store_ps(ker, _mm512_castps512_ps256(_k_a));

		ker += 8;
	}
}
#else
inline __m256 _mm_broadcastps128_ps256(__m128 a) {
	return _mm256_castsi256_ps(_mm256_broadcastsi128_si256(_mm_castps_si128(a)));
}

template<>
inline void eval_kernel_bulk_generic<4, 7>(float* c, float* kernel_vals, float* x1, const BIGINT size)
{
	__m256 _two = _mm256_set1_ps(2.0f);
	__m256 _ns_m_1 = _mm256_set1_ps(3.0f);

	BIGINT size8 = size - size % 8;

	float* ker = kernel_vals;

	__m256 _c0 = _mm_broadcastps128_ps256(_mm_load_ps(c + 0));
	__m256 _c1 = _mm_broadcastps128_ps256(_mm_load_ps(c + 4));
	__m256 _c2 = _mm_broadcastps128_ps256(_mm_load_ps(c + 8));
	__m256 _c3 = _mm_broadcastps128_ps256(_mm_load_ps(c + 12));
	__m256 _c4 = _mm_broadcastps128_ps256(_mm_load_ps(c + 16));
	__m256 _c5 = _mm_broadcastps128_ps256(_mm_load_ps(c + 20));
	__m256 _c6 = _mm_broadcastps128_ps256(_mm_load_ps(c + 24));

	// main loop
	for (BIGINT i = 0; i < size8; i += 8)
	{
		__m256 _x_ab = _mm256_insertf128_ps(_mm256_set1_ps(x1[i + 0]), _mm_set1_ps(x1[i + 1]), 1);
		__m256 _x_cd = _mm256_insertf128_ps(_mm256_set1_ps(x1[i + 2]), _mm_set1_ps(x1[i + 3]), 1);
		__m256 _x_ef = _mm256_insertf128_ps(_mm256_set1_ps(x1[i + 4]), _mm_set1_ps(x1[i + 5]), 1);
		__m256 _x_gh = _mm256_insertf128_ps(_mm256_set1_ps(x1[i + 6]), _mm_set1_ps(x1[i + 7]), 1);

		// scale so local grid offset z in [-1,1]
		__m256 _z_ab = _mm256_fmadd_ps(_x_ab, _two, _ns_m_1);
		__m256 _z_cd = _mm256_fmadd_ps(_x_cd, _two, _ns_m_1);
		__m256 _z_ef = _mm256_fmadd_ps(_x_ef, _two, _ns_m_1);
		__m256 _z_gh = _mm256_fmadd_ps(_x_gh, _two, _ns_m_1);

		__m256 _k_ab = _mm256_fmadd_ps(_c6, _z_ab, _c5);
		__m256 _k_cd = _mm256_fmadd_ps(_c6, _z_cd, _c5);
		__m256 _k_ef = _mm256_fmadd_ps(_c6, _z_ef, _c5);
		__m256 _k_gh = _mm256_fmadd_ps(_c6, _z_gh, _c5);

		_k_ab = _mm256_fmadd_ps(_k_ab, _z_ab, _c4);
		_k_cd = _mm256_fmadd_ps(_k_cd, _z_cd, _c4);
		_k_ef = _mm256_fmadd_ps(_k_ef, _z_ef, _c4);
		_k_gh = _mm256_fmadd_ps(_k_gh, _z_gh, _c4);

		_k_ab = _mm256_fmadd_ps(_k_ab, _z_ab, _c3);
		_k_cd = _mm256_fmadd_ps(_k_cd, _z_cd, _c3);
		_k_ef = _mm256_fmadd_ps(_k_ef, _z_ef, _c3);
		_k_gh = _mm256_fmadd_ps(_k_gh, _z_gh, _c3);

		_k_ab = _mm256_fmadd_ps(_k_ab, _z_ab, _c2);
		_k_cd = _mm256_fmadd_ps(_k_cd, _z_cd, _c2);
		_k_ef = _mm256_fmadd_ps(_k_ef, _z_ef, _c2);
		_k_gh = _mm256_fmadd_ps(_k_gh, _z_gh, _c2);

		_k_ab = _mm256_fmadd_ps(_k_ab, _z_ab, _c1);
		_k_cd = _mm256_fmadd_ps(_k_cd, _z_cd, _c1);
		_k_ef = _mm256_fmadd_ps(_k_ef, _z_ef, _c1);
		_k_gh = _mm256_fmadd_ps(_k_gh, _z_gh, _c1);

		_k_ab = _mm256_fmadd_ps(_k_ab, _z_ab, _c0);
		_k_cd = _mm256_fmadd_ps(_k_cd, _z_cd, _c0);
		_k_ef = _mm256_fmadd_ps(_k_ef, _z_ef, _c0);
		_k_gh = _mm256_fmadd_ps(_k_gh, _z_gh, _c0);

		_mm256_store_ps(ker + 0, _k_ab);
		_mm256_store_ps(ker + 8, _k_cd);
		_mm256_store_ps(ker + 16, _k_ef);
		_mm256_store_ps(ker + 24, _k_gh);

		ker += 32;
	}

	// short tail
	for (BIGINT i = size8; i < size; i++)
	{
		__m256 _x_a = _mm256_set1_ps(x1[i]);
		// scale so local grid offset z in [-1,1]
		__m256 _z_a = _mm256_fmadd_ps(_x_a, _two, _ns_m_1);

		__m256 _k_a = _mm256_fmadd_ps(_c6, _z_a, _c5);

		_k_a = _mm256_fmadd_ps(_k_a, _z_a, _c4);
		_k_a = _mm256_fmadd_ps(_k_a, _z_a, _c3);
		_k_a = _mm256_fmadd_ps(_k_a, _z_a, _c2);
		_k_a = _mm256_fmadd_ps(_k_a, _z_a, _c1);
		_k_a = _mm256_fmadd_ps(_k_a, _z_a, _c0);

		_mm_store_ps(ker, _mm256_castps256_ps128(_k_a));

		ker += 4;
	}
}

#endif
#else
template<int ns, int np>
void eval_kernel_bulk_generic(FLT* c, FLT* kernel_vals, FLT* x1, const BIGINT size)
{
	// pad ns to mult of 4
	int nsPadded = 4 * (1 + (ns - 1) / 4);

	for (BIGINT m = 0; m < size; m++)
	{
		FLT* ker = kernel_vals + (m * nsPadded);
		FLT x = x1[m];
		FLT z = 2 * x + ns - 1.0;         // scale so local grid offset z in [-1,1]

		for (int i = 0; i < nsPadded; i++)
		{
			int j = np - 1;
			FLT k = c[nsPadded * j + i];

			for (; j-- > 0;)
			{
				k = c[nsPadded * j + i] + z * k;
			}

			ker[i] = k;
		}
	}
}
#endif

void eval_kernel_bulk_Horner(FLT* kernel_vals, FLT* x1, const int w, const BIGINT size,
	const spread_opts& opts)
	/* Fill ker[] with Horner piecewise poly approx to [-w/2,w/2] ES kernel eval at
	   x_j = x + j,  for j=0,..,w-1.  Thus x in [-w/2,-w/2+1].   w is aka ns.
	   This is the current evaluation method, since it's faster (except i7 w=16).
	   Two upsampfacs implemented. Params must match ref formula. Barnett 4/24/18 */
{
	alignas(64) FLT c1252[] = {
		2.3711015472112514E+01f, 2.3711015472112514E+01f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		2.5079742199350562E+01f, -2.5079742199350562E+01f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		-3.5023281580177050E+00f, -3.5023281580177086E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		-7.3894949249195587E+00f, 7.3894949249195632E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f };

	alignas(64) FLT c1253[] = {
		5.9620016143346824E+01f, 2.4110216701187497E+02f, 5.9620016148621815E+01f, 0.0000000000000000E+00f,
		9.7575520958604258E+01f, 9.4807967775797928E-16f, -9.7575520952908519E+01f, 0.0000000000000000E+00f,
		3.5838417859768512E+01f, -7.3472145274965371E+01f, 3.5838417865129472E+01f, 0.0000000000000000E+00f,
		-1.0721643298166471E+01f, -2.1299978194824344E-16f, 1.0721643303220413E+01f, 0.0000000000000000E+00f,
		-7.0570630207138318E+00f, 9.1538553399011260E+00f, -7.0570630151506633E+00f, 0.0000000000000000E+00f };

	alignas(64) FLT c1254[] = {
		1.2612470018753689E+02f, 1.1896204292999116E+03f, 1.1896204292999118E+03f, 1.2612470018753696E+02f,
		2.6158034850676626E+02f, 5.6161104654809810E+02f, -5.6161104654809844E+02f, -2.6158034850676620E+02f,
		1.7145379463699527E+02f, -1.6695967127766517E+02f, -1.6695967127766514E+02f, 1.7145379463699527E+02f,
		2.3525961965887870E+01f, -1.0057439659768858E+02f, 1.0057439659768873E+02f, -2.3525961965887827E+01f,
		-1.5608307370340880E+01f, 9.5627412100260845E+00f, 9.5627412100260205E+00f, -1.5608307370340908E+01f,
		-4.5715207776748699E+00f, 7.9904373067895493E+00f, -7.9904373067893877E+00f, 4.5715207776749462E+00f };

	alignas(64) FLT c1255[] = {
		2.4106943677442615E+02f, 4.3538384278025542E+03f, 9.3397486707381995E+03f, 4.3538384278025515E+03f, 2.4106943677442607E+02f, 0.0000000000000000E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		5.8781364250328272E+02f, 3.4742855804122028E+03f, -7.3041306797303120E-14f, -3.4742855804122009E+03f, -5.8781364250328249E+02f, 0.0000000000000000E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		5.1234107167555862E+02f, 3.5219546517037116E+02f, -1.7076861141633149E+03f, 3.5219546517037247E+02f, 5.1234107167555862E+02f, 0.0000000000000000E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		1.7540956907856057E+02f, -3.5792356187777074E+02f, -4.9888896652511712E-13f, 3.5792356187777165E+02f, -1.7540956907856059E+02f, 0.0000000000000000E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		-2.1768066955094961E-01f, -7.8322173187697558E+01f, 1.3904039464934516E+02f, -7.8322173187697842E+01f, -2.1768066955103071E-01f, 0.0000000000000000E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		-1.4207955403641256E+01f, 1.6019466986221790E+01f, 5.4386376890865855E-13f, -1.6019466986220916E+01f, 1.4207955403641320E+01f, 0.0000000000000000E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		-2.1966493586753826E+00f, 5.0672636163194582E+00f, -6.7340544905090631E+00f, 5.0672636163189448E+00f, -2.1966493586753089E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00 };

	alignas(64) FLT c1256[] = {
		4.3011762559089101E+02f, 1.3368828836127070E+04f, 4.9861340433371224E+04f, 4.9861340433371253E+04f, 1.3368828836127073E+04f, 4.3011762559835148E+02f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		1.1857225840065141E+03f, 1.4112553227730617E+04f, 1.5410005180819440E+04f, -1.5410005180819426E+04f, -1.4112553227730616E+04f, -1.1857225839984601E+03f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		1.2460481448413077E+03f, 4.3127030215084960E+03f, -5.5438591621431169E+03f, -5.5438591621431306E+03f, 4.3127030215084960E+03f, 1.2460481448488902E+03f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		6.0825549344387753E+02f, -3.4106010789547094E+02f, -1.9775725023673197E+03f, 1.9775725023673208E+03f, 3.4106010789547116E+02f, -6.0825549343673094E+02f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		1.1264961069783706E+02f, -3.9740822717991142E+02f, 2.7557540616463064E+02f, 2.7557540616462472E+02f, -3.9740822717991210E+02f, 1.1264961070570448E+02f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		-1.5387906304333878E+01f, -3.2640579296387394E+01f, 1.1683718215647470E+02f, -1.1683718215646800E+02f, 3.2640579296390861E+01f, 1.5387906311562851E+01f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		-9.3947198873910249E+00f, 1.5069930500881778E+01f, -8.0900452409597179E+00f, -8.0900452409538364E+00f, 1.5069930500884301E+01f, -9.3947198802581902E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		-5.6048841964539509E-01f, 2.3377422080924530E+00f, -4.2391567591836514E+00f, 4.2391567591841817E+00f, -2.3377422080928629E+00f, 5.6048842664294984E-01f, 0.0000000000000000E+00f, 0.0000000000000000E+00 };

	alignas(64) FLT c1257[] = {
		7.2950392616203249E+02f, 3.6439117038309480E+04f, 2.1220891582018422E+05f, 3.6180058567561524E+05f, 2.1220891582018445E+05f, 3.6439117038309487E+04f, 7.2950392617434545E+02f, 0.0000000000000000E+00f,
		2.2197790785452576E+03f, 4.6392067080426248E+04f, 1.1568051746995670E+05f, -1.1902861988308852E-11f, -1.1568051746995671E+05f, -4.6392067080426241E+04f, -2.2197790785319785E+03f, 0.0000000000000000E+00f,
		2.6796845075663955E+03f, 2.0921129984587249E+04f, 3.9399551345574849E+01f, -4.7251335435527435E+04f, 3.9399551345580633E+01f, 2.0921129984587245E+04f, 2.6796845075789142E+03f, 0.0000000000000000E+00f,
		1.6253748990844499E+03f, 2.6138488347211564E+03f, -1.0037546705421508E+04f, 2.6823166126907972E-11f, 1.0037546705421508E+04f, -2.6138488347211546E+03f, -1.6253748990726619E+03f, 0.0000000000000000E+00f,
		4.9106375852553418E+02f, -8.6668269315416171E+02f, -1.0513434716618249E+03f, 2.8444456471590756E+03f, -1.0513434716618387E+03f, -8.6668269315416057E+02f, 4.9106375853851472E+02f, 0.0000000000000000E+00f,
		4.0739167949763157E+01f, -2.8515155742293922E+02f, 3.9930326803801455E+02f, 2.4847312048933061E-11f, -3.9930326803798215E+02f, 2.8515155742293899E+02f, -4.0739167937835738E+01f, 0.0000000000000000E+00f,
		-1.7148987139838667E+01f, 7.5799002551700223E-01f, 6.3260304953160343E+01f, -1.0529869309160161E+02f, 6.3260304953194023E+01f, 7.5799002552709915E-01f, -1.7148987128069749E+01f, 0.0000000000000000E+00f,
		-4.5424411501060264E+00f, 9.8749254058318616E+00f, -9.6456179777547195E+00f, 2.0621161109877312E-11f, 9.6456179778118027E+00f, -9.8749254058319202E+00f, 4.5424411616514604E+00f, 0.0000000000000000E+00f,
		-5.0793946806832954E-02f, 7.3273813711856639E-01f, -2.0117140544738263E+00f, 2.6999257940856816E+00f, -2.0117140545416512E+00f, 7.3273813711318592E-01f, -5.0793935653327994E-02f, 0.0000000000000000E+00 };

	alignas(64) FLT c1258[] = {
		1.1895823653767145E+03f, 9.0980236725236929E+04f, 7.7438826909537485E+05f, 2.0077596413122697E+06f, 2.0077596413122697E+06f, 7.7438826909537497E+05f, 9.0980236725236929E+04f, 1.1895823653767147E+03f,
		3.9313191526977798E+03f, 1.3318570706800820E+05f, 5.7275848637687636E+05f, 4.6250273225257988E+05f, -4.6250273225257976E+05f, -5.7275848637687659E+05f, -1.3318570706800820E+05f, -3.9313191526977798E+03f,
		5.2976026193612370E+03f, 7.5628970871188430E+04f, 1.0073339198368321E+05f, -1.8165150843791291E+05f, -1.8165150843791291E+05f, 1.0073339198368321E+05f, 7.5628970871188460E+04f, 5.2976026193612397E+03f,
		3.7552239608473842E+03f, 1.8376340228970901E+04f, -2.3878081117551585E+04f, -4.6296734056047833E+04f, 4.6296734056048226E+04f, 2.3878081117551632E+04f, -1.8376340228970901E+04f, -3.7552239608473833E+03f,
		1.4742862505418652E+03f, 1.2842168112178376E+02f, -9.1969665138398723E+03f, 7.5990739935234687E+03f, 7.5990739935234151E+03f, -9.1969665138399178E+03f, 1.2842168112178072E+02f, 1.4742862505418645E+03f,
		2.8158981009344416E+02f, -8.8613607108855206E+02f, 5.3457145342334378E+01f, 2.1750989694614777E+03f, -2.1750989694609211E+03f, -5.3457145342173561E+01f, 8.8613607108856843E+02f, -2.8158981009344393E+02f,
		-1.4786862436240726E+00f, -1.3935442261830281E+02f, 3.2599325739083491E+02f, -1.9541889343332295E+02f, -1.9541889343339443E+02f, 3.2599325739083696E+02f, -1.3935442261827953E+02f, -1.4786862436237442E+00f,
		-1.1542034522902307E+01f, 1.2000512051397084E+01f, 1.9687328710129744E+01f, -6.3962883082482271E+01f, 6.3962883082874910E+01f, -1.9687328710101575E+01f, -1.2000512051407391E+01f, 1.1542034522902124E+01f,
		-1.7448292513542445E+00f, 4.8577330433956609E+00f, -6.8794163043773890E+00f, 3.4611708987408365E+00f, 3.4611708985348386E+00f, -6.8794163043605385E+00f, 4.8577330433771184E+00f, -1.7448292513550807E+00f,
		1.5044951479021193E-01f, 9.6230159355094713E-02f, -7.0399250398052082E-01f, 1.3251401132916929E+00f, -1.3251401128795544E+00f, 7.0399250407339709E-01f, -9.6230159355094713E-02f, -1.5044951479003055E-01 };

	alignas(64) FLT c1259[] = {
		1.8793738965776997E+03f, 2.1220891582018419E+05f, 2.5233246441351641E+06f, 9.2877384983420596E+06f, 1.4015330434461458E+07f, 9.2877384983420689E+06f, 2.5233246441351632E+06f, 2.1220891582018507E+05f, 1.8793738965777015E+03f, 0.0000000000000000E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		6.6675066501609344E+03f, 3.4704155240986997E+05f, 2.2890184838322559E+06f, 3.8705035445351214E+06f, -1.6037058324963857E-09f, -3.8705035445351251E+06f, -2.2890184838322555E+06f, -3.4704155240987107E+05f, -6.6675066501609363E+03f, 0.0000000000000000E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		9.8412775404612330E+03f, 2.3171563090202375E+05f, 6.8167589492092200E+05f, -2.1140963571671984E+05f, -1.4236515118873848E+06f, -2.1140963571672366E+05f, 6.8167589492092165E+05f, 2.3171563090202425E+05f, 9.8412775404612312E+03f, 0.0000000000000000E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		7.8762358364031033E+03f, 7.6500585979636104E+04f, 1.2434778984075023E+04f, -2.8572091469430045E+05f, 1.5952874106327477E-09f, 2.8572091469430359E+05f, -1.2434778984075045E+04f, -7.6500585979636220E+04f, -7.8762358364031052E+03f, 0.0000000000000000E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		3.6941911906762084E+03f, 9.9232929169975941E+03f, -3.3472877669902169E+04f, -1.4082384858052235E+04f, 6.7911966136972551E+04f, -1.4082384858047793E+04f, -3.3472877669902322E+04f, 9.9232929169976087E+03f, 3.6941911906762070E+03f, 0.0000000000000000E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		9.8900189723050266E+02f, -1.2736589324621855E+03f, -5.0407308390126955E+03f, 9.8914296140171609E+03f, 1.0742991696587890E-09f, -9.8914296140222541E+03f, 5.0407308390134704E+03f, 1.2736589324621880E+03f, -9.8900189723050198E+02f, 0.0000000000000000E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		1.1165868717715853E+02f, -5.9057035448564977E+02f, 5.5860705835603983E+02f, 9.1996097522959656E+02f, -2.0290255886377897E+03f, 9.1996097523001129E+02f, 5.5860705835622480E+02f, -5.9057035448564693E+02f, 1.1165868717715870E+02f, 0.0000000000000000E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		-1.3142584300868881E+01f, -4.2852762793304592E+01f, 1.8188640945795066E+02f, -2.1362000457567430E+02f, 6.1024810759112463E-10f, 2.1362000457722939E+02f, -1.8188640945795305E+02f, 4.2852762793363922E+01f, 1.3142584300866494E+01f, 0.0000000000000000E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		-5.8088068374878068E+00f, 1.0201832931362965E+01f, -3.5220973519213472E-01f, -2.6632420896811951E+01f, 4.2737607182672249E+01f, -2.6632420895534445E+01f, -3.5220973562147767E-01f, 1.0201832931230712E+01f, -5.8088068374901178E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		-4.0642645973308456E-01f, 1.8389772328416343E+00f, -3.5549484953682806E+00f, 3.2273562233914270E+00f, 1.3413454081272250E-09f, -3.2273562258526494E+00f, 3.5549484959023196E+00f, -1.8389772328242200E+00f, 4.0642645973371377E-01f, 0.0000000000000000E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f };

	alignas(64) FLT c125a[] = {
		2.8923571298063562E+03f, 4.6856831608341925E+05f, 7.5304732752870023E+06f, 3.7576537584215783E+07f, 7.9591606307847857E+07f, 7.9591606307847857E+07f, 3.7576537584215745E+07f, 7.5304732752870042E+06f, 4.6856831608341780E+05f, 2.8923571298063575E+03f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		1.0919387804943191E+04f, 8.3976685277206497E+05f, 7.9494027659552367E+06f, 2.1606786285174552E+07f, 1.4625897641453246E+07f, -1.4625897641453277E+07f, -2.1606786285174549E+07f, -7.9494027659552367E+06f, -8.3976685277206241E+05f, -1.0919387804943171E+04f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		1.7418455635504150E+04f, 6.3489952164419880E+05f, 3.1358985409389879E+06f, 2.2547438801903646E+06f, -6.0429762783920728E+06f, -6.0429762783920513E+06f, 2.2547438801903692E+06f, 3.1358985409389860E+06f, 6.3489952164419706E+05f, 1.7418455635504110E+04f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		1.5396188098732160E+04f, 2.5490607173283451E+05f, 4.2818880748176615E+05f, -9.5435463094349275E+05f, -1.2004850139039254E+06f, 1.2004850139039545E+06f, 9.5435463094349345E+05f, -4.2818880748176581E+05f, -2.5490607173283395E+05f, -1.5396188098732138E+04f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		8.2616700456447434E+03f, 5.2880641964112285E+04f, -6.1165055141131161E+04f, -2.1590299490711108E+05f, 2.1595822052157650E+05f, 2.1595822052157007E+05f, -2.1590299490713840E+05f, -6.1165055141131197E+04f, 5.2880641964112183E+04f, 8.2616700456447306E+03f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		2.7267169079066489E+03f, 2.4572549134030801E+03f, -2.6065821571078384E+04f, 1.3919259807559451E+04f, 4.6802084705699206E+04f, -4.6802084705714289E+04f, -1.3919259807536537E+04f, 2.6065821571078890E+04f, -2.4572549134029036E+03f, -2.7267169079066425E+03f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		5.0402062537834070E+02f, -1.3640153425625381E+03f, -1.4063198459019245E+03f, 7.0858129627834105E+03f, -4.8375233777605163E+03f, -4.8375233777670810E+03f, 7.0858129627894641E+03f, -1.4063198459014579E+03f, -1.3640153425626913E+03f, 5.0402062537833700E+02f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		2.4199726682542348E+01f, -2.8393731159249540E+02f, 5.1652001352543709E+02f, 7.4578914842705018E+01f, -1.1556759026365337E+03f, 1.1556759026651935E+03f, -7.4578914839714216E+01f, -5.1652001352595710E+02f, 2.8393731159268043E+02f, -2.4199726682540959E+01f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		-1.0545675122360885E+01f, -3.0306758891224317E+00f, 7.2305523762173834E+01f, -1.3808908570221064E+02f, 7.6293213403386517E+01f, 7.6293213419205742E+01f, -1.3808908572505672E+02f, 7.2305523760424833E+01f, -3.0306758894244412E+00f, -1.0545675122369961E+01f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		-2.1836930570474395E+00f, 5.4992367509081630E+00f, -4.5624617253163446E+00f, -6.6492709819863256E+00f, 2.0339240341691568E+01f, -2.0339240351164950E+01f, 6.6492710020476089E+00f, 4.5624617253163446E+00f, -5.4992367508501152E+00f, 2.1836930570530630E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		-9.1748741459757727E-02f, 5.2562451739588611E-01f, -1.4144257958835973E+00f, 1.8629578990262812E+00f, -9.0169874554123419E-01f, -9.0169876258108816E-01f, 1.8629579026113960E+00f, -1.4144257947447987E+00f, 5.2562451738534777E-01f, -9.1748741464373396E-02f, 0.0000000000000000E+00f, 0.0000000000000000E+00f };

	alignas(64) FLT c125b[] = {
		4.3537972057094357E+03f, 9.8872306817881018E+05f, 2.0938056062983289E+07f, 1.3701428307175827E+08f, 3.8828289972017348E+08f, 5.4292197128519189E+08f, 3.8828289972017324E+08f, 1.3701428307175821E+08f, 2.0938056062983286E+07f, 9.8872306817881099E+05f, 4.3537972057093830E+03f, 0.0000000000000000E+00f,
		1.7371472778611496E+04f, 1.9155790709433770E+06f, 2.4914432724618733E+07f, 9.7792160665338323E+07f, 1.3126779387874992E+08f, 1.1003518489948497E-08f, -1.3126779387874992E+08f, -9.7792160665338367E+07f, -2.4914432724618725E+07f, -1.9155790709433774E+06f, -1.7371472778611387E+04f, 0.0000000000000000E+00f,
		2.9650558537745437E+04f, 1.6014973065836846E+06f, 1.1867448782239100E+07f, 2.0812212822540633E+07f, -1.1749875870571069E+07f, -4.5121922350041404E+07f, -1.1749875870571032E+07f, 2.0812212822540659E+07f, 1.1867448782239093E+07f, 1.6014973065836851E+06f, 2.9650558537745299E+04f, 0.0000000000000000E+00f,
		2.8505604980264394E+04f, 7.4166660874053277E+05f, 2.5711466441825330E+06f, -1.2146931938153899E+06f, -8.3931576510116160E+06f, -1.5221113764487218E-08f, 8.3931576510117017E+06f, 1.2146931938154220E+06f, -2.5711466441825316E+06f, -7.4166660874053324E+05f, -2.8505604980264285E+04f, 0.0000000000000000E+00f,
		1.7045632829988481E+04f, 1.9785834209758078E+05f, 8.6361403553701501E+04f, -1.0584472412326147E+06f, -1.3367486018960556E+05f, 1.7818009619467217E+06f, -1.3367486018952832E+05f, -1.0584472412326441E+06f, 8.6361403553699885E+04f, 1.9785834209758087E+05f, 1.7045632829988419E+04f, 0.0000000000000000E+00f,
		6.5462464716912918E+03f, 2.5347576368078855E+04f, -7.5810878908805942E+04f, -8.0774039751690128E+04f, 2.5492801112955116E+05f, 3.6655592491345995E-08f, -2.5492801112950110E+05f, 8.0774039751702396E+04f, 7.5810878908810162E+04f, -2.5347576368078677E+04f, -6.5462464716912700E+03f, 0.0000000000000000E+00f,
		1.5684149291082115E+03f, -1.0302687059852267E+03f, -1.3446845770824435E+04f, 2.0814393480320545E+04f, 1.4366994276523908E+04f, -4.4581342385955380E+04f, 1.4366994276463982E+04f, 2.0814393480325110E+04f, -1.3446845770824308E+04f, -1.0302687059850016E+03f, 1.5684149291082128E+03f, 0.0000000000000000E+00f,
		1.9398419323286222E+02f, -8.7329293867281388E+02f, 2.4796533428938184E+02f, 3.2905701326623416E+03f, -4.8989871768459579E+03f, 2.8861239463615327E-09f, 4.8989871768722078E+03f, -3.2905701326312101E+03f, -2.4796533429068171E+02f, 8.7329293867237629E+02f, -1.9398419323287882E+02f, 0.0000000000000000E+00f,
		-4.2288232505124679E+00f, -9.9574929618003850E+01f, 2.9563077146126534E+02f, -1.9453049352240328E+02f, -4.0107401572039475E+02f, 7.9532514195009401E+02f, -4.0107401576942334E+02f, -1.9453049354949908E+02f, 2.9563077145563869E+02f, -9.9574929618160851E+01f, -4.2288232505049734E+00f, 0.0000000000000000E+00f,
		-5.3741131162167548E+00f, 5.5350606003782072E+00f, 1.9153744596147156E+01f, -6.3189447483342484E+01f, 6.6921287710344444E+01f, 2.6543499136172006E-08f, -6.6921287588490713E+01f, 6.3189447458080132E+01f, -1.9153744593546620E+01f, -5.5350606004478644E+00f, 5.3741131162113120E+00f, 0.0000000000000000E+00f,
		-7.0359426508237854E-01f, 2.2229112757468452E+00f, -3.2054079720618520E+00f, 8.3392526913327172E-02f, 6.8879260281453520E+00f, -1.0795498333352139E+01f, 6.8879260220718077E+00f, 8.3392507342704467E-02f, -3.2054079702060019E+00f, 2.2229112757257625E+00f, -7.0359426507941902E-01f, 0.0000000000000000E+00f,
		5.2648094861126392E-02f, 9.9912561389764148E-02f, -4.3913938527232693E-01f, 7.9792987484770361E-01f, -6.9191816827427566E-01f, -1.2022534526020762E-09f, 6.9191820562024531E-01f, -7.9792984883890594E-01f, 4.3913938443394634E-01f, -9.9912561446925147E-02f, -5.2648094869462925E-02f, 0.0000000000000000E+00f };

	alignas(64) FLT c125c[] = {
		6.4299692685485315E+03f, 2.0077596413122714E+06f, 5.4904521978991628E+07f, 4.5946106674819350E+08f, 1.6835469840840104E+09f, 3.1308386544851556E+09f, 3.1308386544851556E+09f, 1.6835469840840099E+09f, 4.5946106674819458E+08f, 5.4904521978991754E+07f, 2.0077596413122730E+06f, 6.4299692685634491E+03f,
		2.6965848540274073E+04f, 4.1625245902732178E+06f, 7.2097002594596952E+07f, 3.8505085985474640E+08f, 7.9479013671674240E+08f, 4.7870231281824082E+08f, -4.7870231281824046E+08f, -7.9479013671674252E+08f, -3.8505085985474682E+08f, -7.2097002594597101E+07f, -4.1625245902732178E+06f, -2.6965848540258085E+04f,
		4.8869694409905111E+04f, 3.7863371066322513E+06f, 3.9530526716552719E+07f, 1.1475134266581042E+08f, 4.6311261797930710E+07f, -2.0442837194260675E+08f, -2.0442837194260725E+08f, 4.6311261797930680E+07f, 1.1475134266581020E+08f, 3.9530526716552787E+07f, 3.7863371066322504E+06f, 4.8869694409920470E+04f,
		5.0530564260114021E+04f, 1.9615784087727289E+06f, 1.1044597342441007E+07f, 7.9812418612436540E+06f, -3.4042228324588493E+07f, -3.3301805987927791E+07f, 3.3301805987928167E+07f, 3.4042228324588671E+07f, -7.9812418612435497E+06f, -1.1044597342440993E+07f, -1.9615784087727286E+06f, -5.0530564260099913E+04f,
		3.3081876469965493E+04f, 6.2011956881368335E+05f, 1.3086001239863748E+06f, -3.1165484297367339E+06f, -5.1982996003442882E+06f, 6.3530947749618590E+06f, 6.3530947749616513E+06f, -5.1982996003444213E+06f, -3.1165484297366543E+06f, 1.3086001239863599E+06f, 6.2011956881368288E+05f, 3.3081876469981333E+04f,
		1.4308966168506788E+04f, 1.1375573205951916E+05f, -1.0318195403424598E+05f, -6.6892418721462542E+05f, 5.9223570255461533E+05f, 1.1093685152673351E+06f, -1.1093685152666988E+06f, -5.9223570255418238E+05f, 6.6892418721489178E+05f, 1.0318195403424004E+05f, -1.1375573205951886E+05f, -1.4308966168492358E+04f,
		4.0848961919700960E+03f, 7.5033277163528910E+03f, -5.2578904182711594E+04f, 6.3431596329919275E+03f, 1.5984798504282799E+05f, -1.2521363434070408E+05f, -1.2521363434057294E+05f, 1.5984798504289921E+05f, 6.3431596327853522E+03f, -5.2578904182714803E+04f, 7.5033277163530738E+03f, 4.0848961919843541E+03f,
		7.1658797373677544E+02f, -1.5499947984100402E+03f, -4.5490740453241297E+03f, 1.4520122796414065E+04f, -3.7896465826366048E+03f, -2.3597107892645658E+04f, 2.3597107892708405E+04f, 3.7896465828577311E+03f, -1.4520122796272850E+04f, 4.5490740453326107E+03f, 1.5499947984094520E+03f, -7.1658797372277388E+02f,
		5.2022749592533359E+01f, -4.0624258132650436E+02f, 5.2256582980122801E+02f, 9.3282469962834807E+02f, -2.8710622267611107E+03f, 1.7594166903207245E+03f, 1.7594166904840572E+03f, -2.8710622269566602E+03f, 9.3282469973848731E+02f, 5.2256582976889342E+02f, -4.0624258132718376E+02f, 5.2022749606062760E+01f,
		-7.0341875498860729E+00f, -2.3043166229077922E+01f, 1.2279331781679724E+02f, -1.6714687548507158E+02f, -4.4746498424591195E+01f, 3.6060906024962412E+02f, -3.6060905985137049E+02f, 4.4746498852565225E+01f, 1.6714687549695972E+02f, -1.2279331779599295E+02f, 2.3043166228938606E+01f, 7.0341875614861786E+00f,
		-2.1556100132617875E+00f, 4.1361104009993737E+00f, 1.8107701723532290E+00f, -2.1223400322208619E+01f, 3.5820961861882218E+01f, -1.8782945665578143E+01f, -1.8782945409136026E+01f, 3.5820961915195049E+01f, -2.1223400242576908E+01f, 1.8107701298380314E+00f, 4.1361104007462801E+00f, -2.1556100021452793E+00f,
		-1.1440899376747954E-01f, 7.0567641591060326E-01f, -1.4530217904770133E+00f, 1.0571984613482723E+00f, 1.4389002957406878E+00f, -4.2241732762744180E+00f, 4.2241733421252539E+00f, -1.4389000664821670E+00f, -1.0571984509828731E+00f, 1.4530218285851431E+00f, -7.0567641613924970E-01f, 1.1440900438178304E-01f,
		-1.4486009663463860E-02f, 2.9387825785034223E-03f, -1.0265969715607470E-01f, 2.6748267835596640E-01f, -3.3606430399849180E-01f, 1.5850148085005597E-01f, 1.5850183161365292E-01f, -3.3606448814949358E-01f, 2.6748281866164947E-01f, -1.0265975004478733E-01f, 2.9387817050372631E-03f, -1.4486000369842612E-02f };

	alignas(64) FLT c125d[] = {
		9.3397060605267689E+03f, 3.9447202186643109E+06f, 1.3701428307175812E+08f, 1.4375660883001409E+09f, 6.6384519128895693E+09f, 1.5848048271166529E+10f, 2.1031560281976665E+10f, 1.5848048271166502E+10f, 6.6384519128895674E+09f, 1.4375660883001378E+09f, 1.3701428307175812E+08f, 3.9447202186642843E+06f, 9.3397060605268125E+03f, 0.0000000000000000E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		4.0984512931817764E+04f, 8.6828943763566799E+06f, 1.9558432133067656E+08f, 1.3674961320373521E+09f, 3.9251291128182430E+09f, 4.5116631434426517E+09f, 4.8375356630808043E-07f, -4.5116631434426460E+09f, -3.9251291128182402E+09f, -1.3674961320373492E+09f, -1.9558432133067656E+08f, -8.6828943763566278E+06f, -4.0984512931817771E+04f, 0.0000000000000000E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		7.8379538318778985E+04f, 8.4928073133582603E+06f, 1.1992091153966437E+08f, 5.0561697705436689E+08f, 6.1845897311593950E+08f, -5.1306326495404470E+08f, -1.4790096327029374E+09f, -5.1306326495404077E+08f, 6.1845897311593986E+08f, 5.0561697705436659E+08f, 1.1992091153966436E+08f, 8.4928073133582156E+06f, 7.8379538318778927E+04f, 0.0000000000000000E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		8.6417670227040013E+04f, 4.8250267333349697E+06f, 3.9836803808039002E+07f, 7.5026052902191013E+07f, -7.7565422849560052E+07f, -2.5393835488011825E+08f, 5.1202971235247489E-07f, 2.5393835488012013E+08f, 7.7565422849558711E+07f, -7.5026052902191967E+07f, -3.9836803808039002E+07f, -4.8250267333349511E+06f, -8.6417670227039998E+04f, 0.0000000000000000E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		6.1161604972829380E+04f, 1.7331203720075535E+06f, 7.0216196997558968E+06f, -3.6027138646117523E+06f, -3.1775875626364492E+07f, 1.6544480876790185E+06f, 4.9816566960114852E+07f, 1.6544480876808946E+06f, -3.1775875626363728E+07f, -3.6027138646113039E+06f, 7.0216196997558847E+06f, 1.7331203720075490E+06f, 6.1161604972829351E+04f, 0.0000000000000000E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		2.9177164557155938E+04f, 3.9318079134661221E+05f, 3.1307448297760956E+05f, -2.7571366584957433E+06f, -9.8421840747392306E+05f, 6.8469173866731795E+06f, 2.9232946975263515E-06f, -6.8469173866698397E+06f, 9.8421840747792379E+05f, 2.7571366584955421E+06f, -3.1307448297758284E+05f, -3.9318079134660971E+05f, -2.9177164557155946E+04f, 0.0000000000000000E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		9.5097815505886610E+03f, 4.8799940773716655E+04f, -1.2734023162441862E+05f, -2.5472337176564379E+05f, 6.3596049196278059E+05f, 2.2361868201841635E+05f, -1.0716559939651759E+06f, 2.2361868202218774E+05f, 6.3596049196161982E+05f, -2.5472337176485342E+05f, -1.2734023162441724E+05f, 4.8799940773713337E+04f, 9.5097815505886447E+03f, 0.0000000000000000E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		2.0601715730545379E+03f, 1.9365931141472569E+02f, -2.5304303117518622E+04f, 2.9151392447034210E+04f, 5.9055020355306144E+04f, -1.1784846181665688E+05f, 1.1400011168699383E-06f, 1.1784846181507374E+05f, -5.9055020356297522E+04f, -2.9151392447480976E+04f, 2.5304303117520958E+04f, -1.9365931141621550E+02f, -2.0601715730545466E+03f, 0.0000000000000000E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		2.5975061893404052E+02f, -1.0025387650583972E+03f, -6.8642481194759603E+02f, 6.7515314205452096E+03f, -7.0772939650079616E+03f, -6.5444514139847633E+03f, 1.6566898963381227E+04f, -6.5444514164662887E+03f, -7.0772939638053231E+03f, 6.7515314202341915E+03f, -6.8642481198706810E+02f, -1.0025387650556635E+03f, 2.5975061893403893E+02f, 0.0000000000000000E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		5.8705282128634133E+00f, -1.4424362302822419E+02f, 3.3390627215295177E+02f, 4.8151337640374301E+01f, -1.1431733953039347E+03f, 1.4557114789663567E+03f, 1.9301282133401762E-06f, -1.4557114797747520E+03f, 1.1431733969207255E+03f, -4.8151337212400264E+01f, -3.3390627213809154E+02f, 1.4424362302302313E+02f, -5.8705282128808269E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		-4.0954969508936898E+00f, -1.2634947188543673E+00f, 3.8134139835466350E+01f, -8.4115524781317148E+01f, 4.2766848228448069E+01f, 1.0573434411021174E+02f, -1.9636661067694894E+02f, 1.0573435394677749E+02f, 4.2766846813968300E+01f, -8.4115525213218916E+01f, 3.8134139824669184E+01f, -1.2634947158177201E+00f, -4.0954969509055461E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		-6.2702735486285888E-01f, 1.8595467772479546E+00f, -1.3027978470952948E+00f, -4.9265265903267785E+00f, 1.3906831953385087E+01f, -1.3753762586104637E+01f, 1.0604155239584518E-06f, 1.3753756761963198E+01f, -1.3906831509501583E+01f, 4.9265273268806409E+00f, 1.3027978586801867E+00f, -1.8595467797630916E+00f, 6.2702735486047489E-01f, 0.0000000000000000E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		-4.8290636703364975E-02f, 1.7531876505199090E-01f, -5.0041292774701596E-01f, 6.3665145473474949E-01f, -1.2476811514471326E-02f, -1.2061603189510861E+00f, 1.8595308638696268E+00f, -1.2061633355215959E+00f, -1.2475969680262359E-02f, 6.3665088474340670E-01f, -5.0041295405456876E-01f, 1.7531876799797264E-01f, -4.8290636708721864E-02f, 0.0000000000000000E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		2.2894665617766322E-02f, -7.1358257229878720E-03f, -1.4950743217821900E-02f, 7.0611745711086651E-02f, -1.2311302279978055E-01f, 1.0342573392772816E-01f, 5.7346192890547669E-07f, -1.0342709034448951E-01f, 1.2311300937219723E-01f, -7.0611830251417942E-02f, 1.4950741891648016E-02f, 7.1358203725587141E-03f, -2.2894665628191136E-02f, 0.0000000000000000E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f };

	alignas(64) FLT c125e[] = {
		1.3368785683552904E+04f, 7.5304732752870144E+06f, 3.2765764524434990E+08f, 4.2418096936485257E+09f, 2.4197690538177525E+10f, 7.2227640697189651E+10f, 1.2261475327356714E+11f, 1.2261475327356711E+11f, 7.2227640697189682E+10f, 2.4197690538177582E+10f, 4.2418096936485257E+09f, 3.2765764524435169E+08f, 7.5304732752870200E+06f, 1.3368785683578039E+04f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		6.1154444023081669E+04f, 1.7488686085101541E+07f, 5.0279014009863263E+08f, 4.4777867842655849E+09f, 1.6916819861812059E+10f, 2.8971884004562843E+10f, 1.6054555293734524E+10f, -1.6054555293734529E+10f, -2.8971884004562843E+10f, -1.6916819861812090E+10f, -4.4777867842655830E+09f, -5.0279014009863406E+08f, -1.7488686085101560E+07f, -6.1154444023056145E+04f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		1.2279790808348049E+05f, 1.8230319600271538E+07f, 3.3815815633683985E+08f, 1.9369899011251254E+09f, 3.9743454154781203E+09f, 7.4954544638351786E+08f, -7.0173920607395000E+09f, -7.0173920607395000E+09f, 7.4954544638351130E+08f, 3.9743454154781117E+09f, 1.9369899011251252E+09f, 3.3815815633684093E+08f, 1.8230319600271557E+07f, 1.2279790808350699E+05f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		1.4339321200624766E+05f, 1.1200899688172188E+07f, 1.2799140125169712E+08f, 4.0176966726270604E+08f, 7.9146174555810899E+07f, -1.1719748245183561E+09f, -9.6919138198233843E+08f, 9.6919138198235476E+08f, 1.1719748245183618E+09f, -7.9146174555819452E+07f, -4.0176966726270568E+08f, -1.2799140125169776E+08f, -1.1200899688172201E+07f, -1.4339321200622554E+05f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		1.0866548538632700E+05f, 4.4565213401510641E+06f, 2.8354150929531462E+07f, 2.2805067924009934E+07f, -1.2058223609889300E+08f, -1.2775415620368913E+08f, 1.9261201640091014E+08f, 1.9261201640090343E+08f, -1.2775415620368628E+08f, -1.2058223609888241E+08f, 2.2805067924009915E+07f, 2.8354150929531943E+07f, 4.4565213401510660E+06f, 1.0866548538635390E+05f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		5.6346565047794407E+04f, 1.1743908345502375E+06f, 3.0601086667309003E+06f, -7.2274020134796975E+06f, -1.6220595157143334E+07f, 2.0773587344466623E+07f, 2.8183198298701070E+07f, -2.8183198298682313E+07f, -2.0773587344454899E+07f, 1.6220595157147046E+07f, 7.2274020134809064E+06f, -3.0601086667310768E+06f, -1.1743908345502312E+06f, -5.6346565047771022E+04f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		2.0435142564639598E+04f, 1.9450977300078847E+05f, -1.1234667576926883E+05f, -1.5205767549240857E+06f, 1.0515640561047094E+06f, 3.7458351782500809E+06f, -3.3794074240119159E+06f, -3.3794074240111569E+06f, 3.7458351782506104E+06f, 1.0515640561079446E+06f, -1.5205767549239916E+06f, -1.1234667576914738E+05f, 1.9450977300078212E+05f, 2.0435142564663307E+04f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		5.1491366053560478E+03f, 1.4735748500440239E+04f, -8.1689482343683034E+04f, -3.5176894225644079E+04f, 3.7034248410400847E+05f, -1.9109669530460562E+05f, -5.2637978465735121E+05f, 5.2637978465564619E+05f, 1.9109669530912716E+05f, -3.7034248412078863E+05f, 3.5176894225852200E+04f, 8.1689482343699274E+04f, -1.4735748500439855E+04f, -5.1491366053330485E+03f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		8.5138795113645585E+02f, -1.2978618911733427E+03f, -8.7500873646623440E+03f, 2.1319159613970569E+04f, 7.6586611605801199E+03f, -6.2424139811455236E+04f, 4.2620771487921840E+04f, 4.2620771491440872E+04f, -6.2424139815176597E+04f, 7.6586611693937375E+03f, 2.1319159613447209E+04f, -8.7500873648877496E+03f, -1.2978618911701635E+03f, 8.5138795115875257E+02f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		7.2176142041616245E+01f, -4.5543406155008586E+02f, 2.8301959891624585E+02f, 2.1994171513769957E+03f, -4.5082500677203352E+03f, 4.7658016853354945E+02f, 7.1044827209848581E+03f, -7.1044827023442112E+03f, -4.7658015978385805E+02f, 4.5082500694322307E+03f, -2.1994171506161529E+03f, -2.8301959873197922E+02f, 4.5543406154525627E+02f, -7.2176142022451799E+01f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		-3.1135380163286266E+00f, -3.8554406982628045E+01f, 1.4396028111579378E+02f, -1.1260050352192819E+02f, -3.0073665460436297E+02f, 7.2079162225452933E+02f, -4.1195308319958349E+02f, -4.1195308907344031E+02f, 7.2079162228692246E+02f, -3.0073665296314113E+02f, -1.1260050391063737E+02f, 1.4396028095922969E+02f, -3.8554406981953719E+01f, -3.1135379980309104E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		-1.6022934776950781E+00f, 1.8678197421257499E+00f, 8.3368944138930576E+00f, -3.0791578217513287E+01f, 3.4749712345962102E+01f, 1.2322522680262193E+01f, -7.3924006859338746E+01f, 7.3924005395986399E+01f, -1.2322518095091780E+01f, -3.4749717239655702E+01f, 3.0791578812609753E+01f, -8.3368942651188451E+00f, -1.8678197375527952E+00f, 1.6022934952009980E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		-1.9362061840948824E-01f, 6.3024467669748396E-01f, -9.3262278519229969E-01f, -4.8908749318740480E-01f, 4.0479376609320967E+00f, -6.2829712900962678E+00f, 3.1767825933699174E+00f, 3.1767865219197975E+00f, -6.2829777441520323E+00f, 4.0479394849078085E+00f, -4.8908801933495105E-01f, -9.3262306580362497E-01f, 6.3024467258732675E-01f, -1.9362060312142931E-01f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		1.8785913718903639E-02f, 3.1605271252714680E-02f, -1.3655798291459853E-01f, 2.5016547139148904E-01f, -1.6654308552073466E-01f, -2.1682598043284024E-01f, 6.1786085249849709E-01f, -6.1785470804340159E-01f, 2.1682794765059335E-01f, 1.6654258378326353E-01f, -2.5016523395036322E-01f, 1.3655803190024704E-01f, -3.1605272440421092E-02f, -1.8785905282938619E-02f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		-1.2896545140952162E-02f, -3.7106972352948116E-03f, 5.8857860695711909E-04f, 1.3987176343065890E-02f, -3.5714007561179102E-02f, 4.3401590960273219E-02f, -2.0034532372716081E-02f, -2.0038454375630149E-02f, 4.3401322628411031E-02f, -3.5713348533616053E-02f, 1.3987046090052241E-02f, 5.8856319054218355E-04f, -3.7106979912720915E-03f, -1.2896537385752806E-02f, 0.0000000000000000E+00f, 0.0000000000000000E+00f };

	alignas(64) FLT c125f[] = {
		1.8887777774374499E+04f, 1.4015330434461417E+07f, 7.5498683300180018E+08f, 1.1900937739619951E+10f, 8.2530965279375351E+10f, 3.0178246269069604E+11f, 6.3775691457119104E+11f, 8.1471473119305554E+11f, 6.3775691457119116E+11f, 3.0178246269069641E+11f, 8.2530965279375519E+10f, 1.1900937739619963E+10f, 7.5498683300180054E+08f, 1.4015330434461435E+07f, 1.8887777774374488E+04f, 0.0000000000000000E+00f,
		8.9780907163796335E+04f, 3.4167636285297148E+07f, 1.2346880033823481E+09f, 1.3719272724135921E+10f, 6.5858241494816696E+10f, 1.5266999939989539E+11f, 1.5687794513790723E+11f, -2.8523584844088883E-05f, -1.5687794513790732E+11f, -1.5266999939989545E+11f, -6.5858241494816811E+10f, -1.3719272724135933E+10f, -1.2346880033823476E+09f, -3.4167636285297163E+07f, -8.9780907163796335E+04f, 0.0000000000000000E+00f,
		1.8850321233130712E+05f, 3.7693640983013541E+07f, 8.9846818051570034E+08f, 6.7094088040439653E+09f, 1.9743296615199215E+10f, 1.8072727219391140E+10f, -2.0634615374559410E+10f, -4.9654335197177498E+10f, -2.0634615374559414E+10f, 1.8072727219391048E+10f, 1.9743296615199223E+10f, 6.7094088040439672E+09f, 8.9846818051570022E+08f, 3.7693640983013526E+07f, 1.8850321233130703E+05f, 0.0000000000000000E+00f,
		2.3185006533495727E+05f, 2.4789475362741601E+07f, 3.7751696829092383E+08f, 1.7167916788178182E+09f, 1.9832401267745295E+09f, -3.4881359830884194E+09f, -7.8785602379628601E+09f, 6.6906528952995499E-05f, 7.8785602379629536E+09f, 3.4881359830884261E+09f, -1.9832401267745163E+09f, -1.7167916788178096E+09f, -3.7751696829092425E+08f, -2.4789475362741597E+07f, -2.3185006533495730E+05f, 0.0000000000000000E+00f,
		1.8672970114818285E+05f, 1.0741068109706732E+07f, 9.8017949708492473E+07f, 2.0291084954252145E+08f, -2.7857869294214898E+08f, -9.4112677968756318E+08f, 1.7886520649334356E+08f, 1.4579673547891481E+09f, 1.7886520649344125E+08f, -9.4112677968753338E+08f, -2.7857869294217581E+08f, 2.0291084954251301E+08f, 9.8017949708492488E+07f, 1.0741068109706739E+07f, 1.8672970114818282E+05f, 0.0000000000000000E+00f,
		1.0411891611891470E+05f, 3.1771463075269456E+06f, 1.4880104152842037E+07f, -6.8136965447538150E+06f, -8.7072998215422541E+07f, 1.8024116530863210E+06f, 1.9067730799615666E+08f, 1.2078175959365315E-04f, -1.9067730799603686E+08f, -1.8024116529155241E+06f, 8.7072998215445980E+07f, 6.8136965447565373E+06f, -1.4880104152841812E+07f, -3.1771463075269484E+06f, -1.0411891611891470E+05f, 0.0000000000000000E+00f,
		4.1300641422694731E+04f, 6.3217168592497683E+05f, 7.7343707634845132E+05f, -5.4575962381476769E+06f, -3.7387211063063843E+06f, 1.8451583614082869E+07f, 3.0480804948189310E+06f, -2.7500445095872246E+07f, 3.0480804948457484E+06f, 1.8451583614064269E+07f, -3.7387211062890980E+06f, -5.4575962381450543E+06f, 7.7343707634841127E+05f, 6.3217168592497602E+05f, 4.1300641422694724E+04f, 0.0000000000000000E+00f,
		1.1710443348523711E+04f, 7.5405449195716908E+04f, -1.6634736996487752E+05f, -5.6069290801842115E+05f, 1.1540571563940533E+06f, 1.0209821660925965E+06f, -2.9641921942009293E+06f, -7.3770236318814628E-06f, 2.9641921942630685E+06f, -1.0209821662946860E+06f, -1.1540571563987043E+06f, 5.6069290801928868E+05f, 1.6634736996459437E+05f, -7.5405449195719295E+04f, -1.1710443348523739E+04f, 0.0000000000000000E+00f,
		2.3142324239350210E+03f, 2.1710560541703007E+03f, -3.6929625713151705E+04f, 2.6143898219588682E+04f, 1.4046980090353978E+05f, -2.1033190114896413E+05f, -1.1132269819276403E+05f, 3.7491447373940505E+05f, -1.1132269820720138E+05f, -2.1033190120894444E+05f, 1.4046980085134835E+05f, 2.6143898217223435E+04f, -3.6929625713258414E+04f, 2.1710560541651053E+03f, 2.3142324239349791E+03f, 0.0000000000000000E+00f,
		2.8879718294281940E+02f, -9.2801372612866078E+02f, -1.9817144428357562E+03f, 9.9004179214302640E+03f, -5.7928268996319048E+03f, -2.1083466266548403E+04f, 3.3285502001854453E+04f, 1.3615676123196788E-04f, -3.3285501884684672E+04f, 2.1083466388283239E+04f, 5.7928269528908959E+03f, -9.9004179214302640E+03f, 1.9817144428357562E+03f, 9.2801372612624596E+02f, -2.8879718294281940E+02f, 0.0000000000000000E+00f,
		1.3121871131759899E+01f, -1.5978845118014243E+02f, 2.7429718889479011E+02f, 4.4598059431432415E+02f, -1.8917609556521720E+03f, 1.5303002256342920E+03f, 1.7542368404254241E+03f, -3.9411530187890685E+03f, 1.7542368839611659E+03f, 1.5303002335812619E+03f, -1.8917609760379448E+03f, 4.4598059250034765E+02f, 2.7429718872202716E+02f, -1.5978845118149314E+02f, 1.3121871131760223E+01f, 0.0000000000000000E+00f,
		-2.4286151057622600E+00f, -6.7839829150137421E+00f, 4.6999223003107119E+01f, -7.4896070454665107E+01f, -3.2010110856873055E+01f, 2.5022929107925501E+02f, -2.8786053481345135E+02f, 1.4424367379967129E-05f, 2.8786057555317575E+02f, -2.5022937123192844E+02f, 3.2010139421505684E+01f, 7.4896073537460509E+01f, -4.6999223012862650E+01f, 6.7839829186720362E+00f, 2.4286151057336860E+00f, 0.0000000000000000E+00f,
		-5.4810555665671257E-01f, 1.1436870859674571E+00f, 8.2471504792547190E-01f, -8.5602131787584241E+00f, 1.5631631237511966E+01f, -6.4979395997142886E+00f, -1.8737629118679905E+01f, 3.3283673647767003E+01f, -1.8737705444926284E+01f, -6.4980552114725620E+00f, 1.5631576798962341E+01f, -8.5602158445716778E+00f, 8.2471481116140977E-01f, 1.1436870769250529E+00f, -5.4810555667406624E-01f, 0.0000000000000000E+00f,
		-1.4554612891837512E-02f, 1.7022157398269799E-01f, -3.7563892964814216E-01f, 2.0131145240492249E-01f, 8.3554123561642435E-01f, -2.1191317631421946E+00f, 1.9961007770939201E+00f, 5.0230495487029605E-05f, -1.9960655197919825E+00f, 2.1191435815870405E+00f, -8.3552330614378623E-01f, -2.0131363341395125E-01f, 3.7563890238546094E-01f, -1.7022157734534860E-01f, 1.4554612875194470E-02f, 0.0000000000000000E+00f,
		-1.2348455978815665E-02f, 2.6143485494326945E-03f, -2.9252290291144727E-02f, 7.5392101552106419E-02f, -8.7986538697867239E-02f, 1.3073120666751545E-03f, 1.5251801232957554E-01f, -2.3235618419546245E-01f, 1.5253703942622115E-01f, 1.3217162898956957E-03f, -8.7999818995735196E-02f, 7.5391507930594778E-02f, -2.9252395603998178E-02f, 2.6143483927929994E-03f, -1.2348455970768767E-02f, 0.0000000000000000E+00f,
		1.4214685591273772E-02f, -1.2364346992375923E-03f, 1.2892328724708124E-03f, 1.6178725688327468E-03f, -8.2104229475896996E-03f, 1.3914679473447157E-02f, -1.1426959041713501E-02f, 1.6590583007947697E-05f, 1.1446333966460217E-02f, -1.3912124902889801E-02f, 8.2298310485774198E-03f, -1.6155336438419190E-03f, -1.2892162843503102E-03f, 1.2364372911314208E-03f, -1.4214685607473108E-02f, 0.0000000000000000E+00f };

	alignas(64) FLT c125g[] = {
		2.6374086784014689E+04f, 2.5501413681212645E+07f, 1.6835469840840099E+09f, 3.1953580806547867E+10f, 2.6584910126662766E+11f, 1.1715858191494619E+12f, 3.0181658330343120E+12f, 4.7888775408612773E+12f, 4.7888775408612764E+12f, 3.0181658330343125E+12f, 1.1715858191494619E+12f, 2.6584910126662772E+11f, 3.1953580806547874E+10f, 1.6835469840840104E+09f, 2.5501413681212656E+07f, 2.6374086784014886E+04f,
		1.2991568388123445E+05f, 6.4986154651133664E+07f, 2.9142305012947259E+09f, 3.9748054433728149E+10f, 2.3649443248440247E+11f, 7.0471088240421252E+11f, 1.0533888905987031E+12f, 5.4832304482297632E+11f, -5.4832304482297687E+11f, -1.0533888905987034E+12f, -7.0471088240421265E+11f, -2.3649443248440250E+11f, -3.9748054433728149E+10f, -2.9142305012947259E+09f, -6.4986154651133649E+07f, -1.2991568388123448E+05f,
		2.8421223836872831E+05f, 7.5448503558118582E+07f, 2.2710828032883868E+09f, 2.1491603403163826E+10f, 8.4299374042308136E+10f, 1.3384457365769528E+11f, 1.8630012765531485E+09f, -2.4384536789321179E+11f, -2.4384536789321094E+11f, 1.8630012765532806E+09f, 1.3384457365769531E+11f, 8.4299374042308090E+10f, 2.1491603403163826E+10f, 2.2710828032883863E+09f, 7.5448503558118552E+07f, 2.8421223836872820E+05f,
		3.6653021243297518E+05f, 5.2693428548387080E+07f, 1.0410094433021281E+09f, 6.3986267576853533E+09f, 1.3313926739756302E+10f, -2.7909761561128025E+09f, -3.9911638977027977E+10f, -2.9236947704012939E+10f, 2.9236947704012939E+10f, 3.9911638977028267E+10f, 2.7909761561128430E+09f, -1.3313926739756279E+10f, -6.3986267576853561E+09f, -1.0410094433021276E+09f, -5.2693428548387088E+07f, -3.6653021243297518E+05f,
		3.1185660915838118E+05f, 2.4564274645530280E+07f, 3.0509279143241835E+08f, 1.0432225146182569E+09f, 6.4966284440222360E+07f, -4.2483903608016477E+09f, -3.1778261722524829E+09f, 5.9880587942832708E+09f, 5.9880587942832832E+09f, -3.1778261722526174E+09f, -4.2483903608017979E+09f, 6.4966284440235756E+07f, 1.0432225146182607E+09f, 3.0509279143241805E+08f, 2.4564274645530272E+07f, 3.1185660915838124E+05f,
		1.8544733523229562E+05f, 7.9824949938292839E+06f, 5.6880943382648192E+07f, 5.4097201999258779E+07f, -3.0776449202833223E+08f, -3.7659931821867347E+08f, 6.8797698944719648E+08f, 7.5429896889866996E+08f, -7.5429896889781320E+08f, -6.8797698944658160E+08f, 3.7659931821898031E+08f, 3.0776449202837497E+08f, -5.4097201999252096E+07f, -5.6880943382647842E+07f, -7.9824949938292857E+06f, -1.8544733523229562E+05f,
		7.9472339236673259E+04f, 1.8159676553648398E+06f, 5.7259818806751696E+06f, -1.2786136236423338E+07f, -3.8677490873147681E+07f, 4.7651450515707508E+07f, 9.0723760109202415E+07f, -9.4532949239946112E+07f, -9.4532949239604995E+07f, 9.0723760109522834E+07f, 4.7651450515667401E+07f, -3.8677490873160362E+07f, -1.2786136236416934E+07f, 5.7259818806752721E+06f, 1.8159676553648538E+06f, 7.9472339236673215E+04f,
		2.4831718998299857E+04f, 2.7536301841716090E+05f, -5.1045953356025166E+04f, -2.6996387880239477E+06f, 1.1656554632125401E+06f, 9.1521923449522462E+06f, -6.8198180925621921E+06f, -1.2555197000954127E+07f, 1.2555197001087580E+07f, 6.8198180925775450E+06f, -9.1521923449367471E+06f, -1.1656554632051867E+06f, 2.6996387880183556E+06f, 5.1045953355832869E+04f, -2.7536301841717580E+05f, -2.4831718998299897E+04f,
		5.6060763597396035E+03f, 2.2154740880101843E+04f, -1.0243462874810334E+05f, -1.1802198892388590E+05f, 6.4061699367506150E+05f, -1.1166716749369531E+05f, -1.4153578101923370E+06f, 1.0790712965214122E+06f, 1.0790712965802078E+06f, -1.4153578102569627E+06f, -1.1166716767280686E+05f, 6.4061699367841065E+05f, -1.1802198892652121E+05f, -1.0243462874831920E+05f, 2.2154740880096295E+04f, 5.6060763597396262E+03f,
		8.7271993222049730E+02f, -7.0074676859193858E+02f, -1.2528372958474913E+04f, 2.3643101054370443E+04f, 3.1699060146436736E+04f, -1.1270133578294520E+05f, 3.6872846840416030E+04f, 1.5168911768972370E+05f, -1.5168911672801850E+05f, -3.6872846329129716E+04f, 1.1270133600206790E+05f, -3.1699060140349993E+04f, -2.3643101053229180E+04f, 1.2528372958403583E+04f, 7.0074676858840917E+02f, -8.7271993222049730E+02f,
		7.8842259458727298E+01f, -4.2070880913717718E+02f, -1.0535142166729695E+02f, 3.3375056757602101E+03f, -4.9426353709826744E+03f, -3.6567309465694352E+03f, 1.5199085032737788E+04f, -9.4972226150681072E+03f, -9.4972224492176338E+03f, 1.5199085307902486E+04f, -3.6567309714471071E+03f, -4.9426353751288962E+03f, 3.3375056795609726E+03f, -1.0535142205602271E+02f, -4.2070880913447866E+02f, 7.8842259458701932E+01f,
		8.9833076760252317E-02f, -4.4163371177310189E+01f, 1.2880771175011134E+02f, 2.8722208980881483E+00f, -5.7164632401064989E+02f, 9.0417621054583299E+02f, 1.1221311957018894E+00f, -1.4190922684153286E+03f, 1.4190926436578332E+03f, -1.1219382673482139E+00f, -9.0417616902565715E+02f, 5.7164633587355513E+02f, -2.8722219907225899E+00f, -1.2880771149646372E+02f, 4.4163371174871045E+01f, -8.9833076793553943E-02f,
		-1.0900468357304585E+00f, -1.1264666580175993E-01f, 1.1810668498718398E+01f, -3.0289105594116332E+01f, 1.5494599855921946E+01f, 6.0130016326899806E+01f, -1.2330195579557967E+02f, 6.7114292010484860E+01f, 6.7114238133033894E+01f, -1.2330200967294053E+02f, 6.0129899592769000E+01f, 1.5494588631452897E+01f, -3.0289108821162568E+01f, 1.1810668060273379E+01f, -1.1264668224327026E-01f, -1.0900468357482698E+00f,
		-1.1763610124684608E-01f, 4.2939195551308978E-01f, -2.7950231695310290E-01f, -1.7354597875532083E+00f, 5.1181749794184972E+00f, -5.0538409872852545E+00f, -2.1268758321444312E+00f, 1.0709572497394593E+01f, -1.0709247944735344E+01f, 2.1270284132327628E+00f, 5.0538814533614023E+00f, -5.1181783143082038E+00f, 1.7354587260576941E+00f, 2.7950208340719496E-01f, -4.2939195720020440E-01f, 1.1763610121354666E-01f,
		-1.8020499708490779E-02f, 3.6694576081450124E-02f, -1.1331174689418615E-01f, 1.3970801507325420E-01f, 8.1708800731612838E-02f, -5.4465632012605969E-01f, 7.9628723318194716E-01f, -3.9045387765910361E-01f, -3.9034731591396871E-01f, 7.9641679205120786E-01f, -5.4465236519348836E-01f, 8.1709687544577886E-02f, 1.3970913694934384E-01f, -1.1331198385459386E-01f, 3.6694575058947500E-02f, -1.8020499699434717E-02f,
		1.4589783457723899E-02f, -7.8885273589694921E-04f, -4.4854775481901451E-03f, 1.8117810622567232E-02f, -3.0563678378015532E-02f, 1.9027105036022670E-02f, 2.4778670881552757E-02f, -6.7767913155521747E-02f, 6.7979444868167399E-02f, -2.4638534439549119E-02f, -1.8992900331546877E-02f, 3.0569915511324409E-02f, -1.8117279802711158E-02f, 4.4857097818771776E-03f, 7.8885377265448060E-04f, -1.4589783469873403E-02f,
		-1.0467998068898355E-02f, -3.2140568385029999E-04f, 5.2979866592800886E-04f, -1.5800624712947203E-04f, -1.4200041949817279E-03f, 3.7626007108648857E-03f, -3.8348321381240775E-03f, 1.6547563335740942E-03f, 1.5759584129276946E-03f, -3.8873640852216617E-03f, 3.7166352571544989E-03f, -1.4265706883689335E-03f, -1.5923746463956793E-04f, 5.2952292450647511E-04f, -3.2141610431099765E-04f, -1.0467998084554094E-02f };

	alignas(64) FLT c2002[] = {
		4.5147043243215315E+01f, 4.5147043243215300E+01f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		5.7408070938221300E+01f, -5.7408070938221293E+01f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		-1.8395117920046484E+00f, -1.8395117920046560E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		-2.0382426253182082E+01f, 2.0382426253182086E+01f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		-2.0940804433577420E+00f, -2.0940804433577389E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f };

	alignas(64) FLT c2003[] = {
		1.5653991189315119E+02f, 8.8006872410780295E+02f, 1.5653991189967152E+02f, 0.0000000000000000E+00f,
		3.1653018869611077E+02f, 7.4325702843759617E-14f, -3.1653018868907071E+02f, 0.0000000000000000E+00f,
		1.7742692790454484E+02f, -3.3149255274727801E+02f, 1.7742692791117119E+02f, 0.0000000000000000E+00f,
		-1.5357716116473156E+01f, 9.5071486252033243E-15f, 1.5357716122720193E+01f, 0.0000000000000000E+00f,
		-3.7757583061523668E+01f, 5.3222970968867315E+01f, -3.7757583054647384E+01f, 0.0000000000000000E+00f,
		-3.9654011076088804E+00f, 1.8062124448285358E-13f, 3.9654011139270540E+00f, 0.0000000000000000E+00f };

	alignas(64) FLT c2004[] = {
		5.4284366850213200E+02f, 1.0073871433088398E+04f, 1.0073871433088396E+04f, 5.4284366850213223E+02f,
		1.4650917259256939E+03f, 6.1905285583602863E+03f, -6.1905285583602881E+03f, -1.4650917259256937E+03f,
		1.4186910680718345E+03f, -1.3995339862725591E+03f, -1.3995339862725598E+03f, 1.4186910680718347E+03f,
		5.1133995502497419E+02f, -1.4191608683682996E+03f, 1.4191608683682998E+03f, -5.1133995502497424E+02f,
		-4.8293622641174039E+01f, 3.9393732546135226E+01f, 3.9393732546135816E+01f, -4.8293622641174061E+01f,
		-7.8386867802392288E+01f, 1.4918904800408930E+02f, -1.4918904800408751E+02f, 7.8386867802392359E+01f,
		-1.0039212571700894E+01f, 5.0626747735616746E+00f, 5.0626747735625512E+00f, -1.0039212571700640E+01f };

	alignas(64) FLT c2005[] = {
		9.9223677575398392E+02f, 3.7794697666613320E+04f, 9.8715771010760494E+04f, 3.7794697666613283E+04f, 9.9223677575398403E+02f, 0.0000000000000000E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		3.0430174925083825E+03f, 3.7938404259811403E+04f, -1.1842989705877139E-11f, -3.7938404259811381E+04f, -3.0430174925083829E+03f, 0.0000000000000000E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		3.6092689177271222E+03f, 7.7501368899498666E+03f, -2.2704627332475000E+04f, 7.7501368899498730E+03f, 3.6092689177271218E+03f, 0.0000000000000000E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		1.9990077310495396E+03f, -3.8875294641277296E+03f, 9.7116927320010791E-12f, 3.8875294641277369E+03f, -1.9990077310495412E+03f, 0.0000000000000000E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		4.0071733590403869E+02f, -1.5861137916762602E+03f, 2.3839858699098645E+03f, -1.5861137916762643E+03f, 4.0071733590403909E+02f, 0.0000000000000000E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		-9.1301168206167262E+01f, 1.2316471075214675E+02f, 2.0698495299948402E-11f, -1.2316471075214508E+02f, 9.1301168206167233E+01f, 0.0000000000000000E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		-5.5339722671223846E+01f, 1.1960590540261879E+02f, -1.5249941358311668E+02f, 1.1960590540262307E+02f, -5.5339722671223605E+01f, 0.0000000000000000E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		-3.3762488150353924E+00f, 2.2839981872948751E+00f, 7.1884725699454154E-12f, -2.2839981872943818E+00f, 3.3762488150341459E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f };

	alignas(64) FLT c2006[] = {
		2.0553833234911876E+03f, 1.5499537739913128E+05f, 8.1177907023291115E+05f, 8.1177907023291173E+05f, 1.5499537739913136E+05f, 2.0553833235005691E+03f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		7.1269776034442639E+03f, 2.0581923258843314E+05f, 3.1559612614917674E+05f, -3.1559612614917627E+05f, -2.0581923258843317E+05f, -7.1269776034341394E+03f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		1.0023404568475091E+04f, 9.0916650498360192E+04f, -1.0095927514054619E+05f, -1.0095927514054628E+05f, 9.0916650498360177E+04f, 1.0023404568484635E+04f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		7.2536109410387417E+03f, 4.8347162752602981E+03f, -5.0512736602018522E+04f, 5.0512736602018478E+04f, -4.8347162752603008E+03f, -7.2536109410297540E+03f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		2.7021878300949752E+03f, -7.8773465553972646E+03f, 5.2105876478342780E+03f, 5.2105876478343343E+03f, -7.8773465553972710E+03f, 2.7021878301048723E+03f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		3.2120291706547636E+02f, -1.8229189469936762E+03f, 3.7928113414429808E+03f, -3.7928113414427025E+03f, 1.8229189469937312E+03f, -3.2120291705638243E+02f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		-1.2051267090537374E+02f, 2.2400507411399673E+02f, -1.2506575852541796E+02f, -1.2506575852521925E+02f, 2.2400507411398695E+02f, -1.2051267089640181E+02f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		-4.5977202613350237E+01f, 1.1536880606853076E+02f, -1.7819720186493959E+02f, 1.7819720186497622E+02f, -1.1536880606854736E+02f, 4.5977202622148909E+01f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		-1.5631081288842275E+00f, 7.1037430591266115E-01f, -6.9838401121429056E-02f, -6.9838401186476856E-02f, 7.1037430589285400E-01f, -1.5631081203754575E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f };

	alignas(64) FLT c2007[] = {
		3.9948351830487481E+03f, 5.4715865608590771E+05f, 5.0196413492771760E+06f, 9.8206709220713247E+06f, 5.0196413492771825E+06f, 5.4715865608590783E+05f, 3.9948351830642519E+03f, 0.0000000000000000E+00f,
		1.5290160332974696E+04f, 8.7628248584320408E+05f, 3.4421061790934438E+06f, -2.6908159596373561E-10f, -3.4421061790934461E+06f, -8.7628248584320408E+05f, -1.5290160332958067E+04f, 0.0000000000000000E+00f,
		2.4458227486779251E+04f, 5.3904618484139396E+05f, 2.4315566181017534E+05f, -1.6133959371974322E+06f, 2.4315566181017453E+05f, 5.3904618484139396E+05f, 2.4458227486795113E+04f, 0.0000000000000000E+00f,
		2.1166189345881645E+04f, 1.3382732160223130E+05f, -3.3113450969689694E+05f, 6.9013724510092140E-10f, 3.3113450969689724E+05f, -1.3382732160223136E+05f, -2.1166189345866893E+04f, 0.0000000000000000E+00f,
		1.0542795672344864E+04f, -7.0739172265098678E+03f, -6.5563293056049893E+04f, 1.2429734005960064E+05f, -6.5563293056049602E+04f, -7.0739172265098332E+03f, 1.0542795672361213E+04f, 0.0000000000000000E+00f,
		2.7903491906228419E+03f, -1.0975382873973093E+04f, 1.3656979541144799E+04f, 7.7346408577822045E-10f, -1.3656979541143772E+04f, 1.0975382873973256E+04f, -2.7903491906078298E+03f, 0.0000000000000000E+00f,
		1.6069721418053300E+02f, -1.5518707872251393E+03f, 4.3634273936642621E+03f, -5.9891976420595174E+03f, 4.3634273936642730E+03f, -1.5518707872251064E+03f, 1.6069721419533221E+02f, 0.0000000000000000E+00f,
		-1.2289277373867256E+02f, 2.8583630927743314E+02f, -2.8318194617327981E+02f, 6.9043515551118249E-10f, 2.8318194617392436E+02f, -2.8583630927760140E+02f, 1.2289277375319763E+02f, 0.0000000000000000E+00f,
		-3.2270164914249058E+01f, 9.1892112257581346E+01f, -1.6710678096334209E+02f, 2.0317049305432383E+02f, -1.6710678096383771E+02f, 9.1892112257416159E+01f, -3.2270164900224913E+01f, 0.0000000000000000E+00f,
		-1.4761409685186277E-01f, -9.1862771280377487E-01f, 1.2845147741777752E+00f, 5.6547359492808854E-10f, -1.2845147728310689E+00f, 9.1862771293147971E-01f, 1.4761410890866353E-01f, 0.0000000000000000E+00f };

	alignas(64) FLT c2008[] = {
		7.3898000697447915E+03f, 1.7297637497600035E+06f, 2.5578341605285794E+07f, 8.4789650417103335E+07f, 8.4789650417103350E+07f, 2.5578341605285816E+07f, 1.7297637497600049E+06f, 7.3898000697447915E+03f,
		3.0719636811267599E+04f, 3.1853145713323927E+06f, 2.3797981861403696E+07f, 2.4569731244678464E+07f, -2.4569731244678471E+07f, -2.3797981861403704E+07f, -3.1853145713323941E+06f, -3.0719636811267606E+04f,
		5.4488498478251728E+04f, 2.4101183255475131E+06f, 6.4554051283428287E+06f, -8.9200440393090546E+06f, -8.9200440393090583E+06f, 6.4554051283428324E+06f, 2.4101183255475126E+06f, 5.4488498478251728E+04f,
		5.3926359802542116E+04f, 9.0469037926849292E+05f, -6.0897036277696118E+05f, -3.0743852105799988E+06f, 3.0743852105800058E+06f, 6.0897036277696711E+05f, -9.0469037926849339E+05f, -5.3926359802542138E+04f,
		3.2444118016247590E+04f, 1.3079802224392134E+05f, -5.8652889370129269E+05f, 4.2333306008151924E+05f, 4.2333306008152053E+05f, -5.8652889370128722E+05f, 1.3079802224392109E+05f, 3.2444118016247590E+04f,
		1.1864306345505294E+04f, -2.2700360645707988E+04f, -5.0713607251414309E+04f, 1.8308704458211688E+05f, -1.8308704458210632E+05f, 5.0713607251413123E+04f, 2.2700360645707628E+04f, -1.1864306345505294E+04f,
		2.2812256770903232E+03f, -1.1569135767377773E+04f, 2.0942387020798891E+04f, -1.1661592834945191E+04f, -1.1661592834940149E+04f, 2.0942387020801420E+04f, -1.1569135767377924E+04f, 2.2812256770903286E+03f,
		8.5503535636821422E+00f, -9.7513976461238224E+02f, 3.8242995179171526E+03f, -6.9201295567267280E+03f, 6.9201295567248662E+03f, -3.8242995179155446E+03f, 9.7513976461209836E+02f, -8.5503535637013552E+00f,
		-1.0230637348345023E+02f, 2.8246898554269114E+02f, -3.8638201738139219E+02f, 1.9106407993320320E+02f, 1.9106407993289886E+02f, -3.8638201738492717E+02f, 2.8246898554219217E+02f, -1.0230637348345138E+02f,
		-1.9200143062947848E+01f, 6.1692257626706223E+01f, -1.2981109187842989E+02f, 1.8681284210471688E+02f, -1.8681284209654376E+02f, 1.2981109187880142E+02f, -6.1692257626845532E+01f, 1.9200143062947120E+01f,
		3.7894993760177598E-01f, -1.7334408836731494E+00f, 2.5271184057877303E+00f, -1.2600963971824484E+00f, -1.2600963917834651E+00f, 2.5271184069685657E+00f, -1.7334408840526812E+00f, 3.7894993760636758E-01f };

	alignas(64) FLT c2009[] = {
		1.3136365370186100E+04f, 5.0196413492771806E+06f, 1.1303327711722563E+08f, 5.8225443924996686E+08f, 9.7700272582690656E+08f, 5.8225443924996758E+08f, 1.1303327711722568E+08f, 5.0196413492772207E+06f, 1.3136365370186135E+04f, 0.0000000000000000E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		5.8623313038274340E+04f, 1.0326318537280345E+07f, 1.2898448324824864E+08f, 3.0522863709830385E+08f, -3.9398045056223735E-08f, -3.0522863709830391E+08f, -1.2898448324824864E+08f, -1.0326318537280388E+07f, -5.8623313038274347E+04f, 0.0000000000000000E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		1.1335001341875963E+05f, 9.0726133144784812E+06f, 5.3501544534038112E+07f, -2.6789524644146336E+05f, -1.2483923718899371E+08f, -2.6789524644172983E+05f, 5.3501544534038112E+07f, 9.0726133144785129E+06f, 1.1335001341875960E+05f, 0.0000000000000000E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		1.2489113703229747E+05f, 4.3035547171861930E+06f, 6.3021978510598792E+06f, -2.6014941986659057E+07f, 6.0417403157325170E-08f, 2.6014941986659389E+07f, -6.3021978510598652E+06f, -4.3035547171862079E+06f, -1.2489113703229751E+05f, 0.0000000000000000E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		8.6425493435991244E+04f, 1.0891182836653308E+06f, -2.0713033564200639E+06f, -2.8994941183506218E+06f, 7.5905338661205899E+06f, -2.8994941183505375E+06f, -2.0713033564200667E+06f, 1.0891182836653353E+06f, 8.6425493435991288E+04f, 0.0000000000000000E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		3.8657354724013814E+04f, 7.9936390113331305E+04f, -7.0458265546791907E+05f, 1.0151095605715880E+06f, 1.2138090419648379E-07f, -1.0151095605717725E+06f, 7.0458265546794771E+05f, -7.9936390113331567E+04f, -3.8657354724013821E+04f, 0.0000000000000000E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		1.0779131453134638E+04f, -3.3466718311300596E+04f, -1.3245366619006139E+04f, 1.8238470515353698E+05f, -2.9285656292977190E+05f, 1.8238470515350526E+05f, -1.3245366619000662E+04f, -3.3466718311299621E+04f, 1.0779131453134616E+04f, 0.0000000000000000E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		1.4992527030548456E+03f, -9.7024371533891372E+03f, 2.3216330734057381E+04f, -2.3465262819040818E+04f, 5.3299736484284360E-08f, 2.3465262819251962E+04f, -2.3216330734049119E+04f, 9.7024371533890644E+03f, -1.4992527030548747E+03f, 0.0000000000000000E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		-7.9857427421129714E+01f, -4.0585588534807385E+02f, 2.6054813773472697E+03f, -6.1806593581075495E+03f, 8.0679596874001718E+03f, -6.1806593581869265E+03f, 2.6054813773147021E+03f, -4.0585588535363172E+02f, -7.9857427421126204E+01f, 0.0000000000000000E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		-7.1572272057937070E+01f, 2.2785637019511205E+02f, -3.9109820765665262E+02f, 3.3597424711470910E+02f, 1.0596763818009852E-07f, -3.3597424723359080E+02f, 3.9109820766854079E+02f, -2.2785637019009673E+02f, 7.1572272057939983E+01f, 0.0000000000000000E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		-9.8886360698074700E+00f, 3.5359026949867051E+01f, -8.5251867715709949E+01f, 1.4285748012617628E+02f, -1.6935269668779691E+02f, 1.4285748010331625E+02f, -8.5251867711661305E+01f, 3.5359026944299828E+01f, -9.8886360698207305E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f };

	alignas(64) FLT c200a[] = {
		2.2594586605749264E+04f, 1.3595989066786593E+07f, 4.4723032442444897E+08f, 3.3781755837397518E+09f, 8.6836783895849819E+09f, 8.6836783895849762E+09f, 3.3781755837397494E+09f, 4.4723032442444897E+08f, 1.3595989066786474E+07f, 2.2594586605749344E+04f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		1.0729981697645642E+05f, 3.0651490267742988E+07f, 5.9387966085130465E+08f, 2.4434902657508330E+09f, 2.0073077861288922E+09f, -2.0073077861288943E+09f, -2.4434902657508330E+09f, -5.9387966085130453E+08f, -3.0651490267742816E+07f, -1.0729981697645638E+05f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		2.2340399734184606E+05f, 3.0258214643190462E+07f, 3.1512411458738232E+08f, 4.3618276932319808E+08f, -7.8178848450497293E+08f, -7.8178848450497019E+08f, 4.3618276932319826E+08f, 3.1512411458738232E+08f, 3.0258214643190313E+07f, 2.2340399734184548E+05f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		2.6917433004353486E+05f, 1.6875651476661228E+07f, 7.4664745481963441E+07f, -9.5882157211118385E+07f, -2.0622994435532519E+08f, 2.0622994435532743E+08f, 9.5882157211118177E+07f, -7.4664745481963515E+07f, -1.6875651476661161E+07f, -2.6917433004353428E+05f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		2.0818422772177903E+05f, 5.6084730690362519E+06f, 1.4435118192351763E+06f, -4.0063869969544649E+07f, 3.2803674392747045E+07f, 3.2803674392746095E+07f, -4.0063869969546899E+07f, 1.4435118192351642E+06f, 5.6084730690362034E+06f, 2.0818422772177853E+05f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		1.0781139496011091E+05f, 9.9202615851199068E+05f, -3.3266265543962116E+06f, -4.8557049011479173E+05f, 1.0176155522772279E+07f, -1.0176155522772269E+07f, 4.8557049011678610E+05f, 3.3266265543963453E+06f, -9.9202615851196018E+05f, -1.0781139496011072E+05f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		3.7380102688153558E+04f, 1.2716675000355666E+04f, -6.2163527451774501E+05f, 1.4157962667184104E+06f, -8.4419693137680157E+05f, -8.4419693137743860E+05f, 1.4157962667189445E+06f, -6.2163527451771160E+05f, 1.2716675000340010E+04f, 3.7380102688153442E+04f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		8.1238936393894646E+03f, -3.4872365530450072E+04f, 2.3913680325196314E+04f, 1.2428850301830019E+05f, -3.2158255329716846E+05f, 3.2158255329951923E+05f, -1.2428850301867779E+05f, -2.3913680325277423E+04f, 3.4872365530457188E+04f, -8.1238936393894255E+03f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		7.8515926628982663E+02f, -6.6607899119372642E+03f, 2.0167398338513311E+04f, -2.8951401344519112E+04f, 1.4622828142848679E+04f, 1.4622828143544031E+04f, -2.8951401346900999E+04f, 2.0167398338398041E+04f, -6.6607899119505255E+03f, 7.8515926628967964E+02f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		-1.0147176570537010E+02f, -3.5304284185385157E+01f, 1.3576976854876134E+03f, -4.3921059353471856E+03f, 7.3232085271125388E+03f, -7.3232085273978546E+03f, 4.3921059367737662E+03f, -1.3576976854043962E+03f, 3.5304284185385157E+01f, 1.0147176570550941E+02f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		-4.3161545259389186E+01f, 1.5498490981579428E+02f, -3.1771250774232175E+02f, 3.7215448796427023E+02f, -1.7181762832770994E+02f, -1.7181763036843782E+02f, 3.7215448789408123E+02f, -3.1771250773692140E+02f, 1.5498490982186786E+02f, -4.3161545259547800E+01f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		-4.2916172038214198E+00f, 1.7402146071148604E+01f, -4.7947588069135868E+01f, 9.2697698088029625E+01f, -1.2821427596894478E+02f, 1.2821427705670308E+02f, -9.2697698297776569E+01f, 4.7947588093524907E+01f, -1.7402146074502035E+01f, 4.2916172038452141E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f };

	alignas(64) FLT c200b[] = {
		3.7794653219809625E+04f, 3.4782300224660739E+07f, 1.6188020733727551E+09f, 1.7196758809615005E+10f, 6.3754384857724617E+10f, 9.7196447559193497E+10f, 6.3754384857724617E+10f, 1.7196758809614998E+10f, 1.6188020733727560E+09f, 3.4782300224660769E+07f, 3.7794653219808984E+04f, 0.0000000000000000E+00f,
		1.8969206922085886E+05f, 8.4769319065313652E+07f, 2.4230555767723408E+09f, 1.5439732722639101E+10f, 2.7112836839612309E+10f, 2.5609833368650835E-06f, -2.7112836839612328E+10f, -1.5439732722639105E+10f, -2.4230555767723408E+09f, -8.4769319065313682E+07f, -1.8969206922085711E+05f, 0.0000000000000000E+00f,
		4.2138380313901440E+05f, 9.2050522922791913E+07f, 1.5259983101266613E+09f, 4.7070559561237173E+09f, -1.2448027572952359E+09f, -1.0161446790279301E+10f, -1.2448027572952316E+09f, 4.7070559561237268E+09f, 1.5259983101266615E+09f, 9.2050522922791913E+07f, 4.2138380313901149E+05f, 0.0000000000000000E+00f,
		5.4814313598122005E+05f, 5.8085130777589552E+07f, 4.9484006166551048E+08f, 1.6222124676640952E+08f, -2.0440440381345339E+09f, 9.1416457449079640E-06f, 2.0440440381345336E+09f, -1.6222124676640788E+08f, -4.9484006166551071E+08f, -5.8085130777589560E+07f, -5.4814313598121714E+05f, 0.0000000000000000E+00f,
		4.6495183529254980E+05f, 2.3067199578027144E+07f, 6.9832590192482382E+07f, -2.2024799260683522E+08f, -1.2820270942588677E+08f, 5.1017181199129778E+08f, -1.2820270942588474E+08f, -2.2024799260683942E+08f, 6.9832590192482322E+07f, 2.3067199578027155E+07f, 4.6495183529254742E+05f, 0.0000000000000000E+00f,
		2.7021781043532980E+05f, 5.6764510325100143E+06f, -5.5650761736748898E+06f, -3.9907385617900200E+07f, 7.2453390663687646E+07f, 1.2300109686762266E-05f, -7.2453390663684472E+07f, 3.9907385617899075E+07f, 5.5650761736749066E+06f, -5.6764510325099993E+06f, -2.7021781043532846E+05f, 0.0000000000000000E+00f,
		1.0933249308680627E+05f, 6.9586821127987828E+05f, -3.6860240321937902E+06f, 2.7428169457736355E+06f, 8.3392008440593518E+06f, -1.6402201025046850E+07f, 8.3392008440698013E+06f, 2.7428169457778852E+06f, -3.6860240321937371E+06f, 6.9586821127989423E+05f, 1.0933249308680571E+05f, 0.0000000000000000E+00f,
		3.0203516161820498E+04f, -3.6879059542768438E+04f, -4.1141031216788280E+05f, 1.4111389975267777E+06f, -1.5914376635331670E+06f, 9.4095582602103753E-06f, 1.5914376635379130E+06f, -1.4111389975247320E+06f, 4.1141031216776522E+05f, 3.6879059542750314E+04f, -3.0203516161820549E+04f, 0.0000000000000000E+00f,
		5.1670143574922731E+03f, -2.8613147115372190E+04f, 4.3560195427081359E+04f, 4.8438679582765450E+04f, -2.5856630639231802E+05f, 3.7994883866738499E+05f, -2.5856630640319458E+05f, 4.8438679579510936E+04f, 4.3560195426766244E+04f, -2.8613147115376054E+04f, 5.1670143574922913E+03f, 0.0000000000000000E+00f,
		3.0888018539740131E+02f, -3.7949446187471626E+03f, 1.4313303204988082E+04f, -2.6681600235594462E+04f, 2.3856005166166615E+04f, 8.6424601730164351E-06f, -2.3856005155895236E+04f, 2.6681600234453199E+04f, -1.4313303205083188E+04f, 3.7949446187583080E+03f, -3.0888018539728523E+02f, 0.0000000000000000E+00f,
		-8.3747489794189363E+01f, 1.1948077479405792E+02f, 4.8528498015072080E+02f, -2.5024391114755094E+03f, 5.3511195318669425E+03f, -6.7655484107390166E+03f, 5.3511195362291774E+03f, -2.5024391131167667E+03f, 4.8528498019392708E+02f, 1.1948077480620087E+02f, -8.3747489794426258E+01f, 0.0000000000000000E+00f,
		-2.2640047135517630E+01f, 9.0840898563949466E+01f, -2.1597187544386938E+02f, 3.1511229111443720E+02f, -2.4856617998395282E+02f, 6.1683918215190516E-06f, 2.4856618439352349E+02f, -3.1511228757800421E+02f, 2.1597187557069353E+02f, -9.0840898570046704E+01f, 2.2640047135565219E+01f, 0.0000000000000000E+00f,
		-1.6306382886201207E+00f, 7.3325946591320434E+00f, -2.3241017682854558E+01f, 5.1715494398901185E+01f, -8.2673000279130790E+01f, 9.6489719151212370E+01f, -8.2673010381149226E+01f, 5.1715494328769353E+01f, -2.3241018024860580E+01f, 7.3325946448852415E+00f, -1.6306382886460551E+00f, 0.0000000000000000E+00f };

	alignas(64) FLT c200c[] = {
		6.1722991679852908E+04f, 8.4789650417103648E+07f, 5.4431675199498701E+09f, 7.8788892335272232E+10f, 4.0355760945670044E+11f, 8.8071481911347949E+11f, 8.8071481911347961E+11f, 4.0355760945670044E+11f, 7.8788892335272430E+10f, 5.4431675199498835E+09f, 8.4789650417103708E+07f, 6.1722991679871957E+04f,
		3.2561466099406168E+05f, 2.2112758120210618E+08f, 8.9911609880089817E+09f, 8.3059508064200943E+10f, 2.3965569143469864E+11f, 1.6939286803305212E+11f, -1.6939286803305203E+11f, -2.3965569143469864E+11f, -8.3059508064201080E+10f, -8.9911609880089989E+09f, -2.2112758120210618E+08f, -3.2561466099404311E+05f,
		7.6621098001581512E+05f, 2.6026568260310286E+08f, 6.4524338253008652E+09f, 3.3729904113826820E+10f, 2.8555202212474091E+10f, -6.8998572040731537E+10f, -6.8998572040731445E+10f, 2.8555202212474079E+10f, 3.3729904113826824E+10f, 6.4524338253008757E+09f, 2.6026568260310274E+08f, 7.6621098001583829E+05f,
		1.0657807616803218E+06f, 1.8144472126890984E+08f, 2.5524827004349842E+09f, 5.2112383911371660E+09f, -1.0268350564014645E+10f, -1.4763245309081306E+10f, 1.4763245309081314E+10f, 1.0268350564014671E+10f, -5.2112383911371059E+09f, -2.5524827004349871E+09f, -1.8144472126890984E+08f, -1.0657807616803099E+06f,
		9.7829638830158755E+05f, 8.2222351241519913E+07f, 5.5676911894064474E+08f, -4.8739037675427330E+08f, -2.7153428193078227E+09f, 2.5627633609246106E+09f, 2.5627633609246163E+09f, -2.7153428193078651E+09f, -4.8739037675430620E+08f, 5.5676911894064546E+08f, 8.2222351241519868E+07f, 9.7829638830161188E+05f,
		6.2536876825114002E+05f, 2.4702814073680203E+07f, 4.1488431554846466E+07f, -2.9274790542418826E+08f, 1.0742154109191516E+08f, 6.2185168968032193E+08f, -6.2185168968012476E+08f, -1.0742154109184742E+08f, 2.9274790542423087E+08f, -4.1488431554843128E+07f, -2.4702814073680237E+07f, -6.2536876825112454E+05f,
		2.8527714307528478E+05f, 4.6266378435690766E+06f, -1.0665598090790771E+07f, -2.6048960239891130E+07f, 9.1597254427317813E+07f, -5.9794495983264342E+07f, -5.9794495983220413E+07f, 9.1597254427343085E+07f, -2.6048960239921503E+07f, -1.0665598090794146E+07f, 4.6266378435690673E+06f, 2.8527714307530399E+05f,
		9.2873647411234080E+04f, 3.6630046787425119E+05f, -3.1271047224730137E+06f, 4.8612412939252760E+06f, 3.3820440907796426E+06f, -1.6880127953704204E+07f, 1.6880127953756198E+07f, -3.3820440907614031E+06f, -4.8612412938993908E+06f, 3.1271047224752530E+06f, -3.6630046787425695E+05f, -9.2873647411217215E+04f,
		2.0817947751046438E+04f, -5.5660303410315042E+04f, -1.9519783923444615E+05f, 1.0804817251338551E+06f, -1.8264985852555393E+06f, 9.7602844968061335E+05f, 9.7602844962902542E+05f, -1.8264985852963410E+06f, 1.0804817251124913E+06f, -1.9519783923503032E+05f, -5.5660303410363231E+04f, 2.0817947751063632E+04f,
		2.7986023314783361E+03f, -1.9404411093655592E+04f, 4.3922625000519314E+04f, -7.6450317451901383E+03f, -1.5273911974273989E+05f, 3.3223441458516393E+05f, -3.3223441441930021E+05f, 1.5273911979752057E+05f, 7.6450317512768806E+03f, -4.3922624998141677E+04f, 1.9404411093637758E+04f, -2.7986023314644049E+03f,
		6.7849020474048089E+01f, -1.7921351308204744E+03f, 8.4980694686552797E+03f, -1.9742624859769410E+04f, 2.4620674845030797E+04f, -1.1676544851227827E+04f, -1.1676544869194569E+04f, 2.4620674845030626E+04f, -1.9742624831436660E+04f, 8.4980694630406069E+03f, -1.7921351308312935E+03f, 6.7849020488592075E+01f,
		-5.4577020998836872E+01f, 1.3637112867242237E+02f, 4.5513616580246023E+01f, -1.1174001367986359E+03f, 3.2018769312434206E+03f, -5.0580351396215219E+03f, 5.0580351683422405E+03f, -3.2018769242193171E+03f, 1.1174000998831286E+03f, -4.5513609243969356E+01f, -1.3637112867730119E+02f, 5.4577021011726984E+01f,
		-1.0538365872268786E+01f, 4.6577222488645518E+01f, -1.2606964198473415E+02f, 2.1881091668968099E+02f, -2.3273399614976032E+02f, 1.0274275204276027E+02f, 1.0274270265494516E+02f, -2.3273401859852868E+02f, 2.1881091865396468E+02f, -1.2606964777237258E+02f, 4.6577222453584369E+01f, -1.0538365860573146E+01f,
		-4.6087004144309118E-01f, 2.5969759128998060E+00f, -9.6946932216381381E+00f, 2.4990041962121211E+01f, -4.6013909139329137E+01f, 6.2056985032913090E+01f, -6.2056925855365186E+01f, 4.6013921000662158E+01f, -2.4990037445376750E+01f, 9.6946954085586885E+00f, -2.5969759201692755E+00f, 4.6087004744129911E-01f };

	alignas(64) FLT c200d[] = {
		9.8715725867495363E+04f, 1.9828875496808097E+08f, 1.7196758809614983E+10f, 3.3083776881353577E+11f, 2.2668873993375439E+12f, 6.7734720591167568E+12f, 9.6695220682534785E+12f, 6.7734720591167432E+12f, 2.2668873993375430E+12f, 3.3083776881353503E+11f, 1.7196758809614998E+10f, 1.9828875496807891E+08f, 9.8715725867496090E+04f, 0.0000000000000000E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		5.4491110456935549E+05f, 5.4903670125539351E+08f, 3.0879465445278183E+10f, 3.9588436413399969E+11f, 1.6860562536749778E+12f, 2.4256447893117891E+12f, -5.5583944938791784E-05f, -2.4256447893117847E+12f, -1.6860562536749768E+12f, -3.9588436413399890E+11f, -3.0879465445278183E+10f, -5.4903670125538898E+08f, -5.4491110456935526E+05f, 0.0000000000000000E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		1.3504711883426071E+06f, 6.9286979077463162E+08f, 2.4618123595484577E+10f, 1.9493985627722607E+11f, 3.9422703517046350E+11f, -1.8678883613919861E+11f, -8.5538079834550110E+11f, -1.8678883613919730E+11f, 3.9422703517046375E+11f, 1.9493985627722589E+11f, 2.4618123595484566E+10f, 6.9286979077462614E+08f, 1.3504711883426069E+06f, 0.0000000000000000E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		1.9937206140846491E+06f, 5.2512029493765980E+08f, 1.1253303793811750E+10f, 4.6205527735932152E+10f, -1.1607472377983305E+10f, -1.6305241755642313E+11f, 3.5385440504350348E-04f, 1.6305241755642365E+11f, 1.1607472377982582E+10f, -4.6205527735932213E+10f, -1.1253303793811750E+10f, -5.2512029493765628E+08f, -1.9937206140846489E+06f, 0.0000000000000000E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		1.9607419630386413E+06f, 2.6425362558103892E+08f, 3.1171259341747193E+09f, 2.9839860297839913E+09f, -1.9585031917561897E+10f, -5.0666917387065792E+09f, 3.6568794485480583E+10f, -5.0666917387057562E+09f, -1.9585031917561817E+10f, 2.9839860297838497E+09f, 3.1171259341747184E+09f, 2.6425362558103728E+08f, 1.9607419630386417E+06f, 0.0000000000000000E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		1.3593773865640305E+06f, 9.1556445104158267E+07f, 4.7074012944133747E+08f, -1.1192579335657008E+09f, -2.1090780087868555E+09f, 5.2270306737951984E+09f, 5.6467240041521856E-04f, -5.2270306737934217E+09f, 2.1090780087880819E+09f, 1.1192579335658383E+09f, -4.7074012944133127E+08f, -9.1556445104157984E+07f, -1.3593773865640305E+06f, 0.0000000000000000E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		6.8417206432039209E+05f, 2.1561705510027152E+07f, 7.5785249893055111E+06f, -2.7456096030221754E+08f, 3.4589095671054310E+08f, 4.0256106808894646E+08f, -1.0074306926603404E+09f, 4.0256106809081393E+08f, 3.4589095670997137E+08f, -2.7456096030236483E+08f, 7.5785249893030487E+06f, 2.1561705510027405E+07f, 6.8417206432039209E+05f, 0.0000000000000000E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		2.5248269397037517E+05f, 3.0985559672616189E+06f, -1.1816517087616559E+07f, -8.2958498770184973E+06f, 8.0546642347355247E+07f, -1.0594657799485898E+08f, 2.1816722293163801E-04f, 1.0594657799424352E+08f, -8.0546642347497791E+07f, 8.2958498771036500E+06f, 1.1816517087615721E+07f, -3.0985559672621777E+06f, -2.5248269397037517E+05f, 0.0000000000000000E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		6.7530100970876694E+04f, 1.2373362326658823E+05f, -2.1245597183281910E+06f, 5.1047323238754412E+06f, -1.4139444405488928E+06f, -1.1818267555096827E+07f, 2.0121548578624789E+07f, -1.1818267557079868E+07f, -1.4139444401348191E+06f, 5.1047323236516044E+06f, -2.1245597183309775E+06f, 1.2373362326702787E+05f, 6.7530100970876316E+04f, 0.0000000000000000E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		1.2421368748961073E+04f, -5.0576243647011936E+04f, -4.8878193436902722E+04f, 6.5307896872028301E+05f, -1.5497610127060430E+06f, 1.5137725917321201E+06f, 4.1615986404011299E-04f, -1.5137725918538549E+06f, 1.5497610130469005E+06f, -6.5307896856811445E+05f, 4.8878193438804832E+04f, 5.0576243646433126E+04f, -1.2421368748961073E+04f, 0.0000000000000000E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		1.2904654687550299E+03f, -1.1169946055009055E+04f, 3.3275109713863385E+04f, -3.1765222274236821E+04f, -5.9810982085323274E+04f, 2.2355863038592847E+05f, -3.1083591705219547E+05f, 2.2355863445202672E+05f, -5.9810982721084511E+04f, -3.1765222464963932E+04f, 3.3275109714208855E+04f, -1.1169946054555618E+04f, 1.2904654687545376E+03f, 0.0000000000000000E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		-1.9043622268674213E+01f, -6.8296542209516542E+02f, 4.2702512274202591E+03f, -1.2165497317825058E+04f, 1.9423733298269544E+04f, -1.6010024066956401E+04f, 3.4018642874429026E-04f, 1.6010021599471667E+04f, -1.9423732817821805E+04f, 1.2165497483905752E+04f, -4.2702512286689680E+03f, 6.8296542153908558E+02f, 1.9043622268312891E+01f, 0.0000000000000000E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		-3.0093984465361217E+01f, 9.8972865724808671E+01f, -9.7437038666761538E+01f, -3.5079928405373198E+02f, 1.5699250566648977E+03f, -3.1287439837941820E+03f, 3.8692196309709061E+03f, -3.1287462825615335E+03f, 1.5699252631958864E+03f, -3.5079944793112952E+02f, -9.7437041893750632E+01f, 9.8972866189610414E+01f, -3.0093984465884773E+01f, 0.0000000000000000E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		-4.3050286009489040E+00f, 2.1108975724659501E+01f, -6.4297198812570272E+01f, 1.2922884632277874E+02f, -1.6991812716212596E+02f, 1.2655005901719436E+02f, 9.2483537895948854E-05f, -1.2655066232531748E+02f, 1.6991805207569072E+02f, -1.2922893667436634E+02f, 6.4297198424711908E+01f, -2.1108976207523057E+01f, 4.3050286009485790E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		-1.0957333716725008E-01f, 7.2949317004436565E-01f, -3.4300816058693728E+00f, 1.0470054474579324E+01f, -2.2292134950656113E+01f, 3.4570827323582719E+01f, -3.9923523442753932E+01f, 3.4573264959502886E+01f, -2.2292358612963266E+01f, 1.0470042004916014E+01f, -3.4300810538570281E+00f, 7.2949352113279253E-01f, -1.0957333740315604E-01f, 0.0000000000000000E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f };

	alignas(64) FLT c200e[] = {
		1.5499533202966207E+05f, 4.4723032442444688E+08f, 5.1495083701694740E+10f, 1.2904576022918071E+12f, 1.1534950432785506E+13f, 4.5650102198520484E+13f, 8.8830582190032641E+13f, 8.8830582190032641E+13f, 4.5650102198520492E+13f, 1.1534950432785527E+13f, 1.2904576022918074E+12f, 5.1495083701695107E+10f, 4.4723032442444855E+08f, 1.5499533202970232E+05f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		8.9188339002980455E+05f, 1.3065352538728635E+09f, 9.9400185225815567E+10f, 1.7136059013402405E+12f, 1.0144146621675832E+13f, 2.3034036018490715E+13f, 1.4630967270448871E+13f, -1.4630967270448855E+13f, -2.3034036018490719E+13f, -1.0144146621675846E+13f, -1.7136059013402405E+12f, -9.9400185225815964E+10f, -1.3065352538728662E+09f, -8.9188339002979454E+05f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		2.3170473769379663E+06f, 1.7532505043698256E+09f, 8.6523535958354309E+10f, 9.7455289065487354E+11f, 3.2977972139362314E+12f, 1.7874626001697781E+12f, -6.1480918082633916E+12f, -6.1480918082633975E+12f, 1.7874626001697690E+12f, 3.2977972139362285E+12f, 9.7455289065487329E+11f, 8.6523535958354630E+10f, 1.7532505043698275E+09f, 2.3170473769380399E+06f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		3.6089249230396422E+06f, 1.4278058213962190E+09f, 4.4296625537022423E+10f, 2.9466624630419781E+11f, 3.1903621584503235E+11f, -9.8834691411254565E+11f, -1.1072264714919226E+12f, 1.1072264714919316E+12f, 9.8834691411255151E+11f, -3.1903621584503467E+11f, -2.9466624630419769E+11f, -4.4296625537022621E+10f, -1.4278058213962219E+09f, -3.6089249230396664E+06f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		3.7733555140851745E+06f, 7.8376718099107409E+08f, 1.4443117772349569E+10f, 4.3197433307418671E+10f, -7.6585042240585556E+10f, -1.8569640140763062E+11f, 2.0385335192657199E+11f, 2.0385335192656519E+11f, -1.8569640140762662E+11f, -7.6585042240580856E+10f, 4.3197433307418686E+10f, 1.4443117772349669E+10f, 7.8376718099107552E+08f, 3.7733555140852560E+06f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		2.8079157920112358E+06f, 3.0340753492383724E+08f, 2.9498136661747241E+09f, -6.2820200387919831E+08f, -2.2372008390623215E+10f, 1.5217518660584890E+10f, 4.0682590266891922E+10f, -4.0682590266869431E+10f, -1.5217518660582748E+10f, 2.2372008390625935E+10f, 6.2820200387968791E+08f, -2.9498136661747637E+09f, -3.0340753492383808E+08f, -2.8079157920112377E+06f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		1.5361613559533111E+06f, 8.3513615594416574E+07f, 3.0077547202708024E+08f, -1.3749596754067802E+09f, -6.6733027297557127E+08f, 5.9590333632819109E+09f, -4.3025685566870070E+09f, -4.3025685566872711E+09f, 5.9590333632806673E+09f, -6.6733027297523963E+08f, -1.3749596754067125E+09f, 3.0077547202709383E+08f, 8.3513615594416171E+07f, 1.5361613559533576E+06f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		6.2759409419592959E+05f, 1.5741723594963098E+07f, -1.5632610223406436E+07f, -1.9294824907078514E+08f, 4.4643806532434595E+08f, 1.5178998385244830E+07f, -9.6771139891725647E+08f, 9.6771139892509627E+08f, -1.5178998381042883E+07f, -4.4643806533176166E+08f, 1.9294824907065383E+08f, 1.5632610223392555E+07f, -1.5741723594963137E+07f, -6.2759409419590747E+05f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		1.9151404903933613E+05f, 1.7156606891563335E+06f, -9.7733523156688716E+06f, 4.2982266233154163E+06f, 5.1660907884347722E+07f, -1.1279400211155911E+08f, 6.4701089573962681E+07f, 6.4701089571562663E+07f, -1.1279400211012064E+08f, 5.1660907891220264E+07f, 4.2982266233826512E+06f, -9.7733523157112263E+06f, 1.7156606891560503E+06f, 1.9151404903936724E+05f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		4.2715272622845026E+04f, -2.2565910611953568E+03f, -1.1769776156959014E+06f, 4.0078399907813077E+06f, -3.8951858063335596E+06f, -5.0944610754510267E+06f, 1.6765992446914168E+07f, -1.6765992426657490E+07f, 5.0944610781778870E+06f, 3.8951858062361716E+06f, -4.0078399907326135E+06f, 1.1769776157141617E+06f, 2.2565910606306688E+03f, -4.2715272622820135E+04f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		6.4806786522793900E+03f, -3.5474227032974472E+04f, 1.8237100709385861E+04f, 3.0934714629696816E+05f, -1.0394703931686131E+06f, 1.4743920333143482E+06f, -7.3356882447856572E+05f, -7.3356882916658197E+05f, 1.4743920305501707E+06f, -1.0394703929917105E+06f, 3.0934714631908614E+05f, 1.8237100665157792E+04f, -3.5474227033406372E+04f, 6.4806786523010323E+03f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		4.9913632908459954E+02f, -5.5416668524952684E+03f, 2.0614058717617296E+04f, -3.2285139072943130E+04f, -5.3099550821623425E+03f, 1.1559000502166932E+05f, -2.2569743259261423E+05f, 2.2569743616896842E+05f, -1.1559000130545651E+05f, 5.3099543129458480E+03f, 3.2285139142872020E+04f, -2.0614058670790018E+04f, 5.5416668533342381E+03f, -4.9913632906195977E+02f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		-3.3076333188134086E+01f, -1.8970588563697331E+02f, 1.8160423493164808E+03f, -6.3715703355644328E+03f, 1.2525624574329036E+04f, -1.4199806452802783E+04f, 6.4441892296909591E+03f, 6.4441909537524216E+03f, -1.4199808176873401E+04f, 1.2525626154733827E+04f, -6.3715704433222418E+03f, 1.8160422729911850E+03f, -1.8970588700495102E+02f, -3.3076333168231550E+01f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		-1.4394533627743886E+01f, 5.7000699089242815E+01f, -1.0101142663923416E+02f, -3.2954197414395189E+01f, 6.1417879182394654E+02f, -1.6177283846697430E+03f, 2.4593386157454975E+03f, -2.4593322941165261E+03f, 1.6177291239900730E+03f, -6.1417952013923764E+02f, 3.2954100943010943E+01f, 1.0101142710333265E+02f, -5.7000699100179844E+01f, 1.4394533639240331E+01f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		-1.5925952284027161E+00f, 8.5113930215357829E+00f, -2.8993523187012922E+01f, 6.6373454994590404E+01f, -1.0329574518449559E+02f, 1.0280184257681817E+02f, -4.3896094875192006E+01f, -4.3899302208087086E+01f, 1.0280039795628096E+02f, -1.0329511291885207E+02f, 6.6373435700858948E+01f, -2.8993536490606409E+01f, 8.5113924808491728E+00f, -1.5925952194145006E+00f, 0.0000000000000000E+00f, 0.0000000000000000E+00f,
		1.5984868520881029E-02f, 1.2876175212962959E-01f, -9.8358742969175483E-01f, 3.7711523389360830E+00f, -9.4305498095765508E+00f, 1.6842854581416674E+01f, -2.2308566502972713E+01f, 2.2308940200151390E+01f, -1.6841512668820517E+01f, 9.4313524091989347E+00f, -3.7710716543179599E+00f, 9.8361025494556609E-01f, -1.2876100566420701E-01f, -1.5984859433053292E-02f, 0.0000000000000000E+00f, 0.0000000000000000E+00f };

	alignas(64) FLT c200f[] = {
		2.3939707792241839E+05f, 9.7700272582690191E+08f, 1.4715933396485257E+11f, 4.7242424833337158E+12f, 5.3987426629953594E+13f, 2.7580474290566078E+14f, 7.0693378336533400E+14f, 9.6196578554477775E+14f, 7.0693378336533400E+14f, 2.7580474290566125E+14f, 5.3987426629953766E+13f, 4.7242424833337246E+12f, 1.4715933396485263E+11f, 9.7700272582690215E+08f, 2.3939707792242285E+05f, 0.0000000000000000E+00f,
		1.4314487885226035E+06f, 2.9961416925358453E+09f, 3.0273361232748438E+11f, 6.8507333793903584E+12f, 5.4192702756911000E+13f, 1.7551587948105309E+14f, 2.1874615668430150E+14f, 3.4316191014053393E-02f, -2.1874615668430150E+14f, -1.7551587948105334E+14f, -5.4192702756911180E+13f, -6.8507333793903701E+12f, -3.0273361232748438E+11f, -2.9961416925358458E+09f, -1.4314487885226049E+06f, 0.0000000000000000E+00f,
		3.8829497354762917E+06f, 4.2473082696966448E+09f, 2.8414312556015540E+11f, 4.3688281331121411E+12f, 2.1823119508000543E+13f, 3.2228098609392094E+13f, -2.1833085454691789E+13f, -7.3750710225100812E+13f, -2.1833085454691820E+13f, 3.2228098609392055E+13f, 2.1823119508000594E+13f, 4.3688281331121479E+12f, 2.8414312556015527E+11f, 4.2473082696966434E+09f, 3.8829497354762889E+06f, 0.0000000000000000E+00f,
		6.3495763451755755E+06f, 3.6841035003733950E+09f, 1.5965774278321045E+11f, 1.5630338683778201E+12f, 3.8749058615819268E+12f, -2.7319740087723574E+12f, -1.3233342822865402E+13f, 6.1642230420317079E-02f, 1.3233342822865449E+13f, 2.7319740087723975E+12f, -3.8749058615819365E+12f, -1.5630338683778203E+12f, -1.5965774278321042E+11f, -3.6841035003733935E+09f, -6.3495763451755764E+06f, 0.0000000000000000E+00f,
		7.0146619045520434E+06f, 2.1782897863065763E+09f, 5.8897780310148087E+10f, 3.1953009601770325E+11f, 4.0651527029737198E+08f, -1.6379148273276064E+12f, -1.1568753137013029E+11f, 2.7451653250460508E+12f, -1.1568753137012485E+11f, -1.6379148273277261E+12f, 4.0651527029819238E+08f, 3.1953009601770361E+11f, 5.8897780310148087E+10f, 2.1782897863065763E+09f, 7.0146619045520443E+06f, 0.0000000000000000E+00f,
		5.5580012413990172E+06f, 9.2345162185944164E+08f, 1.4522950934020109E+10f, 2.7025952371212009E+10f, -1.2304576967641914E+11f, -1.0116752717202786E+11f, 3.8517418245458325E+11f, 1.0918347404432817E-01f, -3.8517418245444312E+11f, 1.0116752717221135E+11f, 1.2304576967643665E+11f, -2.7025952371214943E+10f, -1.4522950934020079E+10f, -9.2345162185944211E+08f, -5.5580012413990181E+06f, 0.0000000000000000E+00f,
		3.2693972344231778E+06f, 2.8610260147425205E+08f, 2.2348528403750563E+09f, -3.4574515574242272E+09f, -1.7480626463583939E+10f, 3.1608597465540653E+10f, 1.9879262560072273E+10f, -6.6148013553772224E+10f, 1.9879262560085339E+10f, 3.1608597465515747E+10f, -1.7480626463576942E+10f, -3.4574515574198236E+09f, 2.2348528403750110E+09f, 2.8610260147425193E+08f, 3.2693972344231787E+06f, 0.0000000000000000E+00f,
		1.4553539959296256E+06f, 6.4136842048384041E+07f, 1.3622336582062906E+08f, -1.2131510424644001E+09f, 6.4322366984221375E+08f, 4.5078753872047586E+09f, -7.1689413746930647E+09f, 3.2906916833662987E-02f, 7.1689413746724453E+09f, -4.5078753875009747E+09f, -6.4322366985365331E+08f, 1.2131510424608817E+09f, -1.3622336582067037E+08f, -6.4136842048384242E+07f, -1.4553539959296256E+06f, 0.0000000000000000E+00f,
		4.9358776531681651E+05f, 9.7772970960585065E+06f, -2.3511574237987626E+07f, -1.0142613816641946E+08f, 3.9421144218035364E+08f, -2.8449115593052310E+08f, -5.7549243243741119E+08f, 1.1608781631182449E+09f, -5.7549243240763104E+08f, -2.8449115600447333E+08f, 3.9421144214381480E+08f, -1.0142613816429654E+08f, -2.3511574237995699E+07f, 9.7772970960588697E+06f, 4.9358776531681546E+05f, 0.0000000000000000E+00f,
		1.2660319987326677E+05f, 7.7519511328119377E+05f, -6.5244610661450895E+06f, 9.0878257488052379E+06f, 2.3116605621149920E+07f, -8.7079594462079599E+07f, 9.5542733739275128E+07f, 6.0548970733798724E-02f, -9.5542733661364838E+07f, 8.7079594608550951E+07f, -2.3116605559600785E+07f, -9.0878257522138134E+06f, 6.5244610661298726E+06f, -7.7519511328133650E+05f, -1.2660319987326639E+05f, 0.0000000000000000E+00f,
		2.3793325531458529E+04f, -4.2305332803808597E+04f, -5.2884156985535356E+05f, 2.5307340127864038E+06f, -4.0404175271559842E+06f, -1.7519992360184138E+05f, 1.0146438805818636E+07f, -1.5828545480742473E+07f, 1.0146438778928882E+07f, -1.7520004389869148E+05f, -4.0404175770437294E+06f, 2.5307340149977510E+06f, -5.2884156989405944E+05f, -4.2305332803937294E+04f, 2.3793325531459184E+04f, 0.0000000000000000E+00f,
		2.9741655196834722E+03f, -2.0687056403786246E+04f, 3.3295507799709936E+04f, 1.0661145730323243E+05f, -5.6644238105382060E+05f, 1.0874811616841732E+06f, -9.6561270266008016E+05f, 1.5626594062671070E-02f, 9.6561272951271443E+05f, -1.0874812528712249E+06f, 5.6644243308078672E+05f, -1.0661145838213131E+05f, -3.3295507812197495E+04f, 2.0687056403630129E+04f, -2.9741655196846405E+03f, 0.0000000000000000E+00f,
		1.5389176594899303E+02f, -2.3864418511494741E+03f, 1.0846266954249364E+04f, -2.2940053396478714E+04f, 1.4780106121058996E+04f, 4.2663651769852157E+04f, -1.3047648013242516E+05f, 1.7468401314164279E+05f, -1.3047645484607235E+05f, 4.2663541429144650E+04f, 1.4780036296018619E+04f, -2.2940053180976502E+04f, 1.0846266927315819E+04f, -2.3864418517113058E+03f, 1.5389176594779781E+02f, 0.0000000000000000E+00f,
		-2.3857631312588978E+01f, -1.9651606133609231E+01f, 6.4183083829803820E+02f, -2.8648433109641578E+03f, 6.8249243722518859E+03f, -9.7944325124827701E+03f, 7.6177757600121276E+03f, 1.8034307737205296E-02f, -7.6177559127722052E+03f, 9.7944326623113047E+03f, -6.8249058342322496E+03f, 2.8648407117981119E+03f, -6.4183085438795774E+02f, 1.9651605969778377E+01f, 2.3857631312809222E+01f, 0.0000000000000000E+00f,
		-6.1348505739169541E+00f, 2.7872915855267404E+01f, -6.5819942538871970E+01f, 5.1366231962952028E+01f, 1.7213955398158618E+02f, -6.9658621010000411E+02f, 1.3192236112353403E+03f, -1.6054106225233884E+03f, 1.3192031991952242E+03f, -6.9663961216547739E+02f, 1.7211403815802629E+02f, 5.1367579954366171E+01f, -6.5819957939661379E+01f, 2.7872915947616441E+01f, -6.1348505735855374E+00f, 0.0000000000000000E+00f,
		-4.9671584513490097E-01f, 3.0617550953446115E+00f, -1.1650665638578070E+01f, 3.0081586723089057E+01f, -5.4028356726202020E+01f, 6.6077203078498044E+01f, -4.7145500171928198E+01f, 4.2118837140985958E-03f, 4.7167106663349848E+01f, -6.6048394423269173E+01f, 5.4062906728994193E+01f, -3.0081603709324451E+01f, 1.1650672008416343E+01f, -3.0617551285208524E+00f, 4.9671584437353217E-01f, 0.0000000000000000E+00f,
		4.3460786767313729E-03f, -1.3199600771767199E-02f, -1.9412688562910244E-01f, 1.1329433700669471E+00f, -3.4442045795063887E+00f, 7.1737626956468912E+00f, -1.1098109271625262E+01f, 1.2385772358881393E+01f, -1.1101471316239516E+01f, 7.0913926025978853E+00f, -3.4845491148773502E+00f, 1.1323523856621058E+00f, -1.9414904754428672E-01f, -1.3200165079792004E-02f, 4.3460782759443158E-03f, 0.0000000000000000E+00f };

	alignas(64) FLT c200g[] = {
		3.6434551345570839E+05f, 2.0744705928579483E+09f, 4.0355760945669995E+11f, 1.6364575388763029E+13f, 2.3514830376056538E+14f, 1.5192201717462528E+15f, 4.9956173084674090E+15f, 8.9287666945127360E+15f, 8.9287666945127390E+15f, 4.9956173084674090E+15f, 1.5192201717462528E+15f, 2.3514830376056538E+14f, 1.6364575388763035E+13f, 4.0355760945670026E+11f, 2.0744705928579524E+09f, 3.6434551345571183E+05f,
		2.2576246485480359E+06f, 6.6499571180086451E+09f, 8.7873753526056287E+11f, 2.5606844387131066E+13f, 2.6313738449330153E+14f, 1.1495095100701460E+15f, 2.1932582707747560E+15f, 1.2860244365132595E+15f, -1.2860244365132600E+15f, -2.1932582707747578E+15f, -1.1495095100701465E+15f, -2.6313738449330159E+14f, -2.5606844387131062E+13f, -8.7873753526056299E+11f, -6.6499571180086451E+09f, -2.2576246485480373E+06f,
		6.3730995546265077E+06f, 9.9060026035198078E+09f, 8.8097248605449023E+11f, 1.7953384130753688E+13f, 1.2398425545001662E+14f, 3.0749346493041262E+14f, 1.0259777520247159E+14f, -5.5291976457534325E+14f, -5.5291976457534325E+14f, 1.0259777520247186E+14f, 3.0749346493041219E+14f, 1.2398425545001659E+14f, 1.7953384130753676E+13f, 8.8097248605448950E+11f, 9.9060026035198040E+09f, 6.3730995546265030E+06f,
		1.0896915393078227E+07f, 9.0890343524593849E+09f, 5.3565169504010010E+11f, 7.3004206720038701E+12f, 2.9692333044160066E+13f, 1.6051737468109549E+13f, -9.1273329108089906E+13f, -8.5999306918502953E+13f, 8.5999306918502422E+13f, 9.1273329108089984E+13f, -1.6051737468109510E+13f, -2.9692333044160082E+13f, -7.3004206720038701E+12f, -5.3565169504010022E+11f, -9.0890343524593849E+09f, -1.0896915393078227E+07f,
		1.2655725616100594E+07f, 5.7342804054544210E+09f, 2.1822836608899570E+11f, 1.8300700858999690E+12f, 2.7770431049857676E+12f, -8.5034969223852568E+12f, -1.2846668467423438E+13f, 1.6519076896571838E+13f, 1.6519076896572182E+13f, -1.2846668467423555E+13f, -8.5034969223850703E+12f, 2.7770431049857896E+12f, 1.8300700858999678E+12f, 2.1822836608899567E+11f, 5.7342804054544210E+09f, 1.2655725616100591E+07f,
		1.0609303958036326E+07f, 2.6255609052371716E+09f, 6.1673589426039413E+10f, 2.6044432099085333E+11f, -3.5431628074578204E+11f, -1.6077602129636348E+12f, 1.5534405614728977E+12f, 2.8019935380857432E+12f, -2.8019935380841978E+12f, -1.5534405614724106E+12f, 1.6077602129635625E+12f, 3.5431628074580896E+11f, -2.6044432099084848E+11f, -6.1673589426039429E+10f, -2.6255609052371716E+09f, -1.0609303958036322E+07f,
		6.6544809363384582E+06f, 8.9490403680928326E+08f, 1.1882638725190845E+10f, 8.1552898137823076E+09f, -1.2575562817886868E+11f, 2.7074695075907585E+10f, 3.9453789461955023E+11f, -3.1679644857468066E+11f, -3.1679644857392346E+11f, 3.9453789461966650E+11f, 2.7074695075992649E+10f, -1.2575562817884555E+11f, 8.1552898137788668E+09f, 1.1882638725190889E+10f, 8.9490403680928278E+08f, 6.6544809363384554E+06f,
		3.1906872142825006E+06f, 2.2785946180651775E+08f, 1.3744578972809248E+09f, -4.3997172592883167E+09f, -9.2011130754043922E+09f, 3.4690551711832901E+10f, -9.4227043395047741E+09f, -5.9308465070198639E+10f, 5.9308465069336540E+10f, 9.4227043396350136E+09f, -3.4690551711738396E+10f, 9.2011130753567543E+09f, 4.3997172592879610E+09f, -1.3744578972813025E+09f, -2.2785946180651844E+08f, -3.1906872142825015E+06f,
		1.1821527096621769E+06f, 4.2281234059839502E+07f, 2.8723226058712766E+07f, -8.3553955857628822E+08f, 1.2447304828823066E+09f, 2.1955280943585949E+09f, -7.0514195726908512E+09f, 4.3745141239718714E+09f, 4.3745141233600502E+09f, -7.0514195728029747E+09f, 2.1955280943510208E+09f, 1.2447304828590808E+09f, -8.3553955857879233E+08f, 2.8723226058761366E+07f, 4.2281234059838109E+07f, 1.1821527096621762E+06f,
		3.3854610744280310E+05f, 5.2176984975081543E+06f, -2.0677283565079328E+07f, -3.5831818968518838E+07f, 2.6599346106412742E+08f, -3.7992777977357000E+08f, -1.3426914417466179E+08f, 9.1752051229224503E+08f, -9.1752051129499328E+08f, 1.3426914497246322E+08f, 3.7992777991069216E+08f, -2.6599346104854536E+08f, 3.5831818968908392E+07f, 2.0677283564896725E+07f, -5.2176984975075833E+06f, -3.3854610744279937E+05f,
		7.3893334077310064E+04f, 2.6983804209559254E+05f, -3.6415998561101072E+06f, 8.4025485849181097E+06f, 4.9278860779345948E+06f, -5.1437033846752726E+07f, 8.7603898676325440E+07f, -4.6199498412402093E+07f, -4.6199498208604209E+07f, 8.7603898435731798E+07f, -5.1437033863736227E+07f, 4.9278861005789889E+06f, 8.4025485831489991E+06f, -3.6415998560990733E+06f, 2.6983804209473461E+05f, 7.3893334077307401E+04f,
		1.1778892113375481E+04f, -4.0077190108724200E+04f, -1.8372552175909068E+05f, 1.3262878399160223E+06f, -2.9738539927520575E+06f, 1.9493509709529271E+06f, 4.1881949951139782E+06f, -1.1066749616505133E+07f, 1.1066749327519676E+07f, -4.1881946843906553E+06f, -1.9493507810665092E+06f, 2.9738539818831389E+06f, -1.3262878384774840E+06f, 1.8372552162922107E+05f, 4.0077190107319519E+04f, -1.1778892113376129E+04f,
		1.2019749667923656E+03f, -1.0378455844500613E+04f, 2.6333352653155256E+04f, 1.7117060106301305E+04f, -2.5133287443653666E+05f, 6.4713914262131555E+05f, -8.1634942572553246E+05f, 3.8623935281825601E+05f, 3.8623876433339820E+05f, -8.1634960962672008E+05f, 6.4713900469564367E+05f, -2.5133289627502396E+05f, 1.7117057951236206E+04f, 2.6333352581335013E+04f, -1.0378455846609291E+04f, 1.2019749667911419E+03f,
		3.1189837632471693E+01f, -8.9083493807061564E+02f, 4.9454293649337906E+03f, -1.3124693635095375E+04f, 1.5834784331991095E+04f, 6.9607870364081436E+03f, -5.9789871879430451E+04f, 1.0841726514394575E+05f, -1.0841709685990328E+05f, 5.9790206615067997E+04f, -6.9607049368128291E+03f, -1.5834783935893831E+04f, 1.3124692974990443E+04f, -4.9454295091588992E+03f, 8.9083493794871868E+02f, -3.1189837631106176E+01f,
		-1.2975319073401824E+01f, 1.8283698218710011E+01f, 1.7684015393859755E+02f, -1.1059917445033070E+03f, 3.1998168298121523E+03f, -5.5988200120063057E+03f, 5.9248751921324047E+03f, -2.5990022806343668E+03f, -2.5990962125709430E+03f, 5.9247537039895724E+03f, -5.5988835070734467E+03f, 3.1998292349030621E+03f, -1.1059926481090836E+03f, 1.7684013881079576E+02f, 1.8283698123134819E+01f, -1.2975319073977776E+01f,
		-2.3155118729954247E+00f, 1.1938503634469159E+01f, -3.4150562973753665E+01f, 4.8898615554511437E+01f, 1.5853185548633874E+01f, -2.4272678107130790E+02f, 6.0151276286907887E+02f, -8.8751856926690448E+02f, 8.8742942550355474E+02f, -6.0136491467620624E+02f, 2.4282489356694586E+02f, -1.5850195971204462E+01f, -4.8897392545563044E+01f, 3.4150562973753665E+01f, -1.1938504430698943E+01f, 2.3155118723150525E+00f,
		-1.5401723686076832E-01f, 9.8067823888634464E-01f, -4.1900843552415639E+00f, 1.2150534299778382E+01f, -2.4763139606227178E+01f, 3.6068014621628578E+01f, -3.4346647779134791E+01f, 1.3259903958585387E+01f, 1.2937147675617604E+01f, -3.4454233206790519E+01f, 3.6027670086257579E+01f, -2.4769863695455662E+01f, 1.2149431128889342E+01f, -4.1901615115388706E+00f, 9.8067695636810759E-01f, -1.5401723756214594E-01f,
		1.1808835093099178E-02f, -2.5444299558662394E-02f, -1.5661344238792723E-04f, 2.5820071204205225E-01f, -1.0930950485268096E+00f, 2.6408492552008669E+00f, -4.4415763059111955E+00f, 6.8227366238712817E+00f, -6.8186662643534008E+00f, 4.4887924763186051E+00f, -2.6327085361651021E+00f, 1.0918739406714428E+00f, -2.5844238963842503E-01f, 1.2680123888735934E-04f, 2.5444206395526567E-02f, -1.1808834826225629E-02f };

	if (!(opts.flags & TF_OMIT_EVALUATE_KERNEL)) {
		// insert the auto-generated code which expects z, w args, writes to ker...
		if (opts.upsampfac == 2.0) {     // floating point equality is fine here
			if (w == 2) {
				eval_kernel_bulk_generic<2, 5>(c2002, kernel_vals, x1, size);
			}
			else if (w == 3) {
				eval_kernel_bulk_generic<3, 6>(c2003, kernel_vals, x1, size);
			}
			else if (w == 4) {
				eval_kernel_bulk_generic<4, 7>(c2004, kernel_vals, x1, size);
			}
			else if (w == 5) {
				eval_kernel_bulk_generic<5, 8>(c2005, kernel_vals, x1, size);
			}
			else if (w == 6) {
				eval_kernel_bulk_generic<6, 9>(c2006, kernel_vals, x1, size);
			}
			else if (w == 7) {
				eval_kernel_bulk_generic<7, 10>(c2007, kernel_vals, x1, size);
			}
			else if (w == 8) {
				eval_kernel_bulk_generic<8, 11>(c2008, kernel_vals, x1, size);
			}
			else if (w == 9) {
				eval_kernel_bulk_generic<9, 11>(c2009, kernel_vals, x1, size);
			}
			else if (w == 10) {
				eval_kernel_bulk_generic<10, 12>(c200a, kernel_vals, x1, size);
			}
			else if (w == 11) {
				eval_kernel_bulk_generic<11, 13>(c200b, kernel_vals, x1, size);
			}
			else if (w == 12) {
				eval_kernel_bulk_generic<12, 14>(c200c, kernel_vals, x1, size);
			}
			else if (w == 13) {
				eval_kernel_bulk_generic<13, 15>(c200d, kernel_vals, x1, size);
			}
			else if (w == 14) {
				eval_kernel_bulk_generic<14, 16>(c200e, kernel_vals, x1, size);
			}
			else if (w == 15) {
				eval_kernel_bulk_generic<15, 17>(c200f, kernel_vals, x1, size);
			}
			else if (w == 16) {
				eval_kernel_bulk_generic<16, 18>(c200g, kernel_vals, x1, size);
			}
			else
				printf("width not implemented!\n");
		}
		else if (opts.upsampfac == 1.25) {
			if (w == 2) {
				eval_kernel_bulk_generic<2, 4>(c1252, kernel_vals, x1, size);
			}
			else if (w == 3) {
				eval_kernel_bulk_generic<3, 5>(c1253, kernel_vals, x1, size);
			}
			else if (w == 4) {
				eval_kernel_bulk_generic<4, 6>(c1254, kernel_vals, x1, size);
			}
			else if (w == 5) {
				eval_kernel_bulk_generic<5, 7>(c1255, kernel_vals, x1, size);
			}
			else if (w == 6) {
				eval_kernel_bulk_generic<6, 8>(c1256, kernel_vals, x1, size);
			}
			else if (w == 7) {
				eval_kernel_bulk_generic<7, 9>(c1257, kernel_vals, x1, size);
			}
			else if (w == 8) {
				eval_kernel_bulk_generic<8, 10>(c1258, kernel_vals, x1, size);
			}
			else if (w == 9) {
				eval_kernel_bulk_generic<9, 10>(c1259, kernel_vals, x1, size);
			}
			else if (w == 10) {
				eval_kernel_bulk_generic<10, 11>(c125a, kernel_vals, x1, size);
			}
			else if (w == 11) {
				eval_kernel_bulk_generic<11, 12>(c125b, kernel_vals, x1, size);
			}
			else if (w == 12) {
				eval_kernel_bulk_generic<12, 13>(c125c, kernel_vals, x1, size);
			}
			else if (w == 13) {
				eval_kernel_bulk_generic<13, 14>(c125d, kernel_vals, x1, size);
			}
			else if (w == 14) {
				eval_kernel_bulk_generic<14, 15>(c125e, kernel_vals, x1, size);
			}
			else if (w == 15) {
				eval_kernel_bulk_generic<15, 16>(c125f, kernel_vals, x1, size);
			}
			else if (w == 16) {
				eval_kernel_bulk_generic<16, 17>(c125g, kernel_vals, x1, size);
			}
			else
				printf("width not implemented!\n");
		}
		else
			fprintf(stderr, "%s: unknown upsampfac, failed!\n", __func__);
	}
}

FLT evaluate_kernel(FLT x, const spread_opts& opts)
/* ES ("exp sqrt") kernel evaluation at single real argument:
	  phi(x) = exp(beta.sqrt(1 - (2x/n_s)^2)),    for |x| < nspread/2
   related to an asymptotic approximation to the Kaiser--Bessel, itself an
   approximation to prolate spheroidal wavefunction (PSWF) of order 0.
   This is the "reference implementation", used by eg finufft/onedim_* 2/17/17
*/
{
	if (abs(x) >= opts.ES_halfwidth)
		// if spreading/FT careful, shouldn't need this if, but causes no speed hit
		return 0.0;
	else
		return exp(opts.ES_beta * sqrt(1.0 - opts.ES_c * x * x));
}

static inline void set_kernel_args(FLT* args, FLT x, const spread_opts& opts)
// Fills vector args[] with kernel arguments x, x+1, ..., x+ns-1.
// needed for the vectorized kernel eval of Ludvig af K.
{
	int ns = opts.nspread;
	for (int i = 0; i < ns; i++)
		args[i] = x + (FLT)i;
}

static inline void evaluate_kernel_vector(FLT* ker, FLT* args, const spread_opts& opts, const int N)
/* Evaluate ES kernel for a vector of N arguments; by Ludvig af K.
   If opts.kerpad true, args and ker must be allocated for Npad, and args is
   written to (to pad to length Npad), only first N outputs are correct.
   Barnett 4/24/18 option to pad to mult of 4 for better SIMD vectorization.

   Obsolete (replaced by Horner), but keep around for experimentation since
   works for arbitrary beta. Formula must match reference implementation. */
{
	FLT b = opts.ES_beta;
	FLT c = opts.ES_c;
	if (!(opts.flags & TF_OMIT_EVALUATE_KERNEL)) {
		// Note (by Ludvig af K): Splitting kernel evaluation into two loops
		// seems to benefit auto-vectorization.
		// gcc 5.4 vectorizes first loop; gcc 7.2 vectorizes both loops
		int Npad = N;
		if (opts.kerpad) {        // since always same branch, no speed hit
			Npad = 4 * (1 + (N - 1) / 4);   // pad N to mult of 4; help i7 GCC, not xeon
			for (int i = N; i < Npad; ++i)    // pad with 1-3 zeros for safe eval
				args[i] = 0.0;
		}
		for (int i = 0; i < Npad; i++) { // Loop 1: Compute exponential arguments
			ker[i] = b * sqrt(1.0 - c * args[i] * args[i]);
		}
		if (!(opts.flags & TF_OMIT_EVALUATE_EXPONENTIAL))
			for (int i = 0; i < Npad; i++) // Loop 2: Compute exponentials
				ker[i] = exp(ker[i]);
	}
	else {
		for (int i = 0; i < N; i++)             // dummy for timing only
			ker[i] = 1.0;
	}
	// Separate check from arithmetic (Is this really needed? doesn't slow down)
	for (int i = 0; i < N; i++)
		if (abs(args[i]) >= opts.ES_halfwidth) ker[i] = 0.0;
}

static inline void eval_kernel_vec_Horner(FLT* ker, const FLT x, const int w,
	const spread_opts& opts)
	/* Fill ker[] with Horner piecewise poly approx to [-w/2,w/2] ES kernel eval at
	   x_j = x + j,  for j=0,..,w-1.  Thus x in [-w/2,-w/2+1].   w is aka ns.
	   This is the current evaluation method, since it's faster (except i7 w=16).
	   Two upsampfacs implemented. Params must match ref formula. Barnett 4/24/18 */
{
	if (!(opts.flags & TF_OMIT_EVALUATE_KERNEL)) {
		FLT z = 2 * x + w - 1.0;         // scale so local grid offset z in [-1,1]
		// insert the auto-generated code which expects z, w args, writes to ker...
		if (opts.upsampfac == 2.0) {     // floating point equality is fine here
#include "ker_horner_allw_loop.c"
		}
		else if (opts.upsampfac == 1.25) {
#include "ker_lowupsampfac_horner_allw_loop.c"
		}
		else
			fprintf(stderr, "%s: unknown upsampfac, failed!\n", __func__);
	}
}

void evaluate_kernel(FLT* kernel_vals, FLT* x, const BIGINT begin, const BIGINT end, spread_opts& opts)
{
	int ns = opts.nspread; // kernel width
	int nsPadded = 4 * (1 + (ns - 1) / 4); // pad ns to mult of 4

	if (opts.kerevalmeth == 0) {
		alignas(64) FLT kernel_args[MAX_NSPREAD];

		for (BIGINT i = begin; i < end; i++)
		{
			set_kernel_args(kernel_args, x[i], opts);
			evaluate_kernel_vector(kernel_vals + i * nsPadded, kernel_args, opts, ns);
		}
	}
	else {
		eval_kernel_bulk_Horner(kernel_vals + begin * nsPadded, x + begin, ns, end - begin, opts);
	}
}

