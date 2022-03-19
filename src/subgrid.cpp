#include <dataTypes.h>

void get_subgrid(BIGINT& offset, BIGINT& size, BIGINT* idx, BIGINT M, int ns)
{
	BIGINT min_idx = idx[0], max_idx = idx[0];
	for (BIGINT i = 1; i < M; i++) {
		if (min_idx > idx[i])
			min_idx = idx[i];
		if (max_idx < idx[i])
			max_idx = idx[i];
	}

	offset = min_idx;
	size = max_idx - min_idx + ns;
}
