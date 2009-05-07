
/* 
   Some simple matrix helper functions that can run on the
   cuda-enabled device.
 */

#ifndef CUDA_MATRIX_CU_H
#define CUDA_MATRIX_CU_H


__device__ void mmult(unsigned int m, unsigned int n, unsigned int k, 
		      float alpha, const float *A, const float *B,
		      float beta, float *C)
{
    int i;
    int j;
    int v;
    for (i = 0; i < m; ++i) {
        for (j = 0; j < n; ++j) {
            float prod = 0;
            for (v = 0; v < k; ++v) {
                prod += A[i * k + v] * B[v * n + j];
            }
            C[i * n + j] = alpha * prod + beta * C[i * n + j];
        }
    }
}


#endif
