
/* 
   Some simple matrix helper functions that can run on the
   cuda-enabled device.
 */

#ifndef CUDA_MATRIX_CU_H
#define CUDA_MATRIX_CU_H


// some handy defines
#define cidx(nrows, r, ncols, c) (c*nrows+r)
#define ridx(nrows, r, ncols, c) (r*ncols+c)
#define idx(r,c,ncols) (r*ncols+c)

__device__ void mmult(char transa, char transb, 
		      unsigned int m, unsigned int n, unsigned int k, 
		      float alpha, const float *A, const float *B,
		      float beta, float *C)
{
 
  // A(m,k) * B(k,n) = C(m,n)
  unsigned int i;
  unsigned int j;
  unsigned int v;

  if (transa != 't')
  {
    // not transposing A
    if (transb != 't')
    {
      // not transposing B
      for (i = 0; i < m; ++i) {
	for (j = 0; j < n; ++j) {
	  float prod = 0;
	  for (v = 0; v < k; ++v) {
	    prod += A[ridx(m,i,k,v)] * B[ridx(k,v,n,j)];
	  }
	  C[ridx(m,i,n,j)] = alpha * prod + beta * C[ridx(m,i,n,j)];
	}
      }
    }
    else
    {
      // transposing B
      for (i = 0; i < m; ++i) {
	for (j = 0; j < n; ++j) {
	  float prod = 0;
	  for (v = 0; v < k; ++v) {
	    prod += A[ridx(m,i,k,v)] * B[cidx(k,v,n,j)];
	  }
	  C[ridx(m,i,n,j)] = alpha * prod + beta * C[ridx(m,i,n,j)];
	}
      }
    }
  }
  else
  {
    // transposing A
    if (transb != 't')
    {
      // not transposing B
      for (i = 0; i < m; ++i) {
	for (j = 0; j < n; ++j) {
	  float prod = 0;
	  for (v = 0; v < k; ++v) {
	    prod += A[cidx(m,i,k,v)] * B[ridx(k,v,n,j)];
	  }
	  C[ridx(m,i,n,j)] = alpha * prod + beta * C[ridx(m,i,n,j)];
	}
      }
    }
    else
    {
      // transposing B
      for (i = 0; i < m; ++i) {
	for (j = 0; j < n; ++j) {
	  float prod = 0;
	  for (v = 0; v < k; ++v) {
	    prod += A[cidx(m,i,k,v)] * B[cidx(k,v,n,j)];
	  }
	  C[ridx(m,i,n,j)] = alpha * prod + beta * C[ridx(m,i,n,j)];
	}
      }
    }
  }
}


__device__ float *zeros(const unsigned int nrows, const unsigned int ncols)
{
  float mat[nrows*ncols];
  for (unsigned int i=0; i<nrows*ncols; i++)
  {
    mat[i] = 0.0;
  }

  return mat
}

#endif
