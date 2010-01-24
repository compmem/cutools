
/* 
   Some simple matrix helper functions that can run on the
   cuda-enabled device.
 */

#ifndef MATRIX_CU_H
#define MATRIX_CU_H

// some handy defines
#define cidx(nrows, r, ncols, c) (c*nrows+r)
#define ridx(nrows, r, ncols, c) (r*ncols+c)
#define idx(r,c,ncols) (r*ncols+c)


__device__ void madd(char transa, char transb, 
		     unsigned int m, unsigned int n,
		     float alpha, const float *A,
		     float beta, const float *B, float *C)
{
  unsigned int i;
  unsigned int j;

  if (transa != 't')
  {
    // not transposing A
    if (transb != 't')
    {
      // not transposing B
      for (i = 0; i < m; ++i) {
	for (j = 0; j < n; ++j) {
	  C[ridx(m,i,n,j)] = alpha * A[ridx(m,i,n,j)] + beta * B[ridx(m,i,n,j)];
	}
      }
    }
    else
    {
      // transposing B
      for (i = 0; i < m; ++i) {
	for (j = 0; j < n; ++j) {
	  C[ridx(m,i,n,j)] = alpha * A[ridx(m,i,n,j)] + beta * B[cidx(m,i,n,j)];
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
	  C[ridx(m,i,n,j)] = alpha * A[cidx(m,i,n,j)] + beta * B[ridx(m,i,n,j)];
	}
      }
    }
    else
    {
      // transposing B
      for (i = 0; i < m; ++i) {
	for (j = 0; j < n; ++j) {
	  C[ridx(m,i,n,j)] = alpha * A[cidx(m,i,n,j)] + beta * B[cidx(m,i,n,j)];
	}
      }
    }
  }
}

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


__device__ void mset(float *mat, float val, const unsigned int nrows, const unsigned int ncols)
{
  for (unsigned int i=0; i<nrows*ncols; i++)
  {
    mat[i] = val;
  }
}

__device__ void zeros(float *mat, const unsigned int nrows, const unsigned int ncols)
{
  mset(mat, 0.0, nrows, ncols);
}

__device__ void ones(float *mat, const unsigned int nrows, const unsigned int ncols)
{
  mset(mat, 1.0, nrows, ncols);
}

__device__ void nans(float *mat, const unsigned int nrows, const unsigned int ncols)
{
  mset(mat, nanf(""), nrows, ncols);
}

__device__ void mcopy(float *src, float *dest, 
		      const unsigned int nrows, const unsigned int ncols)
{
  for (unsigned int i=0; i<nrows*ncols; i++)
  {
    dest[i] = src[i];
  }
}

#endif
