import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy as np

from cuda_utils import mt_rand, matrix

mod = SourceModule(
    """
    #include "mt_rand.cu.h"
    #include "matrix.cu.h"

__global__ void cu_mat_test(unsigned int m, unsigned int k, unsigned int n,
                            float *A, float *B, float *C)
{
  // get the thread id
  unsigned int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

  // initialize the MT
  MersenneTwisterState mtState;
  MersenneTwisterInitialise(mtState, idx);

  mmult(m, n, k, 
	1.0, A, B,
	0.0, C);
}

__global__ void cu_mat_test2(float *C)
{
  // get the thread id
  unsigned int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

  // initialize the MT
  MersenneTwisterState mtState;
  MersenneTwisterInitialise(mtState, idx);

  const unsigned int m = 2;
  const unsigned int k = 5;
  const unsigned int n = 10;

  float A[m*k];
  float B[k*n];

  // set some values for A
  for (int i=0; i<k; i++)
  {
    for (int j=0; j<m; j++)
    {
      if (j==0)
        A[j*k+i] = mt_rand(mtState, idx);
      else
        A[j*k+i] = 0.0;
    }
  }

  // set some values for B
  for (int i=0; i<n; i++)
  {
    for (int j=0; j<k; j++)
    {
      if (i==4)
        B[j*k+i] = mt_rand(mtState, idx);
      else
        B[j*k+i] = 0.0;
    }
  }

  mmult(m, n, k, 
	1.0, A, B,
	0.0, C);
}


    """,
    include_dirs=[mt_rand.get_include_dir(),
                  matrix.get_include_dir()])

# seed the random number generator
mt_rand.seed(cuda,mod)

cu_mmult = mod.get_function("cu_mat_test")
cu_mmult2 = mod.get_function("cu_mat_test2")

m = np.uint32(2)
k = np.uint32(5)
n = np.uint32(10)

A = np.zeros((m,k),dtype=np.float32)
B = np.zeros((k,n),dtype=np.float32)
A[0,:] = np.random.rand(k)
B[:,4] = np.random.rand(k)

C = np.empty((2,10),dtype=np.float32)
Cr = C.ravel()
#cC = cuda.mem_alloc(C.nbytes)
cu_mmult(m, k, n, cuda.In(A.ravel()), cuda.In(B.ravel()), cuda.Out(Cr),block=(1,1,1))
#cu_mmult2(cuda.Out(Cr),block=(1,1,1))


#__global__ void cu_mat_test(unsigned int m, unsigned int k, unsigned int n,
#                            float *A, float *B, float *C)

print Cr.reshape((2,10))
#print np.dot(A,B)
