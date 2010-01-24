import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy as np
import struct

from cutools import mt_rand, matrix

mod = SourceModule(
    """
    #include "mt_rand.cu.h"
    #include "matrix.cu.h"

struct results
{
  float *C;
  unsigned int num;
};

struct source
{
  float *A;
  float *B;
  unsigned int m;
  unsigned int n;
  unsigned int k;
};

__global__ void cu_struct_test(source src, float *C)
{
  mmult('f','f', src.m, src.n, src.k, 
	1.0, src.A, src.B,
	0.0, C);
}

__global__ void cu_mat_test(unsigned int m, unsigned int n, unsigned int k,
                            float *A, float *B, results res)
{
  mmult('f','f',m, n, k, 
	1.0, A, B,
	0.0, res.C);
  res.num = 42;
}

struct two_mat
{
  float *A;
  float *B;
};

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

  // test the use of struct to hold data
  two_mat dat;
  float A[m*k];
  dat.A = A;
  zeros(dat.A,m,k);
  float B[k*n];
  dat.B = B;
  zeros(dat.B,k,n);

  // set some values for A
  rands(mtState,idx,dat.A,m,k);
  int i=0;
  int j=0;
//  for (j=0; j<k; j++)
//    dat.A[idx(i,j,k)] = mt_rand(mtState, idx);

  j = 4;
  for (i=0; i<k; i++)
    dat.B[idx(i,j,n)] = mt_rand(mtState, idx);

  mmult('t','f',m, n, k, 
	1.0, dat.A, dat.B,
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

class Results(object):

    def __init__(self):
        self._cptr = None
        self.C = np.empty((2,10),dtype=np.float32)
        self.num = np.uint32(0)
        
    def send_to_gpu(self):
        if self._cptr is None:
            self._cptr = cuda.mem_alloc(self.nbytes())
        #cuda.memcpy_htod(self._cptr, self.pack())
    def get_from_gpu(self):
        if not self._cptr is None:
            tempstr = np.array([' ']*self.nbytes())
            cuda.memcpy_dtoh(tempstr,self._cptr)
            self.C = np.fromstring(tempstr[:self.C.nbytes],
                                   dtype=self.C.dtype).reshape(self.C.shape)
            self.num = np.fromstring(tempstr[self.C.nbytes:],
                                     dtype=self.num.dtype)
    def pack(self):
        return self.C.tostring() + self.num.tostring()
    def nbytes(self):
        return self.C.nbytes + self.num.nbytes

res = Results()
res.send_to_gpu()
#C = np.empty((2,10),dtype=np.float32)
#cC = cuda.mem_alloc(C.nbytes)
#cu_mmult(m, n, k, cuda.In(A), cuda.In(B), cuda.Out(C),block=(1,1,1))
cu_mmult(m, n, k, cuda.In(A), cuda.In(B), res._cptr, block=(1,1,1))
#cu_mmult2(cuda.Out(C),block=(1,1,1))
res.get_from_gpu()

#__global__ void cu_mat_test(unsigned int m, unsigned int k, unsigned int n,
#                            float *A, float *B, float *C)

print res.C
print res.num
#print C
#print Cr.reshape((2,10))
#print np.dot(A,B)


pass



class Source(object):

    def __init__(self):
        self.m = np.uint32(2)
        self.k = np.uint32(5)
        self.n = np.uint32(10)

        self.A = np.zeros((self.m,self.k),dtype=np.float32)
        self.B = np.zeros((self.k,self.n),dtype=np.float32)
        self.A[0,:] = np.random.rand(self.k)
        self.B[:,4] = np.random.rand(self.k)

        self._cptr = None
        
    def send_to_gpu(self):
        if self._cptr is None:
            self._cptr = cuda.mem_alloc(self.nbytes())
        #cuda.memcpy_htod(self._cptr, self.pack())
    def get_from_gpu(self):
        if not self._cptr is None:
            tempstr = np.array([' ']*self.nbytes())
            cuda.memcpy_dtoh(tempstr,self._cptr)
            self.C = np.fromstring(tempstr[:self.C.nbytes],
                                   dtype=self.C.dtype).resize(self.C.shape)
            self.num = np.fromstring(tempstr[self.C.nbytes:],
                                     dtype=self.num.dtype)
    def pack(self):
        return self.C.tostring() + self.num.tostring()
    def nbytes(self):
        return self.C.nbytes + self.num.nbytes
