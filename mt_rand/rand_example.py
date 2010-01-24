import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy as np

from cutools import mt_rand

mod = SourceModule(
    """
#include "mt_rand.cu.h"
__global__ void cu_rand_test(float *x, int N)
{
  unsigned int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (idx < N)
  {
    // initialize the MT
    MersenneTwisterState mtState;
    MersenneTwisterInitialise(mtState, idx);

    //
    x[idx] = mt_rand(mtState, idx);
  }
}

    """,
    include_dirs=[mt_rand.get_include_dir()])

# seed the random number generator
mt_rand.seed(cuda,mod)

cu_rand = mod.get_function("cu_rand_test")

asize = 10000
bsize = 16
a = np.zeros((asize,), dtype=np.float32)
ac = cuda.mem_alloc(a.nbytes)
#cuda.memcpy_htod(ac,a)
block = (bsize,1,1)
cu_rand.set_block_shape(*block)
cu_rand.param_set(ac, np.int32(asize))
gsize = (asize/bsize)+1
cu_rand.launch_grid(gsize,1)
cuda.memcpy_dtoh(a,ac)
print a[:10]
