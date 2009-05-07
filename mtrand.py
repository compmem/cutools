#
#
#

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy as np
import struct



#mod = SourceModule(file('mtrand.cu.h','r').read())

mod = SourceModule(
    """
#include "mtrand.cu.h"

__global__ void cu_rand(float *x, int N)
{
  unsigned int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (idx < N)
  {
    // initialize the MT
    MersenneTwisterState mtState;
    MersenneTwisterInitialise(mtState, idx);
    x[idx] = (float)MersenneTwisterGenerate(mtState, thrOffset) / 4294967295.0f;
  }
}

    """)

cu_rand = mod.get_function("cu_rand")



