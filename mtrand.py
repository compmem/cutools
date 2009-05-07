#
#
#

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy as np
import struct


#mod = SourceModule(file('mtrand.cu.h','r').read())

mod = SourceModule(file('mtrand.cu.h','r').read()+
    """
__global__ void cu_rand(float *x, int N)
{
  unsigned int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (idx < N)
  {
    // initialize the MT
    MersenneTwisterState mtState;
    MersenneTwisterInitialise(mtState, idx);
    x[idx] = (float)MersenneTwisterGenerate(mtState, idx) / 4294967295.0f;
  }
}

    """)

cu_rand = mod.get_function("cu_rand")
cu_set_seed = mod.get_function("MersineTwisterSetSeed")

# init the twister with the dat

# # to do the seed at same time
# mtfile = file('MersenneTwister.dat','rb')
# nseeds = 32768
# mtinit = ''
# for i in xrange(nseeds):
#     mt = list(struct.unpack('IIII',mtfile.read(16)))
#     mt[3] = np.random.randint(0,high=4294967296)
#     mtinit += struct.pack('IIII',*mt)

mtinit = file('MersenneTwister.dat','rb').read()
nseeds = len(mtinit)/struct.calcsize('IIII')

cuMT = mod.get_global("MT")
if cuMT[1] != len(mtinit):
    raise ValueError("The dat file is not the same size as the internal global var.")
cuda.memcpy_htod(cuMT[0],mtinit)

# set the seed
seeds = np.asarray(np.random.randint(0, high=4294967296, size=nseeds), dtype=np.uint32)
cu_set_seed(cuda.In(seeds),np.int32(nseeds), block=(1,1,1))

asize = 10
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
print a

