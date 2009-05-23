#
#
#

import numpy as np
import struct
import os

UINT32_MAX = 4294967295

def get_include_dir():
    return os.path.dirname(os.path.abspath(__file__))

def seed(cuda, mod, seeds=None):
    """
    Set the seeds for the Mersenne Twisters.
    """
    # get the set seed function
    cu_set_seed = mod.get_function("MersineTwisterSetSeed")

    # load in the precalculated lookup table file
    datfile = os.path.join(get_include_dir(),'MersenneTwister.dat')
    mtinit = file(datfile,'rb').read()
    nseeds = len(mtinit)/struct.calcsize('IIII')

    # get and set the global lookup table on the card
    cuMT = mod.get_global("MT")
    if cuMT[1] != len(mtinit):
        raise ValueError("The dat file is not the same size as the internal global var.")
    cuda.memcpy_htod(cuMT[0],mtinit)

    # set the seed to random starting values
    if seeds is None:
        # generate them
        seeds = np.asarray(np.random.randint(0, high=UINT32_MAX+1, size=nseeds), dtype=np.uint32)

    # set the seeds
    #cu_set_seed(cuda.In(seeds),np.int32(nseeds), block=(1,1,1))
    #cu_set_seed(cuda.In(seeds), block=(1,1,1))

    # call it in parallel
    bsize = 32
    gsize = (nseeds/bsize)
    if gsize*bsize < nseeds:
        gsize += 1
    cu_set_seed(cuda.In(seeds), block=(bsize,1,1), grid=(gsize,1))
    
    return seeds


# Inefficient way...
# init the twister with the dat

# # to do the seed at same time
# mtfile = file('MersenneTwister.dat','rb')
# nseeds = 32768
# mtinit = ''
# for i in xrange(nseeds):
#     mt = list(struct.unpack('IIII',mtfile.read(16)))
#     mt[3] = np.random.randint(0,high=4294967296)
#     mtinit += struct.pack('IIII',*mt)

