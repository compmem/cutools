// This file is derived from the NVIDIA CUDA SDK example 'MersenneTwister'.
// We make use of the expanded dat file from:
// http://www.jcornwall.me.uk/2009/04/mersenne-twisters-in-cuda/


/*
 * Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

#ifndef MTRAND_CU_H
#define MTRAND_CU_H

#define MT_RNG_COUNT 32768

// Record format for MersenneTwister.dat, created by spawnTwisters.c
struct mt_struct_stripped {
  unsigned int matrix_a;
  unsigned int mask_b;
  unsigned int mask_c;
  unsigned int seed;
};


#define MT_MM     9
#define MT_NN     19
#define MT_WMASK  0xFFFFFFFFU
#define MT_UMASK  0xFFFFFFFEU
#define MT_LMASK  0x1U
#define MT_SHIFT0 12
#define MT_SHIFTB 7
#define MT_SHIFTC 15
#define MT_SHIFT1 18

struct MersenneTwisterState {
  unsigned int mt[MT_NN];
  int iState;
  unsigned int mti1;
  unsigned int has_randn_val;
  float randn_val;
};

__device__ static mt_struct_stripped MT[MT_RNG_COUNT];

//__global__ void MersineTwisterSetSeed(unsigned int *seeds, int N)
__global__ void MersineTwisterSetSeed(unsigned int *seeds)
{
  //for (int i=0; i<N; i++)
  /* for (int i=0; i<MT_RNG_COUNT; i++) */
  /* { */
  /*   MT[i].seed = seeds[i]; */
  /* } */

  // get the thread id
  unsigned int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (idx < MT_RNG_COUNT)
    MT[idx].seed = seeds[idx];
}

__device__ void MersenneTwisterInitialise(MersenneTwisterState &state, unsigned int threadID) {
  state.mt[0] = MT[threadID].seed;
  for(int i = 1; i < MT_NN; ++ i) {
    state.mt[i] = (1812433253U * (state.mt[i - 1] ^ (state.mt[i - 1] >> 30)) + i) & MT_WMASK;
  }

  state.iState = 0;
  state.mti1 = state.mt[0];
	
  // does not have randn to start
  state.has_randn_val = 0;
}

__device__ unsigned int MersenneTwisterGenerate(MersenneTwisterState &state, unsigned int threadID) {
  int iState1 = state.iState + 1;
  int iStateM = state.iState + MT_MM;

  if(iState1 >= MT_NN) iState1 -= MT_NN;
  if(iStateM >= MT_NN) iStateM -= MT_NN;

  unsigned int mti = state.mti1;
  state.mti1 = state.mt[iState1];
  unsigned int mtiM = state.mt[iStateM];

  unsigned int x = (mti & MT_UMASK) | (state.mti1 & MT_LMASK);
  x = mtiM ^ (x >> 1) ^ ((x & 1) ? MT[threadID].matrix_a : 0);
  state.mt[state.iState] = x;
  state.iState = iState1;

  // Tempering transformation.
  x ^= (x >> MT_SHIFT0);
  x ^= (x << MT_SHIFTB) & MT[threadID].mask_b;
  x ^= (x << MT_SHIFTC) & MT[threadID].mask_c;
  x ^= (x >> MT_SHIFT1);

  return x;
}

__device__ float mt_rand(MersenneTwisterState &state, unsigned int threadID) 
{
  return (float)MersenneTwisterGenerate(state, threadID) / 4294967295.0f;
}

////////////////////////////////////////////////////////////////////////////////
// Transform each of MT_RNG_COUNT lanes of NPerRng uniformly distributed 
// random samples, produced by RandomGPU(), to normally distributed lanes
// using Cartesian form of Box-Muller transformation.
// NPerRng must be even.
////////////////////////////////////////////////////////////////////////////////
#define PI 3.14159265358979f
__device__ void BoxMuller(float& u1, float& u2){
    float   r = sqrtf(-2.0f * logf(u1));
    float phi = 2 * PI * u2;
    u1 = r * __cosf(phi);
    u2 = r * __sinf(phi);
}

__device__ float mt_randn(MersenneTwisterState &state, unsigned int threadID) 
{
  if (state.has_randn_val == 1)
  {
    // use it
    state.has_randn_val = 0;
    return state.randn_val;
  }
  else
  {
    // generate two and return one
    float u1, u2;
    u1 = mt_rand(state, threadID);
    u2 = mt_rand(state, threadID);
    BoxMuller(u1, u2);

    // keep one for the state
    state.has_randn_val = 1;
    state.randn_val = u2;

    // return the other one
    return u1;
  }
}

__device__ void rands(MersenneTwisterState &state, unsigned int threadID,
		      float *mat, const unsigned int nrows, const unsigned int ncols)
{
  for (unsigned int i=0; i<nrows*ncols; i++)
  {
    mat[i] = mt_rand(state, threadID);
  }
}

__device__ void randns(MersenneTwisterState &state, unsigned int threadID,
		       float *mat, const unsigned int nrows, const unsigned int ncols)
{
  for (unsigned int i=0; i<nrows*ncols; i++)
  {
    mat[i] = mt_randn(state, threadID);
  }
}


#endif
