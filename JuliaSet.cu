//======================================================================================================================
//
// CUDA version of Julia set calculation
//
//! \file JuliaSet.cu
//! \author Sebastian Eibl <sebastian.eibl@fau.de>
//
//======================================================================================================================

#include "lodepng/lodepng.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>
#include <string>

typedef float real_t;

template <typename T>
real_t real_c(T val) { return static_cast<real_t>(val); }

#define CHECK(call)                                                                                                   \
{                                                                                                                     \
   const cudaError_t error = (call);                                                                                  \
   if (error != cudaSuccess)                                                                                          \
   {                                                                                                                  \
      std::cout << "Error: " << __FILE__ << ":"  << __LINE__ << std::endl;                                            \
      std::cout << "code: " << error << ", reason: " << cudaGetErrorString(error) << std::endl;                       \
      exit(1);                                                                                                        \
   }                                                                                                                  \
}

__global__ void juliaGPU( const real_t cr,
                          const real_t ci,
                          const real_t originX,
                          const real_t originY,
                          const real_t width,
                          const real_t height,
                          const int numX,
                          const int numY,
                          const int ld,
                          int* level,
                          unsigned char* RGBpic)
{
   const int idx = blockIdx.x * blockDim.x + threadIdx.x;
   const int idy = blockIdx.y * blockDim.y + threadIdx.y;

   if ((idx < 2048) && (idy < 2048))
   {
      const int id = ld * idy + idx;
      const real_t x = originX + width  * ((real_t)(idx) / (real_t)(numX));
      const real_t y = originY + height * ((real_t)(idy) / (real_t)(numY));
      real_t z0r = x;
      real_t z0i = y;

      int i;
      for (i = 0; i < 100; ++i)
      {
         const real_t zrTemp = z0r;
         z0r = (z0r * z0r - z0i * z0i) + cr;
         z0i = 2 * zrTemp * z0i + ci;
         if ((z0r * z0r + z0i * z0i) > 100.0f)
         {
            break;
         }
      }
      level[id] = i;
   }
}

__global__ void findMinMax(const int n,
                           const int* level,
                           int* min,
                           int* max)
{
   __shared__ int sMin[512];
   __shared__ int sMax[512];
   const unsigned int tid = threadIdx.x;
   unsigned int i = blockIdx.x*(blockDim.x)+tid;
   unsigned int gridSize = blockDim.x * gridDim.x;
   sMin[tid] = level[i];
   sMax[tid] = level[i];

   i += gridSize;
   while (i<n)
   {
      if (sMin[tid] > level[i]) sMin[tid] = level[i];
      if (sMax[tid] < level[i]) sMax[tid] = level[i];
      i += gridSize;
   }

   __syncthreads();

   for (unsigned int s = blockDim.x/2; s>0; s>>=1)
   {
      if (tid < s)
      {
         if (sMin[tid] > sMin[tid + s]) sMin[tid] = sMin[tid + s];
         if (sMax[tid] < sMax[tid + s]) sMax[tid] = sMax[tid + s];
      }
      __syncthreads();
   }

   if (tid == 0)
   {
      min[blockIdx.x] = sMin[0];
      max[blockIdx.x] = sMax[0];
   }
}

__global__ void colorPicture( const int ld, const int min, const int max, const int* level, unsigned char* RGBpic) {
   const int idx = blockIdx.x * blockDim.x + threadIdx.x;
   const int idy = blockIdx.y * blockDim.y + threadIdx.y;

   if ((idx < 2048) && (idy < 2048))
   {
      const int id = ld * idy + idx;
      const real_t frac = static_cast<real_t>(level[id] - min) / static_cast<real_t>(max-min);
      RGBpic[id * 3 + 0] = static_cast<unsigned char> (static_cast<real_t>(255) * (1 - (1-frac) * (1-frac) * (1-frac) * (1-frac) * (1-frac) * (1-frac)) );
   }
}

int main(int argc, char **argv) {
   if (argc != 3)
   {
      std::cout << "./01_Julia blockDimX blockDimY" << std::endl;
      exit(EXIT_FAILURE);
   }
   const unsigned int size = 2048;
   const unsigned int blockDimX = atoi(argv[1]);
   const unsigned int blockDimY = atoi(argv[2]);

   // set up device
   cudaSetDevice(0);

   std::vector<unsigned char> RGBpic;
   RGBpic.resize(size*size*3);
   for (int i = 0; i<RGBpic.size(); ++i)
   {
      RGBpic[i] = 0;
   }

   int* d_level;
   CHECK( cudaMalloc((int**) &d_level, size * size * sizeof(int)) );
   std::vector<int> h_min(1024);
   int* d_min;
   CHECK( cudaMalloc((int**) &d_min, 1024 * sizeof(int)) );
   std::vector<int> h_max(1024);
   int* d_max;
   CHECK( cudaMalloc((int**) &d_max, 1024 * sizeof(int)) );
   unsigned char* d_pic;
   CHECK( cudaMalloc((unsigned char**) &d_pic, RGBpic.size()) );
   CHECK( cudaMemcpy(d_pic, &RGBpic[0], RGBpic.size(), cudaMemcpyHostToDevice) );
   dim3 block( blockDimX, blockDimY, 1 );
   dim3 grid ( (2048+block.x-1)/block.x, (2048+block.y-1)/block.y, 1);
   juliaGPU<<< grid, block >>> (-0.0, 0.8, real_c(-2), real_c(-2), real_c(4), real_c(4), size, size, size, d_level, d_pic);
   CHECK( cudaPeekAtLastError() );
   CHECK( cudaDeviceSynchronize() );
   findMinMax<<< 1024, 512 >>> (size*size, d_level, d_min, d_max);
   CHECK( cudaMemcpy(&h_min[0], d_min, h_min.size() * sizeof(int), cudaMemcpyDeviceToHost) );
   CHECK( cudaMemcpy(&h_max[0], d_max, h_max.size() * sizeof(int), cudaMemcpyDeviceToHost) );
   std::cout << "min: " << *std::min_element(h_min.begin(), h_min.end()) << "\tmax: " << *std::max_element(h_max.begin(), h_max.end()) << std::endl;
   colorPicture<<< grid, block >>> (size, *std::min_element(h_min.begin(), h_min.end()), *std::max_element(h_max.begin(), h_max.end()), d_level, d_pic);
   CHECK( cudaMemcpy(&RGBpic[0], d_pic, RGBpic.size(), cudaMemcpyDeviceToHost) );
   lodepng::encode("julia.png", RGBpic, size, size, LCT_RGB);
   CHECK( cudaFree(d_level) );
   CHECK( cudaFree(d_pic) );
   return(0);
}
