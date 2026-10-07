#include <iostream>
#include <cstdlib>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// example
#define TILE_H 8   
#define TILE_W 8   
#define TILE_C 16  

// Kernel declaration
__global__ void gemm_gpu_o4_kernel(
    const float* __restrict__ x,       // input: N x C x H x W
    const float* __restrict__ w,       // weights: C_out x C_in x KH x KW
    float* __restrict__ out,           // output: N x C x H x W
    int N, int C_in, int H, int W,
    int C_out, int KH, int KW,
    int stride, int pad,
    int out_h, int out_w
) {
    extern __shared__ float shmem[];  // shared memory for partial sums
    
    // TO DO : Tiled matrix multiplication by using shmem
    float *psum = shmem;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int oh = blockIdx.y * TILE_H + ty;
    int ow = blockIdx.x * TILE_W + tx;

    int num_c_tiles = (C_out + TILE_C - 1) / TILE_C;

    int n = blockIdx.z / num_c_tiles;
    int c_tile = blockIdx.z % num_c_tiles;

    int c_start = c_tile * TILE_C;

    for (int lc = 0; lc < TILE_C; ++lc) {
      int psum_idx = (lc * TILE_H + ty) * TILE_W + tx;
      psum[psum_idx] = 0.0f;
    }

    __syncthreads();
    if (n < N && oh < out_h && ow < out_w) {
      for (int ci = 0; ci < C_in; ++ci) {
        for (int kh = 0; kh < KH; ++kh) {
          for (int kw = 0; kw < KW; ++kw) {
            int ih = oh * stride + kh - pad;
            int iw = ow * stride + kw - pad;

            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
              float x_val = x[((n * C_in + ci) * H + ih) * W + iw];
              for (int lc = 0; lc < TILE_C; ++lc) {
                int co = c_start + lc;
                if (co < C_out) {
                  float w_val = w[((co * C_in + ci) * KH + kh) * KW + kw];
                  int psum_idx = (lc * TILE_H + ty) * TILE_W + tx;
                  psum[psum_idx] += x_val * w_val;
              }
              }
            }
          }
        }
      }
    }
    __syncthreads();


    if (n < N && oh < out_h && ow < out_w) {
      for (int lc = 0; lc < TILE_C; ++lc) {
        int co = c_start + lc;
        if (co < C_out) {
          int psum_idx = (lc * TILE_H + ty) * TILE_W + tx;

          int out_idx = ((n * C_out + co) * out_h + oh) * out_w + ow;

          out[out_idx] = psum[psum_idx];
      }
      }
    }

    

}

// Function for Python binding
torch::Tensor conv_cuda(torch::Tensor x, torch::Tensor w,
                          int stride, int pad) {
    int N = x.size(0);
    int C_in = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    int C_out = w.size(0);
    int KH = w.size(2);
    int KW = w.size(3);



    int out_h = (H + 2 * pad - KH) / stride + 1;
    int out_w = (W + 2 * pad - KW) / stride + 1;

    auto out = torch::zeros({N, C_out, out_h, out_w}, x.options());

    dim3 block(8, 8);
    dim3 grid((out_w + block.x - 1)/block.x,
              (out_h + block.y - 1)/block.y,
              N);

    size_t shmem_size = TILE_C * TILE_H * TILE_W * sizeof(float);

    gemm_gpu_o4_kernel<<<grid, block, shmem_size>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C_in, H, W,
        C_out, KH, KW,
        stride, pad,
        out_h, out_w);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_cuda", &conv_cuda, "Custom Conv2D (CUDA)");
}
