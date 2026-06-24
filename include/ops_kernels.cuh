#pragma once

// kernels
#include <cstddef>
template <typename T>
__global__ void
add_kernel (const T *left, const T *right, T *out_sum, size_t n)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  out_sum[i] = left[i] + right[i];
}

template <typename T>
__global__ void
subtract_kernel (const T *left, const T *right, T *out_sum, size_t n)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  out_sum[i] = left[i] - right[i];
}

template <typename T>
__global__ void
dot_kernel (const T *left, const T *right, T *out_sum, size_t n)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  out_sum[i] = left[i] * right[i];
}

template <typename T>
__global__ void
equality_kernel (const T *left, const T *right, bool *out_equal, size_t n)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  if (left[i] != right[i])
    {
      *out_equal = false;
    }
}

template <typename T>
__global__ void
multiply_with_scalar_kernel (const T *left, T right, T *out_sum, size_t n)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  out_sum[i] = left[i] * right;
}

template <typename T>
__global__ void
divide_by_scalar_kernel (const T *left, T right, T *out_sum, size_t n)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  out_sum[i] = left[i] / right;
}

// copy kernel
template <typename T>
__global__ void
copy_nd_kernel (T *dst, const T *src, const size_t *shape,
                const size_t *src_strides, size_t ndim, size_t numel)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= numel)
    return;

  size_t accumulate = 0;
  size_t remainder = i;
  for (int i = ndim - 1; i >= 0; i--)
    {
      size_t coord = remainder % shape[i];
      remainder /= shape[i];
      accumulate += coord * src_strides[i];
    }

  // index by stride into src and set dst[i] to that
  dst[i] = src[accumulate];
}

// mat mul from here
template <typename T>
__global__ void
mat_mul_contiguous_kernel (const T *left, const T *right, T *out, size_t m,
                           size_t k, size_t n)
{
  // left has a k, 1 stride
  // right has a n, 1 stride
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= m || col >= n)
    return;

  T sum = 0;
  for (size_t i = 0; i < k; i++)
    {
      sum += left[row * k + i] * right[i * n + col];
    }

  out[col + row * n] = sum;
}

template <typename T>
__global__ void
mat_mul_contiguous_shared_mem_kernel (const T *left, const T *right, T *out,
                                      size_t m, size_t k, size_t n)
{
  // allocate shared memory for the block
  const size_t TILE_SIZE = 16;
  __shared__ T tile_left[TILE_SIZE][TILE_SIZE];
  __shared__ T tile_right[TILE_SIZE][TILE_SIZE];

  // element idx for out
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;

  // split the k dimension up evently based on tile size
  size_t num_tiles = (k - 1 + TILE_SIZE) / TILE_SIZE;

  // iterate over each tile
  T sum = 0;
  for (size_t tile_idx = 0; tile_idx < num_tiles; tile_idx++)
    {
      // loal left tile, left matrix has stride (k, 1)
      if (row < m && (tile_idx * TILE_SIZE + threadIdx.x) < k)
        tile_left[threadIdx.y][threadIdx.x]
            = left[(tile_idx * TILE_SIZE + threadIdx.x) + row * k];
      else
        tile_left[threadIdx.y][threadIdx.x] = 0;

      // load right tile, right matrix has stride (n, 1)
      if (col < n && (tile_idx * TILE_SIZE + threadIdx.y) < k)
        tile_right[threadIdx.y][threadIdx.x]
            = right[col + (tile_idx * TILE_SIZE + threadIdx.y) * n];
      else
        tile_right[threadIdx.y][threadIdx.x] = 0;

      // wait for tiles to be fully copied
      __syncthreads ();

      // multiply and accumulate loaded tiles
      for (size_t i = 0; i < TILE_SIZE; i++)
        {
          sum += tile_left[threadIdx.y][i] * tile_right[i][threadIdx.x];
        }

      __syncthreads ();
    }

  if (row < m && col < n)
    out[col + row * n] = sum;
}
