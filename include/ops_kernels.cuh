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

  out_sum[col + row * n] = sum;
}
