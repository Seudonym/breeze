#include "ops.h"
#include <cuda_runtime.h>
#include <stdexcept>

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
Tensor<T>
operator+ (const Tensor<T> &left, const Tensor<T> &right)
{
  if (left.shape () != right.shape ())
    throw std::invalid_argument ("tensor shapes must match for addition");

  const auto shape = left.shape ();
  const size_t n = left.size ();

  const size_t block_size = 256;
  const size_t grid_size = (n - 1 + block_size) / block_size;

  Tensor<T> result (left.shape ());

  add_kernel<<<grid_size, block_size>>> (left.data (), right.data (),
                                         result.data (), n);

  return result;
}

template <typename T>
Tensor<T>
operator- (const Tensor<T> &left, const Tensor<T> &right)
{
  if (left.shape () != right.shape ())
    throw std::invalid_argument ("tensor shapes must match for addition");

  const auto shape = left.shape ();
  const size_t n = left.size ();

  const size_t block_size = 256;
  const size_t grid_size = (n - 1 + block_size) / block_size;

  Tensor<T> result (left.shape ());

  subtract_kernel<<<grid_size, block_size>>> (left.data (), right.data (),
                                              result.data (), n);

  return result;
}

template <typename T>
Tensor<T>
operator* (const Tensor<T> &left, const Tensor<T> &right)
{
  if (left.shape () != right.shape ())
    throw std::invalid_argument ("tensor shapes must match for addition");

  const auto shape = left.shape ();
  const size_t n = left.size ();

  const size_t block_size = 256;
  const size_t grid_size = (n - 1 + block_size) / block_size;

  Tensor<T> result (left.shape ());

  dot_kernel<<<grid_size, block_size>>> (left.data (), right.data (),
                                         result.data (), n);

  return result;
}

template <typename T>
bool
operator== (const Tensor<T> &left, const Tensor<T> &right)
{
  if (left.shape () != right.shape ())
    return false;

  const auto shape = left.shape ();
  const size_t n = left.size ();

  const size_t block_size = 256;
  const size_t grid_size = (n - 1 + block_size) / block_size;

  bool *equal;
  cudaMallocManaged (&equal, sizeof (bool));
  *equal = true;

  equality_kernel<<<grid_size, block_size>>> (left.data (), right.data (),
                                              equal, n);
  cudaDeviceSynchronize ();

  bool result = *equal;
  cudaFree (equal);

  return result;
}

template Tensor<float> operator+ <float> (const Tensor<float> &left,
                                          const Tensor<float> &right);
template Tensor<float> operator- <float> (const Tensor<float> &left,
                                          const Tensor<float> &right);
template Tensor<float> operator* <float> (const Tensor<float> &left,
                                          const Tensor<float> &right);
template bool operator== <float> (const Tensor<float> &left,
                                  const Tensor<float> &right);
