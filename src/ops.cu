#include "ops.h"
#include <stdexcept>

template <typename T>
__global__ void
add_kernel (const T *left, const T *right, T *result, size_t n)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  result[i] = left[i] + right[i];
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

template Tensor<float> operator+ <float> (const Tensor<float> &left,
                                          const Tensor<float> &right);
