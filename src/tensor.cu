#include "tensor.h"
#include <cuda_runtime.h>
#include <functional>
#include <numeric>
#include <vector>

template <typename T>
Tensor<T>::Tensor (std::vector<size_t> size) : size_ (size)
{
  size_t total_size = std::accumulate (size.begin (), size.end (), 1,
                                       std::multiplies<size_t> ());
  cudaMalloc (&data_, total_size * sizeof (T));
}

template <typename T> Tensor<T>::~Tensor () { cudaFree (data_); }

template class Tensor<float>;
