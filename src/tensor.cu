#include "tensor.h"
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <functional>
#include <numeric>
#include <vector>

template <typename T>
Tensor<T>::Tensor (std::vector<size_t> shape) : shape_ (shape)
{
  size_t total_size = std::accumulate (shape.begin (), shape.end (), 1,
                                       std::multiplies<size_t> ());
  cudaMalloc (&data_, total_size * sizeof (T));
}
template <typename T> Tensor<T>::~Tensor () { cudaFree (data_); }

template <typename T>
Tensor<T>::Tensor (Tensor &&other) noexcept
    : data_ (other.data_), shape_ (other.shape_)
{
  other.data_ = nullptr;
}

template <typename T>
Tensor<T> &
Tensor<T>::operator= (Tensor<T> &&other) noexcept
{
  if (this != &other)
    {
      cudaFree (data_);
      data_ = other.data_;
      shape_ = std::move (other.shape_);
      other.data_ = nullptr;
    }
  return *this;
}

template <typename T>
const std::vector<size_t> &
Tensor<T>::shape () const
{
  return this->shape_;
}

template class Tensor<float>;
