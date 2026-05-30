#include "tensor.h"
#include <algorithm>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <functional>
#include <numeric>
#include <ostream>
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

template <typename T>
size_t
Tensor<T>::size () const
{
  return std::accumulate (this->shape_.begin (), this->shape_.end (), 1,
                          std::multiplies<size_t> ());
}

template <typename T>
T *
Tensor<T>::data ()
{
  return this->data_;
}

template <typename T>
const T *
Tensor<T>::data () const
{
  return this->data_;
}

template class Tensor<float>;
