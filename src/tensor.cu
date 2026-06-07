#include "tensor.h"
#include <cuda_runtime.h>
#include <functional>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

struct CudaMallocDeleter
{
  void
  operator() (void *ptr) const
  {
    cudaFree (ptr);
  }
};

template <typename T>
Tensor<T>::Tensor (std::vector<size_t> shape) : shape_ (std::move (shape))
{
  strides_.resize (shape_.size ());
  size_t stride = 1;
  for (size_t i = shape_.size (); i-- > 0;)
    {
      strides_[i] = stride;
      stride *= shape_[i];
    }

  size_t total_size = std::accumulate (shape_.begin (), shape_.end (), 1,
                                       std::multiplies<size_t> ());
  T *raw_ptr;
  cudaMalloc (&raw_ptr, total_size * sizeof (T));
  data_ = std::shared_ptr<T> (raw_ptr, CudaMallocDeleter ());
}

template <typename T>
Tensor<T>::Tensor (std::vector<T> data, std::vector<size_t> shape)
    : shape_ (std::move (shape))
{
  strides_.resize (shape_.size ());
  size_t stride = 1;
  for (size_t i = shape_.size (); i-- > 0;)
    {
      strides_[i] = stride;
      stride *= shape_[i];
    }

  size_t total_size = std::accumulate (shape_.begin (), shape_.end (), 1,
                                       std::multiplies<size_t> ());
  if (total_size != data.size ())
    throw std::invalid_argument ("size mismatch");

  const size_t size_in_bytes = total_size * sizeof (T);

  T *raw_ptr;
  cudaMalloc (&raw_ptr, size_in_bytes);
  cudaMemcpy (raw_ptr, data.data (), size_in_bytes, cudaMemcpyHostToDevice);

  data_ = std::shared_ptr<T> (raw_ptr, CudaMallocDeleter ());
}

template <typename T>
const std::vector<size_t> &
Tensor<T>::shape () const
{
  return this->shape_;
}

template <typename T>
const std::vector<size_t> &
Tensor<T>::strides () const
{
  return this->strides_;
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
  return this->data_.get ();
}

template <typename T>
const T *
Tensor<T>::data () const
{
  return this->data_.get ();
}

template class Tensor<float>;
