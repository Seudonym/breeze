#pragma once
#include <cstddef>
#include <cuda_runtime.h>
#include <functional>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

// custom deleter for shared ptr
struct CudaMallocDeleter
{
  void
  operator() (void *ptr) const
  {
    cudaFree (ptr);
  }
};

template <typename T> class Tensor
{
public:
  // constructors
  Tensor (std::vector<size_t> shape) : shape_ (std::move (shape))
  {
    recalc_strides_ ();

    size_t total_size = this->numel ();
    T *raw_ptr;
    cudaMalloc (&raw_ptr, total_size * sizeof (T));
    this->storage_ = std::shared_ptr<T> (raw_ptr, CudaMallocDeleter ());
  }

  Tensor (std::vector<T> data, std::vector<size_t> shape)
      : shape_ (std::move (shape))
  {
    recalc_strides_ ();

    size_t total_size = this->numel ();
    if (total_size != data.size ())
      throw std::invalid_argument ("shape mismatch");

    const size_t size_in_bytes = total_size * sizeof (T);

    T *raw_ptr;
    cudaMalloc (&raw_ptr, size_in_bytes);
    cudaMemcpy (raw_ptr, data.data (), size_in_bytes, cudaMemcpyHostToDevice);

    this->storage_ = std::shared_ptr<T> (raw_ptr, CudaMallocDeleter ());
  }

  // getters
  const std::vector<size_t> &
  shape () const
  {
    return this->shape_;
  }

  const std::vector<size_t> &
  strides () const
  {
    return this->strides_;
  }

  T *
  data ()
  {
    return this->storage_.get ();
  }

  const T *
  data () const
  {
    return this->storage_.get ();
  }

  // info funcs
  size_t
  numel () const
  {
    return std::accumulate (this->shape_.begin (), this->shape_.end (), 1,
                            std::multiplies<size_t> ());
  }

  // transform
  Tensor
  reshape (std::vector<size_t> new_shape) const
  {
    size_t new_numel = std::accumulate (new_shape.begin (), new_shape.end (),
                                        1, std::multiplies<size_t> ());

    if (this->numel () != new_numel)
      {
        throw std::invalid_argument ("shape mismatch");
      }

    return Tensor (this->storage_, std::move (new_shape));
  }

private:
  std::shared_ptr<T> storage_;
  std::vector<size_t> shape_;
  std::vector<size_t> strides_;

  void
  recalc_strides_ ()
  {
    strides_.resize (shape_.size ());
    size_t stride = 1;
    for (size_t i = shape_.size (); i-- > 0;)
      {
        strides_[i] = stride;
        stride *= shape_[i];
      }
  }
};
