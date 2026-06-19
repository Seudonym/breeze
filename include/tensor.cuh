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
    if (ptr)
      cudaFree (ptr);
  }
};

template <typename T> class Tensor
{
public:
  // constructors

  // shape only constructor
  Tensor (std::vector<size_t> shape) : shape_ (std::move (shape))
  {
    recalc_strides_ ();

    // allocate memory and wrap in shared_ptr
    size_t total_size = this->numel ();
    T *raw_ptr;
    cudaMalloc (&raw_ptr, total_size * sizeof (T));
    this->storage_ = std::shared_ptr<T> (raw_ptr, CudaMallocDeleter ());
  }

  Tensor (std::vector<T> data, std::vector<size_t> shape)
      : shape_ (std::move (shape))
  {
    // check if data and shape match sizes appropriately before
    // recalc_strides_ is called
    size_t total_size = this->numel ();
    if (total_size != data.size ())
      throw std::invalid_argument ("shape mismatch");
    recalc_strides_ ();

    // allocate memory and wrap in shared_ptr
    const size_t size_in_bytes = total_size * sizeof (T);
    T *raw_ptr;
    cudaMalloc (&raw_ptr, size_in_bytes);
    this->storage_ = std::shared_ptr<T> (raw_ptr, CudaMallocDeleter ());
    cudaMemcpy (raw_ptr, data.data (), size_in_bytes, cudaMemcpyHostToDevice);
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
    return std::accumulate (this->shape_.begin (), this->shape_.end (),
                            size_t{ 1 }, std::multiplies<size_t> ());
  }

  // transform
  Tensor
  reshape (std::vector<size_t> new_shape) const
  {
    // check if new shape is valid
    size_t new_numel
        = std::accumulate (new_shape.begin (), new_shape.end (), size_t{ 1 },
                           std::multiplies<size_t> ());
    if (this->numel () != new_numel)
      throw std::invalid_argument ("shape mismatch");

    // return new tensor with different metadata
    return Tensor (private_tag{}, this->storage_, new_shape);
  }

private:
  std::shared_ptr<T> storage_;
  std::vector<size_t> shape_;
  std::vector<size_t> strides_;

  struct private_tag
  {
  };

  Tensor (private_tag, std::shared_ptr<T> storage, std::vector<size_t> shape)
      : storage_ (storage), shape_ (std::move (shape))
  {
    recalc_strides_ ();
  }

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
