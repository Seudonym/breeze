#pragma once
#include <cstddef>
#include <cuda_runtime.h>
#include <memory>
#include <ostream>
#include <vector>

template <typename T> class Tensor
{
public:
  // constructors
  Tensor (std::vector<size_t> shape);
  Tensor (std::vector<T> data, std::vector<size_t> shape);

  // getters
  const std::vector<size_t> &shape () const;
  const std::vector<size_t> &strides () const;
  T *data ();
  const T *data () const;

  // info funcs
  size_t numel () const;

private:
  std::shared_ptr<T> storage_;
  std::vector<size_t> shape_;
  std::vector<size_t> strides_;
};
