#pragma once
#include <cstddef>
#include <cuda_runtime.h>
#include <memory>
#include <ostream>
#include <vector>

template <typename T> class Tensor
{
public:
  Tensor (std::vector<size_t> shape);
  Tensor (std::vector<T> data, std::vector<size_t> shape);

  const std::vector<size_t> &shape () const;
  size_t size () const;

  T *data ();
  const T *data () const;

private:
  std::shared_ptr<T> data_;
  std::vector<size_t> shape_;
};
