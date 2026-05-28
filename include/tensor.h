#pragma once
#include <cstddef>
#include <vector>

template <typename T> class Tensor
{
public:
  Tensor (std::vector<size_t> size);
  ~Tensor ();

private:
  T *data_;
  std::vector<size_t> size_;
};
