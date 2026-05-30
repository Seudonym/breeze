#pragma once
#include <cstddef>
#include <vector>

template <typename T> class Tensor
{
public:
  Tensor (std::vector<size_t> shape);
  ~Tensor ();

  Tensor (const Tensor &) = delete;
  Tensor &operator= (const Tensor &) = delete;

  Tensor (Tensor &&other) noexcept;
  Tensor &operator= (Tensor &&other) noexcept;

  const std::vector<size_t> &shape () const;

private:
  T *data_;
  std::vector<size_t> shape_;
};
