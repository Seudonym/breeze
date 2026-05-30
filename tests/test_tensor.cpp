#include "tensor.h"
#include <gtest/gtest.h>

TEST (TensorTest, ConstructorDestructor)
{
  std::vector<size_t> size = { 2, 3 };
  Tensor<float> tensor (size);
}

TEST (TensorTest, Ostream)
{
  std::vector<size_t> size = { 2, 3 };
  Tensor<float> tensor (size);
  std::cout << tensor << std::endl;
}
