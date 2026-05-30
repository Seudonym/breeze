#include "tensor.h"
#include <gtest/gtest.h>

TEST (TensorTest, ConstructorDestructor)
{
  std::vector<size_t> size = { 2, 3 };
  Tensor<float> tensor (size);
  std::cout << "0 tensor: " << std::endl << tensor << std::endl;

  Tensor<float> tensor_new ({ 1, 2, 3, 4, 5, 6 }, size);
  std::cout << "1-6 tensor: " << std::endl << tensor_new << std::endl;
}

TEST (TensorTest, Ostream)
{
  std::vector<size_t> size = { 2, 3 };
  Tensor<float> tensor (size);
  std::cout << tensor << std::endl;
}
