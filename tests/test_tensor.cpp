#include "ops.h"
#include "tensor.h"
#include <gtest/gtest.h>

TEST (TensorTest, DefaultConstructor)
{
  std::vector<size_t> shape = { 2, 3 };
  Tensor<float> tensor (shape);
  std::cout << "0 tensor: " << std::endl << tensor << std::endl;
}

TEST (TensorTest, DataConstructor)
{
  std::vector<size_t> shape = { 2, 3 };
  Tensor<float> tensor ({ 1, 2, 3, 4, 5, 6 }, shape);
  std::cout << "1-6 tensor: " << std::endl << tensor << std::endl;
}

TEST (TensorTest, Ostream)
{
  std::vector<size_t> shape = { 2, 3 };
  Tensor<float> tensor (shape);
  std::cout << tensor << std::endl;
}

TEST (TensorTest, OstreamOverflow)
{
  std::vector<size_t> shape = { 5, 5 };
  Tensor<float> tensor (shape);
  std::cout << tensor << std::endl;
}

TEST (TensorTest, Equality)
{
  std::vector<size_t> shape = { 2, 3 };

  Tensor<float> tensor1 ({ 1, 2, 3, 4, 5, 6 }, shape);
  Tensor<float> tensor2 ({ -1, 2, -3, 6, 5, 3 }, shape);
  Tensor<float> tensor3 ({ 0 }, { 1 });

  ASSERT_EQ (tensor1, tensor1);
  ASSERT_NE (tensor1, tensor2);
  ASSERT_NE (tensor1, tensor3);
}
TEST (TensorTest, Addition)
{
  std::vector<size_t> shape = { 2, 3 };

  Tensor<float> tensor1 ({ 1, 2, 3, 4, 5, 6 }, shape);
  Tensor<float> tensor2 ({ -1, 2, -3, 6, 5, 3 }, shape);
  Tensor<float> sum ({ 0, 4, 0, 10, 10, 9 }, shape);

  ASSERT_EQ (sum, tensor1 + tensor2);
}
