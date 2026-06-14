#include "ops.cuh"
#include "tensor.cuh"
#include <gtest/gtest.h>

TEST (TensorTest, Constructors)
{
  std::vector<size_t> shape = { 2, 3 };
  Tensor<float> tensor1 (shape);
  std::cout << "0 tensor: " << std::endl << tensor1 << std::endl;

  Tensor<float> tensor2 ({ 1, 2, 3, 4, 5, 6 }, shape);
  std::cout << "1-6 tensor: " << std::endl << tensor2 << std::endl;
}

TEST (TensorTest, Ostream)
{
  Tensor<float> tensor1 ({ 2, 3 });
  std::cout << tensor1 << std::endl;

  Tensor<float> tensor2 ({ 4, 5 });
  std::cout << tensor2 << std::endl;
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

TEST (TensorTest, ElementWiseOps)
{
  std::vector<size_t> shape = { 2, 3 };

  Tensor<float> tensor1 ({ 1, 2, 3, 4, 5, 6 }, shape);
  Tensor<float> tensor2 ({ -1, 2, -3, 6, 5, 3 }, shape);

  Tensor<float> sum ({ 0, 4, 0, 10, 10, 9 }, shape);
  Tensor<float> diff ({ 2, 0, 6, -2, 0, 3 }, shape);
  Tensor<float> negative_diff = diff * -1.0f;
  Tensor<float> prod ({ -1, 4, -9, 24, 25, 18 }, shape);
  Tensor<float> half ({ 1, 0, 3, -1, 0, 1.5 }, shape);

  ASSERT_EQ (sum, tensor1 + tensor2);
  ASSERT_EQ (sum, tensor2 + tensor1);
  ASSERT_EQ (diff, tensor1 - tensor2);
  ASSERT_EQ (negative_diff, tensor2 - tensor1);
  ASSERT_EQ (prod, tensor1 * tensor2);
  ASSERT_EQ (prod, tensor2 * tensor1);
  ASSERT_EQ (half, diff / 2.0f);
  ASSERT_EQ (half, diff * 0.5f);
}

TEST (TensorTest, MatMul)
{
  Tensor<float> left ({ 1, 2, 3, 4, 5, 6 }, { 2, 3 });

  Tensor<float> right ({ 7, 8, 9, 10, 11, 12 }, { 3, 2 });

  Tensor<float> expected ({ 58, 64, 139, 154 }, { 2, 2 });

  ASSERT_EQ (expected, mat_mul (left, right));
}
