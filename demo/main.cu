#include "ops.cuh"
#include "tensor.cuh"
#include <iostream>
#include <vector>

int
main ()
{
  std::vector<size_t> shape = { 2, 3 };
  std::vector<float> data = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
  Tensor<float> a (data, shape);

  std::cout << "Tensor A:" << std::endl;
  std::cout << a << std::endl;

  Tensor<float> b = a.reshape ({ 3, 2 });
  std::cout << "\nTensor B:" << std::endl;
  std::cout << b << std::endl;

  Tensor<float> c = mat_mul (a, b);
  std::cout << "\nTensor C (A @ B):" << std::endl;
  std::cout << c << std::endl;

  return 0;
}
