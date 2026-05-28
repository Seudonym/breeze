#include <iostream>
#include <tensor.h>

int
main (int argc, char *argv[])
{
  Tensor<float> x ({ 10 });
  Tensor<float> y ({ 10, 2, 3 });
  std::cout << "didnt segfault" << std::endl;
  return 0;
}
