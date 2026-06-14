#pragma once
#include "ops_kernels.cuh"
#include "tensor.cuh"
#include <ostream>

// operator overloads with tensors
template <typename T>
Tensor<T>
operator+ (const Tensor<T> &left, const Tensor<T> &right)
{
  if (left.shape () != right.shape ())
    throw std::invalid_argument ("tensor shapes must match for addition");

  const auto shape = left.shape ();
  const size_t n = left.numel ();

  const size_t block_size = 256;
  const size_t grid_size = (n - 1 + block_size) / block_size;

  Tensor<T> result (left.shape ());

  add_kernel<<<grid_size, block_size>>> (left.data (), right.data (),
                                         result.data (), n);

  return result;
}

template <typename T>
Tensor<T>
operator- (const Tensor<T> &left, const Tensor<T> &right)
{
  if (left.shape () != right.shape ())
    throw std::invalid_argument ("tensor shapes must match for addition");

  const auto shape = left.shape ();
  const size_t n = left.numel ();

  const size_t block_size = 256;
  const size_t grid_size = (n - 1 + block_size) / block_size;

  Tensor<T> result (left.shape ());

  subtract_kernel<<<grid_size, block_size>>> (left.data (), right.data (),
                                              result.data (), n);

  return result;
}

template <typename T>
Tensor<T>
operator* (const Tensor<T> &left, const Tensor<T> &right)
{
  if (left.shape () != right.shape ())
    throw std::invalid_argument ("tensor shapes must match for addition");

  const auto shape = left.shape ();
  const size_t n = left.numel ();

  const size_t block_size = 256;
  const size_t grid_size = (n - 1 + block_size) / block_size;

  Tensor<T> result (left.shape ());

  dot_kernel<<<grid_size, block_size>>> (left.data (), right.data (),
                                         result.data (), n);

  return result;
}

template <typename T>
bool
operator== (const Tensor<T> &left, const Tensor<T> &right)
{
  if (left.shape () != right.shape ())
    return false;

  const auto shape = left.shape ();
  const size_t n = left.numel ();

  const size_t block_size = 256;
  const size_t grid_size = (n - 1 + block_size) / block_size;

  bool *equal;
  cudaMallocManaged (&equal, sizeof (bool));
  *equal = true;

  equality_kernel<<<grid_size, block_size>>> (left.data (), right.data (),
                                              equal, n);
  cudaDeviceSynchronize ();

  bool result = *equal;
  cudaFree (equal);

  return result;
}

// operator overloads with scalars
template <typename T>
Tensor<T>
operator* (const Tensor<T> &left, T right)
{
  const auto shape = left.shape ();
  const size_t n = left.numel ();

  const size_t block_size = 256;
  const size_t grid_size = (n - 1 + block_size) / block_size;

  Tensor<T> result (left.shape ());

  multiply_with_scalar_kernel<<<grid_size, block_size>>> (left.data (), right,
                                                          result.data (), n);

  return result;
}

template <typename T>
Tensor<T>
operator* (T left, const Tensor<T> &right)
{
  return right * left;
}

template <typename T>
Tensor<T>
operator/ (const Tensor<T> &left, T right)
{
  const auto shape = left.shape ();
  const size_t n = left.numel ();

  const size_t block_size = 256;
  const size_t grid_size = (n - 1 + block_size) / block_size;

  Tensor<T> result (left.shape ());

  divide_by_scalar_kernel<<<grid_size, block_size>>> (left.data (), right,
                                                      result.data (), n);

  return result;
}

template <typename T>
std::ostream &
operator<< (std::ostream &os, const Tensor<T> &tensor)
{
  size_t total_size = tensor.numel ();
  size_t preview_size = std::min (total_size, size_t{ 16 });

  // copy tensor device storage to host storage
  std::vector<T> host (preview_size);
  cudaError_t err
      = cudaMemcpy (host.data (), tensor.data (), preview_size * sizeof (T),
                    cudaMemcpyDeviceToHost);

  // print shape
  os << "Tensor(shape=[";
  for (size_t i = 0; i < tensor.shape ().size (); ++i)
    {
      if (i > 0)
        os << ", ";
      os << tensor.shape ()[i];
    }

  // print strides
  os << "], strides=[";
  for (size_t i = 0; i < tensor.strides ().size (); ++i)
    {
      if (i > 0)
        os << ", ";
      os << tensor.strides ()[i];
    }

  // print values
  os << "], values=[";
  if (err == cudaSuccess)
    {
      for (size_t i = 0; i < preview_size; ++i)
        {
          if (i > 0)
            os << ", ";
          os << host[i];
        }

      if (preview_size < total_size)
        os << ", ...";
    }
  else
    {
      os << "<cuda error: " << cudaGetErrorString (err) << ">";
    }

  os << "])";
  return os;
}

template <typename T>
Tensor<T>
mat_mul (const Tensor<T> &left, const Tensor<T> &right)
{
  const auto left_shape = left.shape ();
  const auto right_shape = right.shape ();
  if (left_shape[1] != right_shape[0])
    throw std::invalid_argument ("tensor shapes not appropriate for mat mul");

  const size_t m = left_shape[0];
  const size_t k = left_shape[1];
  const size_t n = right_shape[1];

  const dim3 block_size (16, 16);
  const dim3 grid_size ((n - 1 + block_size.x) / block_size.x,
                        (m - 1 + block_size.y) / block_size.y);

  Tensor<T> result ({ m, n });

  mat_mul_contiguous<<<grid_size, block_size>>> (left.data (), right.data (),
                                                 result.data (), m, k, n);

  return result;
}
