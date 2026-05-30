#pragma once
#include <cstddef>
#include <cuda_runtime.h>
#include <ostream>
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
  size_t size () const;

  T *data ();
  const T *data () const;

  friend std::ostream &
  operator<< (std::ostream &os, const Tensor<T> &tensor)
  {
    size_t total_size = tensor.size ();
    size_t preview_size = std::min (total_size, size_t{ 16 });

    std::vector<T> host (preview_size);
    cudaError_t err
        = cudaMemcpy (host.data (), tensor.data (), preview_size * sizeof (T),
                      cudaMemcpyDeviceToHost);

    os << "Tensor(shape=[";
    for (size_t i = 0; i < tensor.shape ().size (); ++i)
      {
        if (i > 0)
          os << ", ";
        os << tensor.shape ()[i];
      }

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

private:
  T *data_;
  std::vector<size_t> shape_;
};
