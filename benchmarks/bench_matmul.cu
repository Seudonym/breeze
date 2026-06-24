#include "ops_kernels.cuh"
#include "tensor.cuh"
#include <benchmark/benchmark.h>
#include <random>
#include <vector>

std::vector<float>
generate_random_vector (size_t size)
{
  std::vector<float> vec (size);
  std::default_random_engine generator (1337);
  std::uniform_real_distribution<float> distribution (-1.0f, 1.0f);
  for (size_t i = 0; i < size; ++i)
    {
      vec[i] = distribution (generator);
    }
  return vec;
}

static void
BM_NaiveMatMul (benchmark::State &state)
{
  const size_t N = state.range (0);

  auto _left = generate_random_vector (N * N);
  auto _right = generate_random_vector (N * N);

  Tensor<float> left (_left, { N, N });
  Tensor<float> right (_right, { N, N });
  Tensor<float> out ({ N, N });

  const dim3 block_size (16, 16);
  const dim3 grid_size ((N - 1 + block_size.x) / block_size.x,
                        (N - 1 + block_size.y) / block_size.y);

  for (auto _ : state)
    {
      mat_mul_contiguous_kernel<<<grid_size, block_size>>> (
          left.data (), right.data (), out.data (), N, N, N);
      cudaDeviceSynchronize ();
    }

  // FLOPs count: 2 * N^3 operations per multiplication
  state.SetItemsProcessed (int64_t (state.iterations ()) * N * N * N * 2);
}

static void
BM_SharedMemMatMul (benchmark::State &state)
{
  const size_t N = state.range (0);

  auto _left = generate_random_vector (N * N);
  auto _right = generate_random_vector (N * N);

  Tensor<float> left (_left, { N, N });
  Tensor<float> right (_right, { N, N });
  Tensor<float> out ({ N, N });

  const dim3 block_size (16, 16);
  const dim3 grid_size ((N - 1 + block_size.x) / block_size.x,
                        (N - 1 + block_size.y) / block_size.y);

  for (auto _ : state)
    {
      mat_mul_contiguous_shared_mem_kernel<<<grid_size, block_size>>> (
          left.data (), right.data (), out.data (), N, N, N);
      cudaDeviceSynchronize ();
    }

  // FLOPs count: 2 * N^3 operations per multiplication
  state.SetItemsProcessed (int64_t (state.iterations ()) * N * N * N * 2);
}

BENCHMARK (BM_NaiveMatMul)
    ->RangeMultiplier (2)
    ->Range (256, 8192 * 2)
    ->Unit (benchmark::kMicrosecond);
BENCHMARK (BM_SharedMemMatMul)
    ->RangeMultiplier (2)
    ->Range (256, 8192 * 2)
    ->Unit (benchmark::kMicrosecond);

BENCHMARK_MAIN ();
