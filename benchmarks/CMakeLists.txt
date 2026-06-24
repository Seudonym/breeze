add_executable(bench_matmul
  bench_matmul.cu
)

target_link_libraries(bench_matmul
  PRIVATE
    breeze
    benchmark::benchmark
)
