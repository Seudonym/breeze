{
  pkgs,
  cudaPackages,
}:
pkgs.mkShell {
  packages = with pkgs; [
    clang-tools
    neocmakelsp

    cmake
    gcc
    gdb
    gtest
    stdenv.cc.cc.lib

    cudaPackages.cuda_nvcc
    cudaPackages.cuda_gdb
    cudaPackages.cuda_cudart
  ];

  shellHook = ''
    export CUDA_HOME=${cudaPackages.cuda_cudart}
    export CUDA_PATH=${cudaPackages.cuda_cudart}
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/run/opengl-driver/lib:/run/opengl-driver-32/lib
  '';
}
