{
  pkgs,
  cudaPackages,
}: {
  default = pkgs.mkShell {
    packages = with pkgs; [
      clang-tools
      neocmakelsp

      cmake
      gcc
      gdb
      gtest

      cudaPackages.cudatoolkit
    ];

    shellHook = ''
      export CUDA_HOME=${cudaPackages.cudatoolkit}
      export CUDA_PATH=${cudaPackages.cudatoolkit}
      export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/run/opengl-driver/lib:/run/opengl-driver-32/lib
    '';
  };
}
