{
  pkgs,
  cudaPackages,
}: {
  default = pkgs.stdenv.mkDerivation {
    pname = "breeze";
    version = "0.1.0";
    src = ./.;

    nativeBuildInputs = [
      pkgs.cmake
      cudaPackages.cuda_nvcc
    ];

    buildInputs = [
      cudaPackages.cudatoolkit
      pkgs.gtest
      pkgs.gbenchmark
    ];

    cmakeFlags = [
      "-DCMAKE_CUDA_ARCHITECTURES=native"
    ];

    strictDeps = true;
    __structuredAttrs = true;

    meta = with pkgs.lib; {
      description = "A lightweight CUDA tensor library";
      license = licenses.mit;
      platforms = platforms.linux;
    };
  };
}
