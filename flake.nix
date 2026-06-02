{
  description = "Breeze flake";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };
  outputs = {nixpkgs, ...}: let
    system = "x86_64-linux";
    pkgs = import nixpkgs {
      inherit system;
      config = {
        allowUnfree = true;
        cudaSupport = true;
      };
    };
    cudaPackages = pkgs.cudaPackages;
  in {
    packages.${system} = import ./package.nix {inherit pkgs cudaPackages;};
    devShells.${system} = import ./shell.nix {inherit pkgs cudaPackages;};
  };
}
