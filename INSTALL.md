We are still working on testing HPSim on different python environment. The python
versions we have tested include 3.7 to 3.9 on Linux.

**  **

To install hpsim as a package, please do the following:

1. **Install `CUDA` library**: `HPSim` is a GPU-based simulation code 
   that runs on NVIDIA GPUs. On linux, one can install the [CUDA toolkit](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux) 
   direcly or one can install via
   ```sudo apt install nvidia-cuda-toolkit```
   To verify the installation, try either `nvidia-smi` to see the GPUs or
   check the `nvcc --version` for the compiler version. Currently, HPSim is 
   developed on nvcc version 11.8.
2. **Install**: After cloning HPSim from a git server. Please do
   ```python -m pip install -v -e .```
   This will install `hpsim` module as an external link.



To uninstall, do `python -m pip uninstall hpsim`.
To clean up, do `python setup.py clean`.