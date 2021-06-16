# SRW FFT Example
This is a simple program demonstrating porting of components of SRW to CUDA. The FFT implementation (via FFTW3) is taken from SRW, and modified to use cufft, the additional processing done on the FFT output has also been ported to CUDA.

## Compiling and Running
Compilation requires nvcc and cufft to be installed and available on the PATH variable.
To compile without plan reuse:
    
    make

To compile with plan reuse:

    make CFLAGS=-DPLAN_REUSE

On Windows this will require GNU/Make to be installed, which for [Chocolatey](https://chocolatey.org/) users is available via:

    choco install make

To run (on Linux):

    ./program

To run (on Windows):

    .\program.exe

## Example Output
    $ .\program.exe
    Starting benchmark
    Benchmark took 0.01993 ms as measured from CPU.
    Benchmark took 0.0213312 ms as measured from GPU.

## Code Structure
* The program entry point and benchmark setup is in ```kernel.cu```
* The FFT wrapper is in ```gmfft.cpp```
* CUDA ports of various portions of FFT pre/post-processing are in ```fft_kernels.cu```