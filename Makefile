all:build

build: fft_kernels.o kernel.o
	nvcc -O3 -I /opt/nvidia/hpc_sdk/Linux_x86_64/20.11/cuda/include -I /opt/nvidia/hpc_sdk/Linux_x86_64/20.11/math_libs/include -o program -L/usr/local/cuda/lib64 -lcuda -lcudart -lcufft -lcufftw gmfft.cpp fft_kernels.o kernel.o

fft_kernels.o:
	nvcc -O3 -std=c++11 -I /opt/nvidia/hpc_sdk/Linux_x86_64/20.11/cuda/include -I /opt/nvidia/hpc_sdk/Linux_x86_64/20.11/math_libs/include -c fft_kernels.cu -o fft_kernels.o

kernel.o:
	nvcc -O3 -std=c++11 -I /opt/nvidia/hpc_sdk/Linux_x86_64/20.11/cuda/include -I /opt/nvidia/hpc_sdk/Linux_x86_64/20.11/math_libs/include -c kernel.cu -o kernel.o

clean:
	rm *.o
	rm program
