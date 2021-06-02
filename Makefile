all:build

build: fft_kernels.o kernel.o
	g++ -I /opt/nvidia/hpc_sdk/Linux_x86_64/20.11/cuda/include -I /opt/nvidia/hpc_sdk/Linux_x86_64/20.11/math_libs/include -o program -L/usr/local/cuda/lib64 -lcuda -lcudart -lcufft -lcufftw gmfft.cpp fft_kernels.o kernel.o

fft_kernels.o:
	nvcc -I /opt/nvidia/hpc_sdk/Linux_x86_64/20.11/cuda/include -I /opt/nvidia/hpc_sdk/Linux_x86_64/20.11/math_libs/include -c fft_kernels.cu

kernel.o:
	nvcc -I /opt/nvidia/hpc_sdk/Linux_x86_64/20.11/cuda/include -I /opt/nvidia/hpc_sdk/Linux_x86_64/20.11/math_libs/include -c kernel.cu

clean:
	rm *.o
	rm program
