all:build

build: fft_kernels.o kernel.o
	g++ -o program -L/usr/local/cuda/lib64 -lcuda -lcudart -lcufft gmfft.cpp fft_kernels.o kernel.o

fft_kernels.o:
	nvcc -c fft_kernels.cu

kernel.o:
	nvcc -c kernel.cu
