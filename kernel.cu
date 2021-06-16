#include "cuda_runtime.h"
#include "defns.h"
#include "gmfft.h"
#include <chrono>
#include <iostream>
#include <stdio.h>


int srwlUtiFFT(char *pcData, char typeData, double *arMesh, int nMesh,
               int dir) {
  int locErNo = 0;
  try {
    long nx = (long)arMesh[2];
    long ny = 1;
    if (nMesh >= 6)
      ny = (long)arMesh[5];

    int dimFFT = 1;
    if (ny > 1)
      dimFFT = 2;
    // float *pfData = (float*)pcData; //OC31012019 (commented-out)

    if (dimFFT == 1) {
      CGenMathFFT1DInfo FFT1DInfo;

      if (typeData == 'f') {
        FFT1DInfo.pInData = (float *)pcData;
        FFT1DInfo.pOutData = FFT1DInfo.pInData;
      }
#ifdef _FFTW3 // OC31012019
      else if (typeData == 'd') {
        FFT1DInfo.pdInData = (double *)pcData;
        FFT1DInfo.pdOutData = FFT1DInfo.pdInData;
      }
#endif

      FFT1DInfo.Dir = (char)dir;
      FFT1DInfo.xStart = arMesh[0];
      FFT1DInfo.xStep = arMesh[1];
      FFT1DInfo.Nx = nx;
      FFT1DInfo.HowMany = 1;
      FFT1DInfo.UseGivenStartTrValue = 0;

      CGenMathFFT1D FFT1D;
      if (locErNo = FFT1D.Make1DFFT(FFT1DInfo))
        return locErNo;

      arMesh[0] = FFT1DInfo.xStartTr;
      arMesh[1] = FFT1DInfo.xStepTr;
    } else {
      CGenMathFFT2DInfo FFT2DInfo;
      // FFT2DInfo.pData = pfData;
      if (typeData == 'f') // OC31012019
      {
        FFT2DInfo.pData = (float *)pcData;
      }
#ifdef _FFTW3 // OC31012019
      else if (typeData == 'd') {
        FFT2DInfo.pdData = (double *)pcData;
      }
#endif

      FFT2DInfo.Dir = (char)dir;
      FFT2DInfo.xStart = arMesh[0];
      FFT2DInfo.xStep = arMesh[1];
      FFT2DInfo.Nx = nx;
      FFT2DInfo.yStart = arMesh[3];
      FFT2DInfo.yStep = arMesh[4];
      FFT2DInfo.Ny = ny;
      FFT2DInfo.UseGivenStartTrValues = 0;

      CGenMathFFT2D FFT2D;
      if (locErNo = FFT2D.Make2DFFT(FFT2DInfo))
        return locErNo;

      arMesh[0] = FFT2DInfo.xStartTr;
      arMesh[1] = FFT2DInfo.xStepTr;
      arMesh[3] = FFT2DInfo.yStartTr;
      arMesh[4] = FFT2DInfo.yStepTr;
    }
  } catch (int erNo) {
    return erNo;
  }
  return 0;
}

int main() {
  double xStart = -5;
  double xRange = 10;
  long xNp = 100000000;
  double xStep = xRange / (xNp - 1);
  double mesh[3] = {xStart, xStep, xNp};
  float *input_data;
  float *input_data_cpy;
  int runs = 50;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaMallocManaged(&input_data, 2 * xNp * sizeof(float));
  cudaMallocManaged(&input_data_cpy, 2 * xNp * sizeof(float));
  memset(input_data, 0, sizeof(float) * 2 * xNp);

  int x0 = (int)((-0.5 - xStart) / xStep);
  int x1 = (int)((0.5 - xStart) / xStep);
  for (unsigned int i = x0; i < x1; i++) {
    input_data[2 * i] = 1;
  }

  double net_time = 0;
  double net_time_gpu = 0;

  cudaMemcpy(input_data_cpy, input_data, 2 * xNp * sizeof(float),
             cudaMemcpyDefault);
  srwlUtiFFT(reinterpret_cast<char *>(input_data), 'f', mesh, 3,
             1); // warm up run

  std::cout << "Starting benchmark " << std::endl;

  for (int i = 0; i < runs; i++) {
    cudaMemcpy(input_data, input_data_cpy, 2 * xNp * sizeof(float),
               cudaMemcpyDefault);
    
    cudaEventRecord(start);

    std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();

    srwlUtiFFT(reinterpret_cast<char *>(input_data), 'f', mesh, 3, 1);
    
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> ts = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    //Measure gpu time
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    net_time_gpu += ms;

    net_time += ts.count();
  }

  net_time_gpu /= runs;
  net_time /= runs;
  net_time *= 1000;

  std::cout << "Benchmark took " << net_time << " ms as measured from CPU." << std::endl;
  std::cout << "Benchmark took " << net_time_gpu << " ms as measured from GPU." << std::endl;

  // cudaDeviceReset must be called before exiting in order for profiling and
  // tracing tools such as Nsight and Visual Profiler to show complete traces.
  cudaError_t cudaStatus = cudaDeviceReset();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaDeviceReset failed!");
    return 1;
  }

  return 0;
}
