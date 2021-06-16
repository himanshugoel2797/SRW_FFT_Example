#pragma once

#define _FFTW3 1
#define _CUFFT 1

void RepairSignAfter2DFFT_CUDA(float* pAfterFFT, long Nx, long Ny);
void RotateDataAfter2DFFT_CUDA(float* pAfterFFT, long Nx, long Ny);
void NormalizeDataAfter2DFFT_CUDA(float* pAfterFFT, long Nx, long Ny, double Mult);

void RepairSignAfter2DFFT_CUDA(double* pAfterFFT, long Nx, long Ny);
void RotateDataAfter2DFFT_CUDA(double* pAfterFFT, long Nx, long Ny);
void NormalizeDataAfter2DFFT_CUDA(double* pAfterFFT, long Nx, long Ny, double Mult);

void RepairSignAfter1DFFT_CUDA(float* pAfterFFT, long HowMany, long Nx);
void RotateDataAfter1DFFT_CUDA(float* pAfterFFT, long HowMany, long Nx);
void RepairAndRotateDataAfter1DFFT_CUDA(float* pAfterFFT, long HowMany, long Nx);
void NormalizeDataAfter1DFFT_CUDA(float* pAfterFFT, long HowMany, long Nx, double Mult);
void FillArrayShift_CUDA(double t0, double tStep, long Nx, float* tShiftX);
void TreatShift_CUDA(float* pData, long HowMany, long Nx, float* tShiftX);

void RepairSignAfter1DFFT_CUDA(double* pAfterFFT, long HowMany, long Nx);
void RotateDataAfter1DFFT_CUDA(double* pAfterFFT, long HowMany, long Nx);
void RepairAndRotateDataAfter1DFFT_CUDA(double* pAfterFFT, long HowMany, long Nx);
void NormalizeDataAfter1DFFT_CUDA(double* pAfterFFT, long HowMany, long Nx, double Mult);
void FillArrayShift_CUDA(double t0, double tStep, long Nx, double* tShiftX);
void TreatShift_CUDA(double* pData, long HowMany, long Nx, double* tShiftX);

#define MEMORY_ALLOCATION_FAILURE -1
#define ERROR_IN_FFT -2

struct CGenMathFFT1DInfo {
	float* pInData, * pOutData;
	double* pdInData, * pdOutData; //OC31012019

	char Dir; // >0: forward; <0: backward
	double xStep, xStart;
	double xStepTr, xStartTr;
	long Nx;
	//long long Nx;
	long HowMany;
	//long long HowMany;
	char UseGivenStartTrValue;
	double MultExtra;

	char TreatSharpEdges;
	double LeftSharpEdge, RightSharpEdge;
	char ApplyAutoShiftAfter;

	CGenMathFFT1DInfo()
	{
		HowMany = 1; UseGivenStartTrValue = 0;
		TreatSharpEdges = 0;
		MultExtra = 1.;
		ApplyAutoShiftAfter = 1;

		pInData = 0; //OC31012019
		pOutData = 0;
		pdInData = 0;
		pdOutData = 0;
	}
};

struct CGenMathFFT2DInfo {
	float* pData;
	double* pdData; //OC31012019

	char Dir; // >0: forward; <0: backward
	double xStep, yStep, xStart, yStart;
	double xStepTr, yStepTr, xStartTr, yStartTr;
	long Nx, Ny;
	//long long Nx, Ny;

	long howMany; //OC151014
	long iStride, iDist; //OC151014
	//From FFTW 2.1.5 Tutorial
	//iStride and iDist describe the input array(s). 
	//There are howMany multi-dimensional input arrays; the first one is pointed to by in (= pData), 
	//the second one is pointed to by in + iDist, and so on, up to in + (howMany - 1) * iDist. 
	//Each multi-dimensional input array consists of complex numbers (see Section Data Types), 
	//stored in row-major format (see Section Multi-dimensional Array Format), which are not necessarily contiguous in memory. 
	//Specifically, in[0] is the first element of the first array, in[istride] is the second element of the first array, and so on. 
	//In general, the i-th element of the j-th input array will be in position in[i * istride + j * idist]. 
	//Note that, here, i refers to an index into the row-major format for the multi-dimensional array, rather than an index in any particular dimension. 
	//In-place transforms:  For plans created with the FFTW_IN_PLACE option, the transform is computed in-place--the output is returned in the in array, 
	//using the same strides, etcetera, as were used in the input. 

	char UseGivenStartTrValues;
	double ExtraMult; //OC20112017

	CGenMathFFT2DInfo()
	{
		howMany = 1; iStride = 1; iDist = 0; //OC151014
		UseGivenStartTrValues = 0;
		ExtraMult = 1.; //OC20112017

		pData = 0; //OC31012019
		pdData = 0;
	}
};
