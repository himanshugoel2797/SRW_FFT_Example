#include "defns.h"
#include "gmfft.h"

int CGenMathFFT1D::Make1DFFT(CGenMathFFT1DInfo& FFT1DInfo)
{// Assumes Nx, Ny even !
	//Added by S.Yakubov (for profiling?) at parallelizing SRW via OpenMP:
	//double start;
	//get_walltime (&start);
	const double RelShiftTol = 1.E-06;

	SetupLimitsTr(FFT1DInfo);

	double xStepNx = FFT1DInfo.Nx * FFT1DInfo.xStep;
	double x0_After = FFT1DInfo.xStart + 0.5 * xStepNx;
	NeedsShiftAfterX = FFT1DInfo.ApplyAutoShiftAfter && (::fabs(x0_After) > RelShiftTol * xStepNx);

	double xStartTr = -0.5 / FFT1DInfo.xStep;

	NeedsShiftBeforeX = 0;
	double x0_Before = 0.;

	if (FFT1DInfo.UseGivenStartTrValue)
	{
		x0_Before = (FFT1DInfo.xStartTr - xStartTr);
		NeedsShiftBeforeX = (::fabs(x0_Before) > RelShiftTol * (::fabs(xStartTr)));
	}

#ifdef _CUFFT
	m_ArrayShiftX = 0;
	m_dArrayShiftX = 0;
	if (NeedsShiftBeforeX || NeedsShiftAfterX)
	{
		if (FFT1DInfo.pInData != 0)
		{
			cudaMallocManaged((void**)&m_ArrayShiftX, sizeof(float) * Nx * 2);
			if (m_ArrayShiftX == 0) return MEMORY_ALLOCATION_FAILURE;
		}
		else if (FFT1DInfo.pdInData != 0)
		{
			cudaMallocManaged((void**)&m_dArrayShiftX, sizeof(double) * Nx * 2);
			if (m_dArrayShiftX == 0) return MEMORY_ALLOCATION_FAILURE;
		}
	}
#else
	m_ArrayShiftX = 0;
	m_dArrayShiftX = 0;
	if (NeedsShiftBeforeX || NeedsShiftAfterX)
	{
		if (FFT1DInfo.pInData != 0)
		{
			m_ArrayShiftX = new float[Nx << 1];
			if (m_ArrayShiftX == 0) return MEMORY_ALLOCATION_FAILURE;
		}
		else if (FFT1DInfo.pdInData != 0)
		{
			m_dArrayShiftX = new double[Nx << 1];
			if (m_dArrayShiftX == 0) return MEMORY_ALLOCATION_FAILURE;
		}
	}
#endif
	//#define _CUFFT
#ifdef _FFTW3 //OC28012019
#ifdef _CUFFT
	cufftHandle Plan1DFFT;
	//fftwf_plan Plan1DFFT;
	fftwf_complex* DataToFFT = 0, * OutDataFFT = 0; //, *pOutDataFFT=0;
	cufftComplex* DataToFFT_cufft = 0, * OutDataFFT_cufft = 0;

	cufftHandle dPlan1DFFT;
	//fftw_plan dPlan1DFFT;
	fftw_complex* dDataToFFT = 0, * dOutDataFFT = 0; //, *pdOutDataFFT=0;
	cufftDoubleComplex* dDataToFFT_cufft = 0, * dOutDataFFT_cufft = 0;

	if ((FFT1DInfo.pInData != 0) && (FFT1DInfo.pOutData != 0))
	{
		DataToFFT = (fftwf_complex*)(FFT1DInfo.pInData);
		OutDataFFT = (fftwf_complex*)(FFT1DInfo.pOutData);

		DataToFFT_cufft = (cufftComplex*)DataToFFT;
		OutDataFFT_cufft = (cufftComplex*)OutDataFFT;
		//cudaMallocManaged((void**)&DataToFFT_cufft, Nx * sizeof(cufftComplex));
		//cudaMallocManaged((void**)&OutDataFFT_cufft, Nx * sizeof(cufftComplex));
		//cudaMemcpy(DataToFFT_cufft, DataToFFT, Nx * sizeof(cufftComplex), cudaMemcpyHostToDevice);

		//DataToFFT_cufft = (cufftComplex*)DataToFFT;
		//OutDataFFT_cufft = (cufftComplex*)OutDataFFT;

		//pOutDataFFT = OutDataFFT; //OC03092016 to be used solely in fftw call
	}
	else if ((FFT1DInfo.pdInData != 0) && (FFT1DInfo.pdOutData != 0))
	{
		dDataToFFT = (fftw_complex*)(FFT1DInfo.pdInData);
		dOutDataFFT = (fftw_complex*)(FFT1DInfo.pdOutData);

		dDataToFFT_cufft = (cufftDoubleComplex*)dDataToFFT;
		dOutDataFFT_cufft = (cufftDoubleComplex*)dOutDataFFT;

		//cudaMallocManaged((void**)&dDataToFFT_cufft, Nx * sizeof(cufftDoubleComplex));
		//cudaMallocManaged((void**)&dOutDataFFT_cufft, Nx * sizeof(cufftDoubleComplex));
		//cudaMemcpy(dDataToFFT_cufft, dDataToFFT, Nx * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);

		//dDataToFFT_cufft = (cufftDoubleComplex*)dDataToFFT;
		//dOutDataFFT_cufft = (cufftDoubleComplex*)dOutDataFFT;

		//pdOutDataFFT = dOutDataFFT;
	}
#else
	fftwf_plan Plan1DFFT;
	fftwf_complex* DataToFFT = 0, * OutDataFFT = 0; //, *pOutDataFFT=0;

	fftw_plan dPlan1DFFT;
	fftw_complex* dDataToFFT = 0, * dOutDataFFT = 0; //, *pdOutDataFFT=0;

	if ((FFT1DInfo.pInData != 0) && (FFT1DInfo.pOutData != 0))
	{
		DataToFFT = (fftwf_complex*)(FFT1DInfo.pInData);
		OutDataFFT = (fftwf_complex*)(FFT1DInfo.pOutData);
		//pOutDataFFT = OutDataFFT; //OC03092016 to be used solely in fftw call
	}
	else if ((FFT1DInfo.pdInData != 0) && (FFT1DInfo.pdOutData != 0))
	{
		dDataToFFT = (fftw_complex*)(FFT1DInfo.pdInData);
		dOutDataFFT = (fftw_complex*)(FFT1DInfo.pdOutData);
		//pdOutDataFFT = dOutDataFFT;
	}
#endif
#else
	fftw_plan Plan1DFFT;
	FFTW_COMPLEX* DataToFFT = (FFTW_COMPLEX*)(FFT1DInfo.pInData);
	FFTW_COMPLEX* OutDataFFT = (FFTW_COMPLEX*)(FFT1DInfo.pOutData);
	FFTW_COMPLEX* pOutDataFFT = OutDataFFT; //OC03092016 to be used solely in fftw call
/**
	Pointed-out by Sergey Yakubov (E-XFEL).
	From FFTW 2.1.5 docs:
	void fftw(fftw_plan plan, int howmany,
		  fftw_complex *in, int istride, int idist,
		  fftw_complex *out, int ostride, int odist);
	...
	out, ostride and odist describe the output array(s). The format is the same as for the input array.
	In-place transforms:  If the plan specifies an in-place transform, ostride and odist are always ignored.
	If out is NULL, out is ignored, too. Otherwise, out is interpreted as a pointer to an array of n complex numbers,
	that FFTW will use as temporary space to perform the in-place computation. out is used as scratch space and its contents destroyed.
	In this case, out must be an ordinary array whose elements are contiguous in memory (no striding).
**/
#endif

	char t0SignMult = (FFT1DInfo.Dir > 0) ? -1 : 1;
	if (NeedsShiftBeforeX)
	{
#ifdef _CUFFT
		if (m_ArrayShiftX != 0) FillArrayShift_CUDA(t0SignMult * x0_Before, FFT1DInfo.xStep, Nx, m_ArrayShiftX);
		else if (m_dArrayShiftX != 0) FillArrayShift_CUDA(t0SignMult * x0_Before, FFT1DInfo.xStep, Nx, m_dArrayShiftX);

		/*if (m_ArrayShiftX != 0) {
			float* m_ArrayShiftX_l = new float[Nx << 1];
			cudaMemcpy(m_ArrayShiftX_l, m_ArrayShiftX, sizeof(float)* Nx * 2, cudaMemcpyDeviceToHost);
			FillArrayShift(t0SignMult * x0_Before, FFT1DInfo.xStep, m_ArrayShiftX_l); //OC02022019
			cudaMemcpy(m_ArrayShiftX, m_ArrayShiftX_l, sizeof(float)* Nx * 2, cudaMemcpyHostToDevice);

			auto tmp = m_ArrayShiftX;
			m_ArrayShiftX = m_ArrayShiftX_l;

			cudaMemcpy(DataToFFT, DataToFFT_cufft, sizeof(float)* Nx * 2, cudaMemcpyDeviceToHost);
			TreatShift(DataToFFT, FFT1DInfo.HowMany);
			cudaMemcpy(DataToFFT_cufft, DataToFFT, sizeof(float)* Nx * 2, cudaMemcpyHostToDevice);

			m_ArrayShiftX = tmp;

			delete[] m_ArrayShiftX_l;
		}
		else if (m_dArrayShiftX != 0) {
			double* m_ArrayShiftX_l = new double[Nx << 1];
			cudaMemcpy(m_ArrayShiftX_l, m_dArrayShiftX, sizeof(double) * Nx * 2, cudaMemcpyDeviceToHost);
			FillArrayShift(t0SignMult * x0_Before, FFT1DInfo.xStep, m_ArrayShiftX_l);
			cudaMemcpy(m_dArrayShiftX, m_ArrayShiftX_l, sizeof(double)* Nx * 2, cudaMemcpyHostToDevice);

			auto tmp = m_dArrayShiftX;
			m_dArrayShiftX = m_ArrayShiftX_l;

			cudaMemcpy(dDataToFFT, dDataToFFT_cufft, sizeof(double)* Nx * 2, cudaMemcpyDeviceToHost);
			TreatShift(dDataToFFT, FFT1DInfo.HowMany);
			cudaMemcpy(dDataToFFT_cufft, dDataToFFT, sizeof(double)* Nx * 2, cudaMemcpyHostToDevice);

			m_dArrayShiftX = tmp;

			delete[] m_ArrayShiftX_l;
		}*/

		if (DataToFFT != 0) TreatShift_CUDA((float*)DataToFFT_cufft, FFT1DInfo.HowMany, Nx, m_ArrayShiftX);
		else if (dDataToFFT != 0) TreatShift_CUDA((double*)dDataToFFT_cufft, FFT1DInfo.HowMany, Nx, m_dArrayShiftX);
#else
		//FillArrayShift(t0SignMult*x0_Before, FFT1DInfo.xStep);
		if (m_ArrayShiftX != 0) FillArrayShift(t0SignMult * x0_Before, FFT1DInfo.xStep, m_ArrayShiftX);
		else if (m_dArrayShiftX != 0) FillArrayShift(t0SignMult * x0_Before, FFT1DInfo.xStep, m_dArrayShiftX);

		if (DataToFFT != 0) TreatShift(DataToFFT, FFT1DInfo.HowMany);

#ifdef _FFTW3 //OC27022019
		else if (dDataToFFT != 0) TreatShift(dDataToFFT, FFT1DInfo.HowMany);
#endif
#endif
	}

	//Added by S.Yakubov (for profiling?) at parallelizing SRW via OpenMP:
	//srwlPrintTime("::Make1DFFT : before fft",&start);

	int flags = FFTW_ESTIMATE; //OC30012019

	if (FFT1DInfo.Dir > 0)
	{
		//int flags = FFTW_ESTIMATE;
#ifdef _FFTW3 //OC28012019
#ifdef _WITH_OMP
		//Still needs to be tested!
		if (DataToFFT != 0)
		{
			fftwf_init_threads(); //initialize threading support
			int nthreads = omp_get_max_threads(); //detect number of OpenMP threads that are available
			fftwf_plan_with_nthreads(nthreads);
		}
		else if (dDataToFFT != 0) //OC02022019
		{
			fftw_init_threads(); //initialize threading support
			int nthreads = omp_get_max_threads(); //detect number of OpenMP threads that are available
			fftw_plan_with_nthreads(nthreads);
		}
#endif //ifndef _WITH_OMP
#ifdef _CUFFT
		int arN[] = { (int)Nx }; //OC14052020
		//int arN[] = {Nx};
		if (DataToFFT != 0)
		{
			cufftPlanMany(&Plan1DFFT, 1, arN, NULL, 1, Nx, NULL, 1, Nx, CUFFT_C2C, 1);
			if (Plan1DFFT == 0) return ERROR_IN_FFT;
			cufftExecC2C(Plan1DFFT, DataToFFT_cufft, OutDataFFT_cufft, CUFFT_FORWARD);
		}
		else if (dDataToFFT != 0) //OC02022019
		{
			cufftPlanMany(&dPlan1DFFT, 1, arN, NULL, 1, Nx, NULL, 1, Nx, CUFFT_Z2Z, 1);
			if (dPlan1DFFT == 0) return ERROR_IN_FFT;
			cufftExecZ2Z(dPlan1DFFT, dDataToFFT_cufft, dOutDataFFT_cufft, CUFFT_FORWARD);
		}
		/*if (DataToFFT != 0)
		{
			//Plan1DFFT = fftwf_plan_many_dft(1, arN, FFT1DInfo.HowMany, DataToFFT, NULL, 1, Nx, pOutDataFFT, NULL, 1, Nx, FFTW_FORWARD, flags);
			Plan1DFFT = fftwf_plan_many_dft(1, arN, FFT1DInfo.HowMany, (fftwf_complex*)DataToFFT_cufft, NULL, 1, Nx, (fftwf_complex*)OutDataFFT_cufft, NULL, 1, Nx, FFTW_FORWARD, flags); //OC02022019
			if (Plan1DFFT == 0) return ERROR_IN_FFT;
			fftwf_execute(Plan1DFFT);
		}
		else if (dDataToFFT != 0) //OC02022019
		{
			dPlan1DFFT = fftw_plan_many_dft(1, arN, FFT1DInfo.HowMany, (fftw_complex*)dDataToFFT_cufft, NULL, 1, Nx, (fftw_complex*)dOutDataFFT_cufft, NULL, 1, Nx, FFTW_FORWARD, flags);
			if (dPlan1DFFT == 0) return ERROR_IN_FFT;
			fftw_execute(dPlan1DFFT);
		}*/
#else
		int arN[] = { (int)Nx }; //OC14052020
		//int arN[] = {Nx};
		if (DataToFFT != 0)
		{
			//Plan1DFFT = fftwf_plan_many_dft(1, arN, FFT1DInfo.HowMany, DataToFFT, NULL, 1, Nx, pOutDataFFT, NULL, 1, Nx, FFTW_FORWARD, flags); 
			Plan1DFFT = fftwf_plan_many_dft(1, arN, FFT1DInfo.HowMany, DataToFFT, NULL, 1, Nx, OutDataFFT, NULL, 1, Nx, FFTW_FORWARD, flags); //OC02022019
			if (Plan1DFFT == 0) return ERROR_IN_FFT;
			fftwf_execute(Plan1DFFT);
		}
		else if (dDataToFFT != 0) //OC02022019
		{
			dPlan1DFFT = fftw_plan_many_dft(1, arN, FFT1DInfo.HowMany, dDataToFFT, NULL, 1, Nx, dOutDataFFT, NULL, 1, Nx, FFTW_FORWARD, flags);
			if (dPlan1DFFT == 0) return ERROR_IN_FFT;
			fftw_execute(dPlan1DFFT);
		}
#endif

#else //ifndef _FFTW3
		if (DataToFFT == OutDataFFT)
		{
			flags |= FFTW_IN_PLACE;
			pOutDataFFT = 0; //OC03092016 (see FFTW 2.1.5 doc clause above)
		}
		Plan1DFFT = fftw_create_plan(Nx, FFTW_FORWARD, flags);
		if (Plan1DFFT == 0) return ERROR_IN_FFT;

		//Added by S.Yakubov (for profiling?) at parallelizing SRW via OpenMP:
		//srwlPrintTime("::Make1DFFT : fft create plan dir>0",&start);

#ifndef _WITH_OMP //OC27102018
		//fftw(Plan1DFFT, FFT1DInfo.HowMany, DataToFFT, 1, Nx, OutDataFFT, 1, Nx);
		fftw(Plan1DFFT, FFT1DInfo.HowMany, DataToFFT, 1, Nx, pOutDataFFT, 1, Nx); //OC03092016
#else //OC27102018
		//SY: split one call into many (for OpenMP)
#pragma omp parallel for if (omp_get_num_threads()==1) // to avoid nested multi-threading (just in case)
		for (int i = 0; i < FFT1DInfo.HowMany; i++)
		{
			//SY: do not use OutDataFFT as scratch space if in-place
			if (DataToFFT == OutDataFFT) fftw_one(Plan1DFFT, DataToFFT + i * Nx, 0);
			else fftw_one(Plan1DFFT, DataToFFT + i * Nx, OutDataFFT + i * Nx);
		}
#endif
#endif
		//Added by S.Yakubov (for profiling?) at parallelizing SRW via OpenMP:
		//srwlPrintTime("::Make1DFFT : fft  dir>0",&start);

#ifdef _CUFFT
		if (OutDataFFT != 0)
		{
			//cudaMemcpy(OutDataFFT, OutDataFFT_cufft, sizeof(float) * Nx * 2, cudaMemcpyDeviceToHost);
			//RepairSignAfter1DFFT(OutDataFFT, FFT1DInfo.HowMany);
			//RotateDataAfter1DFFT(OutDataFFT, FFT1DInfo.HowMany);
			//cudaMemcpy(OutDataFFT_cufft, OutDataFFT, sizeof(float) * Nx * 2, cudaMemcpyHostToDevice);
			//RepairSignAfter1DFFT_CUDA((float*)OutDataFFT_cufft, FFT1DInfo.HowMany, Nx);
			//RotateDataAfter1DFFT_CUDA((float*)OutDataFFT_cufft, FFT1DInfo.HowMany, Nx);
			RepairAndRotateDataAfter1DFFT_CUDA((float*)OutDataFFT_cufft, FFT1DInfo.HowMany, Nx);
		}
		else if (dOutDataFFT != 0)
		{
			//cudaMemcpy(dOutDataFFT, dOutDataFFT_cufft, sizeof(double) * Nx * 2, cudaMemcpyDeviceToHost);
			//RepairSignAfter1DFFT(dOutDataFFT, FFT1DInfo.HowMany);
			//RotateDataAfter1DFFT(dOutDataFFT, FFT1DInfo.HowMany);
			//cudaMemcpy(dOutDataFFT_cufft, dOutDataFFT, sizeof(double) * Nx * 2, cudaMemcpyDeviceToHost);
			RepairAndRotateDataAfter1DFFT_CUDA((double*)dOutDataFFT_cufft, FFT1DInfo.HowMany, Nx);
			//RepairSignAfter1DFFT_CUDA((double*)dOutDataFFT_cufft, FFT1DInfo.HowMany, Nx);
			//RotateDataAfter1DFFT_CUDA((double*)dOutDataFFT_cufft, FFT1DInfo.HowMany, Nx);
		}
#else
		if (OutDataFFT != 0)
		{
			RepairSignAfter1DFFT(OutDataFFT, FFT1DInfo.HowMany);
			RotateDataAfter1DFFT(OutDataFFT, FFT1DInfo.HowMany);
		}
#ifdef _FFTW3 //OC27022019
		else if (dOutDataFFT != 0)
		{
			RepairSignAfter1DFFT(dOutDataFFT, FFT1DInfo.HowMany);
			RotateDataAfter1DFFT(dOutDataFFT, FFT1DInfo.HowMany);
		}
#endif
#endif

	}
	else
	{
		//int flags = FFTW_ESTIMATE; //OC30012019 (commented-out)
#ifdef _FFTW3 //OC28012019
#ifdef _WITH_OMP

		//Still needs to be tested!
		if (DataToFFT != 0)
		{
			fftwf_init_threads(); //initialize threading support
			int nthreads = omp_get_max_threads(); //detect number of OpenMP threads that are available
			fftwf_plan_with_nthreads(nthreads);
		}
		else if (dDataToFFT != 0)
		{
			fftw_init_threads(); //initialize threading support
			int nthreads = omp_get_max_threads(); //detect number of OpenMP threads that are available
			fftw_plan_with_nthreads(nthreads);
		}

#endif

#ifdef _CUFFT
		int arN[] = { (int)Nx }; //OC14052020
		//int arN[] = {Nx};
		if (DataToFFT != 0)
		{
			cufftPlanMany(&Plan1DFFT, 1, arN, NULL, 1, Nx, NULL, 1, Nx, CUFFT_C2C, 1);
			if (Plan1DFFT == 0) return ERROR_IN_FFT;

			RotateDataAfter1DFFT_CUDA((float*)DataToFFT_cufft, FFT1DInfo.HowMany, Nx);
			//cudaMemcpy(DataToFFT, DataToFFT_cufft, sizeof(float)* Nx * 2, cudaMemcpyDeviceToHost);
			//RotateDataAfter1DFFT(DataToFFT, FFT1DInfo.HowMany);
			//RepairSignAfter1DFFT(DataToFFT, FFT1DInfo.HowMany);
			//cudaMemcpy(DataToFFT_cufft, DataToFFT, sizeof(float)* Nx * 2, cudaMemcpyHostToDevice);
			RepairSignAfter1DFFT_CUDA((float*)DataToFFT_cufft, FFT1DInfo.HowMany, Nx);
			cufftExecC2C(Plan1DFFT, DataToFFT_cufft, OutDataFFT_cufft, CUFFT_INVERSE);
		}
		else if (dDataToFFT != 0) //OC02022019
		{
			cufftPlanMany(&dPlan1DFFT, 1, arN, NULL, 1, Nx, NULL, 1, Nx, CUFFT_Z2Z, 1);
			if (dPlan1DFFT == 0) return ERROR_IN_FFT;

			RotateDataAfter1DFFT_CUDA((double*)dDataToFFT_cufft, FFT1DInfo.HowMany, Nx);
			RepairSignAfter1DFFT_CUDA((double*)dDataToFFT_cufft, FFT1DInfo.HowMany, Nx);
			cufftExecZ2Z(dPlan1DFFT, dDataToFFT_cufft, dOutDataFFT_cufft, CUFFT_INVERSE);
		}
		/*if (DataToFFT != 0)
		{
			//Plan1DFFT = fftwf_plan_many_dft(1, arN, FFT1DInfo.HowMany, DataToFFT, NULL, 1, Nx, pOutDataFFT, NULL, 1, Nx, FFTW_BACKWARD, flags);
			Plan1DFFT = fftwf_plan_many_dft(1, arN, FFT1DInfo.HowMany, (fftwf_complex*)DataToFFT_cufft, NULL, 1, Nx, (fftwf_complex*)OutDataFFT_cufft, NULL, 1, Nx, FFTW_BACKWARD, flags); //OC02022019
			if (Plan1DFFT == 0) return ERROR_IN_FFT;

			//cudaMemcpy(DataToFFT, DataToFFT_cufft, sizeof(float)* Nx * 2, cudaMemcpyDeviceToHost);
			//RotateDataAfter1DFFT(DataToFFT, FFT1DInfo.HowMany);
			//RepairSignAfter1DFFT(DataToFFT, FFT1DInfo.HowMany);
			//cudaMemcpy(DataToFFT_cufft, DataToFFT, sizeof(float)* Nx * 2, cudaMemcpyHostToDevice);
			RotateDataAfter1DFFT_CUDA((float*)DataToFFT_cufft, FFT1DInfo.HowMany, Nx);
			RepairSignAfter1DFFT_CUDA((float*)DataToFFT_cufft, FFT1DInfo.HowMany, Nx);
			fftwf_execute(Plan1DFFT);
		}
		else if (dDataToFFT != 0) //OC02022019
		{
			dPlan1DFFT = fftw_plan_many_dft(1, arN, FFT1DInfo.HowMany, (fftw_complex*)dDataToFFT_cufft, NULL, 1, Nx, (fftw_complex*)dOutDataFFT_cufft, NULL, 1, Nx, FFTW_BACKWARD, flags);
			if (dPlan1DFFT == 0) return ERROR_IN_FFT;
			RotateDataAfter1DFFT_CUDA((double*)dDataToFFT_cufft, FFT1DInfo.HowMany, Nx);
			RepairSignAfter1DFFT_CUDA((double*)dDataToFFT_cufft, FFT1DInfo.HowMany, Nx);
			fftw_execute(dPlan1DFFT);
		}*/
#else
		int arN[] = { (int)Nx }; //OC14052020
//int arN[] = {Nx};
		if (DataToFFT != 0)
		{
			//Plan1DFFT = fftwf_plan_many_dft(1, arN, FFT1DInfo.HowMany, DataToFFT, NULL, 1, Nx, pOutDataFFT, NULL, 1, Nx, FFTW_BACKWARD, flags); 
			Plan1DFFT = fftwf_plan_many_dft(1, arN, FFT1DInfo.HowMany, DataToFFT, NULL, 1, Nx, OutDataFFT, NULL, 1, Nx, FFTW_BACKWARD, flags); //OC02022019
			if (Plan1DFFT == 0) return ERROR_IN_FFT;
			RotateDataAfter1DFFT(DataToFFT, FFT1DInfo.HowMany);
			RepairSignAfter1DFFT(DataToFFT, FFT1DInfo.HowMany);

			fftwf_execute(Plan1DFFT);
		}
		else if (dDataToFFT != 0) //OC02022019
		{
			dPlan1DFFT = fftw_plan_many_dft(1, arN, FFT1DInfo.HowMany, dDataToFFT, NULL, 1, Nx, dOutDataFFT, NULL, 1, Nx, FFTW_BACKWARD, flags);
			if (dPlan1DFFT == 0) return ERROR_IN_FFT;
			RotateDataAfter1DFFT(dDataToFFT, FFT1DInfo.HowMany);
			RepairSignAfter1DFFT(dDataToFFT, FFT1DInfo.HowMany);
			fftw_execute(dPlan1DFFT);
		}
#endif

#else //ifndef _FFTW3
		if (DataToFFT == OutDataFFT)
		{
			flags |= FFTW_IN_PLACE;
			pOutDataFFT = 0; //OC03092016 (see FFTW 2.1.5 doc clause above)
		}
		Plan1DFFT = fftw_create_plan(Nx, FFTW_BACKWARD, flags);
		if (Plan1DFFT == 0) return ERROR_IN_FFT;

		//Added by S.Yakubov (for profiling?) at parallelizing SRW via OpenMP:
		//srwlPrintTime("::Make1DFFT : fft create plan dir<0",&start);

		RotateDataAfter1DFFT(DataToFFT, FFT1DInfo.HowMany);
		//srwlPrintTime("::Make1DFFT : rotate dir<0",&start);

		RepairSignAfter1DFFT(DataToFFT, FFT1DInfo.HowMany);
		//srwlPrintTime("::Make1DFFT : repair dir<0",&start);

#ifndef _WITH_OMP //OC27102018
		//fftw(Plan1DFFT, FFT1DInfo.HowMany, DataToFFT, 1, Nx, OutDataFFT, 1, Nx);
		fftw(Plan1DFFT, FFT1DInfo.HowMany, DataToFFT, 1, Nx, pOutDataFFT, 1, Nx); //OC03092016
#else //OC27102018
		//SY: split one call into many (for OpenMP)
#pragma omp parallel for if (omp_get_num_threads()==1) // to avoid nested multi-threading (just in case)
		for (int i = 0; i < FFT1DInfo.HowMany; i++)
		{
			if (DataToFFT == OutDataFFT) fftw_one(Plan1DFFT, DataToFFT + i * Nx, 0);
			else fftw_one(Plan1DFFT, DataToFFT + i * Nx, OutDataFFT + i * Nx);
		}
#endif
#endif //_FFTW3
		//Added by S.Yakubov (for profiling?) at parallelizing SRW via OpenMP:
		//srwlPrintTime("::Make1DFFT : fft  dir<0",&start);
	}
	//double Mult = FFT1DInfo.xStep;
	double Mult = FFT1DInfo.xStep * FFT1DInfo.MultExtra;

#ifdef _CUFFT
	if (OutDataFFT != 0) {
		//cudaMemcpy(OutDataFFT, OutDataFFT_cufft, sizeof(float) * Nx * 2, cudaMemcpyDeviceToHost);
		//if (OutDataFFT[0][0] == NAN)
		//	printf("NAN_NORM\r\n");
		//NormalizeDataAfter1DFFT(OutDataFFT, FFT1DInfo.HowMany, Mult);
		//cudaMemcpy(OutDataFFT_cufft, OutDataFFT, sizeof(float) * Nx * 2, cudaMemcpyHostToDevice);

		NormalizeDataAfter1DFFT_CUDA((float*)OutDataFFT_cufft, FFT1DInfo.HowMany, Nx, Mult);
	}
	else if (dOutDataFFT != 0)
		NormalizeDataAfter1DFFT_CUDA((double*)dOutDataFFT_cufft, FFT1DInfo.HowMany, Nx, Mult);
#else
	if (OutDataFFT != 0) NormalizeDataAfter1DFFT(OutDataFFT, FFT1DInfo.HowMany, Mult);
#ifdef _FFTW3 //OC27022019
	else if (dOutDataFFT != 0) NormalizeDataAfter1DFFT(dOutDataFFT, FFT1DInfo.HowMany, Mult);
#endif
#endif


	//Added by S.Yakubov (for profiling?) at parallelizing SRW via OpenMP:
	//srwlPrintTime("::Make1DFFT : NormalizeDataAfter1DFFT",&start);

	if (NeedsShiftAfterX)
	{
#ifdef _CUFFT
		if (m_ArrayShiftX != 0) FillArrayShift_CUDA(t0SignMult * x0_After, FFT1DInfo.xStepTr, Nx, m_ArrayShiftX); //OC02022019
		else if (m_dArrayShiftX != 0) FillArrayShift_CUDA(t0SignMult * x0_After, FFT1DInfo.xStepTr, Nx, m_dArrayShiftX);

		/*if (m_ArrayShiftX != 0) {
			float* m_ArrayShiftX_l = new float[Nx << 1];
			cudaMemcpy(m_ArrayShiftX_l, m_ArrayShiftX, sizeof(float)* Nx * 2, cudaMemcpyDeviceToHost);
			FillArrayShift(t0SignMult * x0_After, FFT1DInfo.xStepTr, m_ArrayShiftX_l); //OC02022019
			cudaMemcpy(m_ArrayShiftX, m_ArrayShiftX_l, sizeof(float)* Nx * 2, cudaMemcpyHostToDevice);

			auto tmp = m_ArrayShiftX;
			m_ArrayShiftX = m_ArrayShiftX_l;

			cudaMemcpy(OutDataFFT, OutDataFFT_cufft, sizeof(float)* Nx * 2, cudaMemcpyDeviceToHost);
			TreatShift(OutDataFFT, FFT1DInfo.HowMany);
			cudaMemcpy(OutDataFFT_cufft, OutDataFFT, sizeof(float)* Nx * 2, cudaMemcpyHostToDevice);

			m_ArrayShiftX = tmp;

			delete[] m_ArrayShiftX_l;
		}
		else if (m_dArrayShiftX != 0) {
			double* m_ArrayShiftX_l = new double[Nx << 1];
			cudaMemcpy(m_ArrayShiftX_l, m_dArrayShiftX, sizeof(double) * Nx * 2, cudaMemcpyDeviceToHost);
			FillArrayShift(t0SignMult * x0_After, FFT1DInfo.xStepTr, m_ArrayShiftX_l);
			cudaMemcpy(m_dArrayShiftX, m_ArrayShiftX_l, sizeof(double)* Nx * 2, cudaMemcpyHostToDevice);

			cudaMemcpy(dOutDataFFT, dOutDataFFT_cufft, sizeof(double)* Nx * 2, cudaMemcpyDeviceToHost);
			TreatShift(dOutDataFFT, FFT1DInfo.HowMany);
			cudaMemcpy(dOutDataFFT_cufft, dOutDataFFT, sizeof(double)* Nx * 2, cudaMemcpyHostToDevice);
			delete[] m_ArrayShiftX_l;
		}*/

		if (OutDataFFT != 0) TreatShift_CUDA((float*)OutDataFFT_cufft, FFT1DInfo.HowMany, Nx, m_ArrayShiftX);
		else if (dOutDataFFT != 0) TreatShift_CUDA((double*)dOutDataFFT, FFT1DInfo.HowMany, Nx, m_dArrayShiftX);

#else
		//FillArrayShift(t0SignMult*x0_After, FFT1DInfo.xStepTr);
		if (m_ArrayShiftX != 0) FillArrayShift(t0SignMult * x0_After, FFT1DInfo.xStepTr, m_ArrayShiftX); //OC02022019
		else if (m_dArrayShiftX != 0) FillArrayShift(t0SignMult * x0_After, FFT1DInfo.xStepTr, m_dArrayShiftX);

		if (OutDataFFT != 0) TreatShift(OutDataFFT, FFT1DInfo.HowMany);
#ifdef _FFTW3 //OC27022019
		else if (dOutDataFFT != 0) TreatShift(dOutDataFFT, FFT1DInfo.HowMany);
#endif
#endif
	}

	//Added by S.Yakubov (for profiling?) at parallelizing SRW via OpenMP:
	//srwlPrintTime("::Make1DFFT : ProcessSharpEdges",&start);

	//OC_NERSC: to comment-out the following line for NERSC (to avoid crash with "python-mpi")
	//OC27102018: thread safety issue?
#ifdef _FFTW3 //OC29012019
#ifdef _CUFFT
	if (DataToFFT != 0) {
		//cudaMemcpy(OutDataFFT, OutDataFFT_cufft, Nx * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		//cudaFree(DataToFFT_cufft);
		//cudaFree(OutDataFFT_cufft);
		cufftDestroy(Plan1DFFT);
		//fftwf_destroy_plan(Plan1DFFT);
	}
	else if (dDataToFFT != 0) {
		//cudaMemcpy(dOutDataFFT, dOutDataFFT_cufft, Nx * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		//cudaFree(dDataToFFT_cufft);
		//cudaFree(dOutDataFFT_cufft);
		cufftDestroy(dPlan1DFFT);
		//fftw_destroy_plan(dPlan1DFFT);
	}
#else
	if (DataToFFT != 0) fftwf_destroy_plan(Plan1DFFT);
	else if (dDataToFFT != 0) fftw_destroy_plan(dPlan1DFFT);

#ifdef _WITH_OMP 

	if (DataToFFT != 0) fftwf_cleanup_threads(); //??
	else if (dDataToFFT != 0) fftw_cleanup_threads();

#endif
#endif
#else //ifndef _FFTW3

	fftw_destroy_plan(Plan1DFFT);

#endif

#ifdef _CUFFT
	if (m_ArrayShiftX != 0)
	{
		cudaFree(m_ArrayShiftX); m_ArrayShiftX = 0;
	}
	if (m_dArrayShiftX != 0)
	{
		cudaFree(m_dArrayShiftX); m_dArrayShiftX = 0;
	}
#else
	if (m_ArrayShiftX != 0)
	{
		delete[] m_ArrayShiftX; m_ArrayShiftX = 0;
	}
	if (m_dArrayShiftX != 0)
	{
		delete[] m_dArrayShiftX; m_dArrayShiftX = 0;
	}
#endif



	//Added by S.Yakubov (for profiling?) at parallelizing SRW via OpenMP:
	//srwlPrintTime("::Make1DFFT : after fft ",&start);
	return 0;
}

//*************************************************************************
//Forward FFT (FFT2DInfo.Dir = 1?): Int f(x,y)*exp(-i*2*Pi*(qx*x + qy*y)) dx dy
//Backward FFT (FFT2DInfo.Dir = -1?): Int f(qx,qy)*exp(i*2*Pi*(qx*x + qy*y)) dqx dqy
//int CGenMathFFT2D::Make2DFFT(CGenMathFFT2DInfo& FFT2DInfo)
//Modification by S.Yakubov for parallelizing SRW via OpenMP:
// SY: creation (and deletion) of FFTW plans is not thread-safe. Therefore added option to use precreated plans
#ifdef _FFTW3 //OC29012019
int CGenMathFFT2D::Make2DFFT(CGenMathFFT2DInfo& FFT2DInfo, fftwf_plan* pPrecreatedPlan2DFFT, fftw_plan* pdPrecreatedPlan2DFFT)
//int CGenMathFFT2D::Make2DFFT(CGenMathFFT2DInfo& FFT2DInfo, fftwf_plan* pPrecreatedPlan2DFFT)
#else
int CGenMathFFT2D::Make2DFFT(CGenMathFFT2DInfo& FFT2DInfo, fftwnd_plan* pPrecreatedPlan2DFFT) //OC27102018
#endif
{// Assumes Nx, Ny even !
	const double RelShiftTol = 1.E-06;

	//debug
	//AuxDebug_TestFFT_Plans();
	//end debug

	SetupLimitsTr(FFT2DInfo);

	double xStepNx = FFT2DInfo.Nx * FFT2DInfo.xStep;
	double yStepNy = FFT2DInfo.Ny * FFT2DInfo.yStep;

	double x0_After = FFT2DInfo.xStart + 0.5 * xStepNx;
	double y0_After = FFT2DInfo.yStart + 0.5 * yStepNy;

	NeedsShiftAfterX = (::fabs(x0_After) > RelShiftTol * xStepNx);
	NeedsShiftAfterY = (::fabs(y0_After) > RelShiftTol * yStepNy);

	double xStartTr = -0.5 / FFT2DInfo.xStep;
	double yStartTr = -0.5 / FFT2DInfo.yStep;

	NeedsShiftBeforeX = NeedsShiftBeforeY = 0;
	double x0_Before = 0., y0_Before = 0.;
	if (FFT2DInfo.UseGivenStartTrValues)
	{
		x0_Before = (FFT2DInfo.xStartTr - xStartTr); // Sign should be probably reversed here: check!!!
		y0_Before = (FFT2DInfo.yStartTr - yStartTr); // Sign should be probably reversed here: check!!!

		NeedsShiftBeforeX = (::fabs(x0_Before) > RelShiftTol * (::fabs(xStartTr)));
		NeedsShiftBeforeY = (::fabs(y0_Before) > RelShiftTol * (::fabs(yStartTr)));
	}

	//ArrayShiftX = 0; ArrayShiftY = 0; 
	m_ArrayShiftX = 0; m_ArrayShiftY = 0; //OC02022019
	m_dArrayShiftX = 0; m_dArrayShiftY = 0;
	if (FFT2DInfo.pData != 0)
	{
		if (NeedsShiftBeforeX || NeedsShiftAfterX)
		{
			//ArrayShiftX = new float[Nx << 1];
			//if(ArrayShiftX == 0) return MEMORY_ALLOCATION_FAILURE;
			m_ArrayShiftX = new float[Nx << 1];
			if (m_ArrayShiftX == 0) return MEMORY_ALLOCATION_FAILURE;
		}
		if (NeedsShiftBeforeY || NeedsShiftAfterY)
		{
			//ArrayShiftY = new float[Ny << 1];
			//if(ArrayShiftY == 0) return MEMORY_ALLOCATION_FAILURE;
			m_ArrayShiftY = new float[Ny << 1];
			if (m_ArrayShiftY == 0) return MEMORY_ALLOCATION_FAILURE;
		}
	}
	else if (FFT2DInfo.pdData != 0)
	{
		if (NeedsShiftBeforeX || NeedsShiftAfterX)
		{
			m_dArrayShiftX = new double[Nx << 1];
			if (m_dArrayShiftX == 0) return MEMORY_ALLOCATION_FAILURE;
		}
		if (NeedsShiftBeforeY || NeedsShiftAfterY)
		{
			m_dArrayShiftY = new double[Ny << 1];
			if (m_dArrayShiftY == 0) return MEMORY_ALLOCATION_FAILURE;
		}
	}

#ifdef _CUFFT
	fftwf_plan Plan2DFFT;
	fftw_plan dPlan2DFFT;
	fftwf_complex* DataToFFT = 0;
	fftw_complex* dDataToFFT = 0;
	cufftComplex* DataToFFT_cufft = 0;
	cufftDoubleComplex* dDataToFFT_cufft = 0;

	if (FFT2DInfo.pData != 0) {
		DataToFFT = (fftwf_complex*)(FFT2DInfo.pData);
		cudaMallocManaged((void**)&DataToFFT_cufft, sizeof(cufftComplex) * Nx * Ny);
		cudaMemcpy(DataToFFT_cufft, DataToFFT, sizeof(cufftComplex) * Nx * Ny, cudaMemcpyHostToDevice);
	}
	else if (FFT2DInfo.pdData != 0) {
		dDataToFFT = (fftw_complex*)(FFT2DInfo.pdData); //OC02022019
		cudaMallocManaged((void**)&dDataToFFT_cufft, sizeof(cufftDoubleComplex) * Nx * Ny);
		cudaMemcpy(dDataToFFT_cufft, dDataToFFT, sizeof(cufftDoubleComplex) * Nx * Ny, cudaMemcpyHostToDevice);
	}
#elif _FFTW3 //OC28012019
	fftwf_plan Plan2DFFT;
	fftw_plan dPlan2DFFT;
	fftwf_complex* DataToFFT = 0;
	fftw_complex* dDataToFFT = 0;

	if (FFT2DInfo.pData != 0) DataToFFT = (fftwf_complex*)(FFT2DInfo.pData);
	else if (FFT2DInfo.pdData != 0) dDataToFFT = (fftw_complex*)(FFT2DInfo.pdData); //OC02022019

#else
	fftwnd_plan Plan2DFFT;
	FFTW_COMPLEX* DataToFFT = (FFTW_COMPLEX*)(FFT2DInfo.pData);
#endif

	char t0SignMult = (FFT2DInfo.Dir > 0) ? -1 : 1;

	//if(NeedsShiftBeforeX) FillArrayShift('x', t0SignMult*x0_Before, FFT2DInfo.xStep);
	//if(NeedsShiftBeforeY) FillArrayShift('y', t0SignMult*y0_Before, FFT2DInfo.yStep);
	if (NeedsShiftBeforeX)
	{//OC02022019
		if (m_ArrayShiftX != 0) FillArrayShift('x', t0SignMult * x0_Before, FFT2DInfo.xStep, m_ArrayShiftX);
		else if (m_dArrayShiftX != 0) FillArrayShift('x', t0SignMult * x0_Before, FFT2DInfo.xStep, m_dArrayShiftX);
	}
	if (NeedsShiftBeforeY)
	{//OC02022019
		if (m_ArrayShiftY != 0) FillArrayShift('y', t0SignMult * y0_Before, FFT2DInfo.yStep, m_ArrayShiftY);
		else if (m_dArrayShiftY != 0) FillArrayShift('y', t0SignMult * y0_Before, FFT2DInfo.yStep, m_dArrayShiftY);
	}
	if (NeedsShiftBeforeX || NeedsShiftBeforeY)
	{
#ifdef _CUFFT
		if (DataToFFT != 0) {
			TreatShifts(DataToFFT);
			cudaMemcpy(DataToFFT_cufft, DataToFFT, sizeof(cufftComplex) * Nx * Ny, cudaMemcpyHostToDevice);
		}
		else if (dDataToFFT != 0) {
			TreatShifts(dDataToFFT); //OC02022019
			cudaMemcpy(dDataToFFT_cufft, dDataToFFT, sizeof(cufftDoubleComplex) * Nx * Ny, cudaMemcpyHostToDevice);
		}
#else
		if (DataToFFT != 0) TreatShifts(DataToFFT);

#ifdef _FFTW3 //OC27022019
		else if (dDataToFFT != 0) TreatShifts(dDataToFFT); //OC02022019
#endif
#endif
	}

	if (FFT2DInfo.Dir > 0)
	{
		//Plan2DFFT = fftw2d_create_plan(Ny, Nx, FFTW_FORWARD, FFTW_IN_PLACE);
		//OC27102018
		//SY: adopted for OpenMP
#ifdef _CUFFT
		if (DataToFFT != 0)
		{
			if (pPrecreatedPlan2DFFT == 0) Plan2DFFT = fftwf_plan_dft_2d(Ny, Nx, (fftwf_complex*)DataToFFT_cufft, (fftwf_complex*)DataToFFT_cufft, FFTW_FORWARD, FFTW_ESTIMATE);
			else Plan2DFFT = *pPrecreatedPlan2DFFT;
			if (Plan2DFFT == 0) return ERROR_IN_FFT;

			fftwf_execute(Plan2DFFT);
		}
		else if (dDataToFFT != 0)
		{
			if (pdPrecreatedPlan2DFFT == 0) dPlan2DFFT = fftw_plan_dft_2d(Ny, Nx, (fftw_complex*)dDataToFFT, (fftw_complex*)dDataToFFT, FFTW_FORWARD, FFTW_ESTIMATE);
			else dPlan2DFFT = *pdPrecreatedPlan2DFFT;
			if (dPlan2DFFT == 0) return ERROR_IN_FFT;

			fftw_execute(dPlan2DFFT);
		}

#elif _FFTW3 //OC28012019

		if (DataToFFT != 0)
		{
			if (pPrecreatedPlan2DFFT == 0) Plan2DFFT = fftwf_plan_dft_2d(Ny, Nx, DataToFFT, DataToFFT, FFTW_FORWARD, FFTW_ESTIMATE);
			else Plan2DFFT = *pPrecreatedPlan2DFFT;
			if (Plan2DFFT == 0) return ERROR_IN_FFT;

			fftwf_execute(Plan2DFFT);
		}
		else if (dDataToFFT != 0)
		{
			if (pdPrecreatedPlan2DFFT == 0) dPlan2DFFT = fftw_plan_dft_2d(Ny, Nx, dDataToFFT, dDataToFFT, FFTW_FORWARD, FFTW_ESTIMATE);
			else dPlan2DFFT = *pdPrecreatedPlan2DFFT;
			if (dPlan2DFFT == 0) return ERROR_IN_FFT;

			fftw_execute(dPlan2DFFT);
		}

#else
		if (pPrecreatedPlan2DFFT == 0) Plan2DFFT = fftw2d_create_plan(Ny, Nx, FFTW_FORWARD, FFTW_IN_PLACE);
		else Plan2DFFT = *pPrecreatedPlan2DFFT;
		if (Plan2DFFT == 0) return ERROR_IN_FFT;
		fftwnd(Plan2DFFT, 1, DataToFFT, 1, 0, DataToFFT, 1, 0);
#endif

#ifdef _CUFFT
		if (DataToFFT != 0)
		{
			RepairSignAfter2DFFT_CUDA((float*)DataToFFT_cufft, Nx, Ny);
			RotateDataAfter2DFFT_CUDA((float*)DataToFFT_cufft, Nx, Ny);

			//cudaMemcpy(DataToFFT, DataToFFT_cufft, sizeof(cufftComplex) * Nx * Ny, cudaMemcpyDeviceToHost);
			//RepairSignAfter2DFFT(DataToFFT);
			//RotateDataAfter2DFFT(DataToFFT);
			//cudaMemcpy(DataToFFT_cufft, DataToFFT, sizeof(cufftComplex) * Nx * Ny, cudaMemcpyHostToDevice);
		}
		else if (dDataToFFT != 0)
		{
			RepairSignAfter2DFFT_CUDA((double*)dDataToFFT_cufft, Nx, Ny);
			RotateDataAfter2DFFT_CUDA((double*)dDataToFFT_cufft, Nx, Ny);

			//cudaMemcpy(dDataToFFT, dDataToFFT_cufft, sizeof(cufftDoubleComplex) * Nx * Ny, cudaMemcpyDeviceToHost);
			//RepairSignAfter2DFFT(dDataToFFT);
			//RotateDataAfter2DFFT(dDataToFFT);
			//cudaMemcpy(dDataToFFT_cufft, dDataToFFT, sizeof(cufftDoubleComplex) * Nx * Ny, cudaMemcpyHostToDevice);
		}
#else
		if (DataToFFT != 0)
		{
			RepairSignAfter2DFFT(DataToFFT);
			RotateDataAfter2DFFT(DataToFFT);
		}

#ifdef _FFTW3 //OC27022019
		else if (dDataToFFT != 0)
		{
			RepairSignAfter2DFFT(dDataToFFT);
			RotateDataAfter2DFFT(dDataToFFT);
		}
#endif
#endif
	}
	else
	{
		//Plan2DFFT = fftw2d_create_plan(Ny, Nx, FFTW_BACKWARD, FFTW_IN_PLACE);
		//OC27102018
		//SY: adopted for OpenMP
#ifdef _CUFFT
		if (DataToFFT != 0)
		{
			if (pPrecreatedPlan2DFFT == 0) Plan2DFFT = fftwf_plan_dft_2d(Ny, Nx, (fftwf_complex*)DataToFFT_cufft, (fftwf_complex*)DataToFFT_cufft, FFTW_BACKWARD, FFTW_ESTIMATE);
			else Plan2DFFT = *pPrecreatedPlan2DFFT;
			if (Plan2DFFT == 0) return ERROR_IN_FFT;
			//cudaMemcpy(DataToFFT, DataToFFT_cufft, sizeof(cufftComplex) * Nx * Ny, cudaMemcpyDeviceToHost);
			//RotateDataAfter2DFFT(DataToFFT);
			//RepairSignAfter2DFFT(DataToFFT);
			//cudaMemcpy(DataToFFT_cufft, DataToFFT, sizeof(cufftComplex) * Nx * Ny, cudaMemcpyHostToDevice);
			RotateDataAfter2DFFT_CUDA((float*)DataToFFT_cufft, Nx, Ny);
			RepairSignAfter2DFFT_CUDA((float*)DataToFFT_cufft, Nx, Ny);
			fftwf_execute(Plan2DFFT);
		}
		else if (dDataToFFT != 0)
		{
			if (pdPrecreatedPlan2DFFT == 0) dPlan2DFFT = fftw_plan_dft_2d(Ny, Nx, (fftw_complex*)dDataToFFT_cufft, (fftw_complex*)dDataToFFT_cufft, FFTW_BACKWARD, FFTW_ESTIMATE);
			else dPlan2DFFT = *pdPrecreatedPlan2DFFT;
			if (dPlan2DFFT == 0) return ERROR_IN_FFT;
			//cudaMemcpy(dDataToFFT, dDataToFFT_cufft, sizeof(cufftDoubleComplex) * Nx * Ny, cudaMemcpyDeviceToHost);
			//RotateDataAfter2DFFT(dDataToFFT);
			//RepairSignAfter2DFFT(dDataToFFT);
			//cudaMemcpy(dDataToFFT_cufft, dDataToFFT, sizeof(cufftDoubleComplex) * Nx * Ny, cudaMemcpyHostToDevice);
			RotateDataAfter2DFFT_CUDA((double*)dDataToFFT_cufft, Nx, Ny);
			RepairSignAfter2DFFT_CUDA((double*)dDataToFFT_cufft, Nx, Ny);
			fftw_execute(dPlan2DFFT);
		}
#elif _FFTW3 //OC28012019
		if (DataToFFT != 0)
		{
			if (pPrecreatedPlan2DFFT == 0) Plan2DFFT = fftwf_plan_dft_2d(Ny, Nx, DataToFFT, DataToFFT, FFTW_BACKWARD, FFTW_ESTIMATE);
			else Plan2DFFT = *pPrecreatedPlan2DFFT;
			if (Plan2DFFT == 0) return ERROR_IN_FFT;
			RotateDataAfter2DFFT(DataToFFT);
			RepairSignAfter2DFFT(DataToFFT);
			fftwf_execute(Plan2DFFT);
		}
		else if (dDataToFFT != 0)
		{
			if (pdPrecreatedPlan2DFFT == 0) dPlan2DFFT = fftw_plan_dft_2d(Ny, Nx, dDataToFFT, dDataToFFT, FFTW_BACKWARD, FFTW_ESTIMATE);
			else dPlan2DFFT = *pdPrecreatedPlan2DFFT;
			if (dPlan2DFFT == 0) return ERROR_IN_FFT;
			RotateDataAfter2DFFT(dDataToFFT);
			RepairSignAfter2DFFT(dDataToFFT);
			fftw_execute(dPlan2DFFT);
		}
#else
		if (pPrecreatedPlan2DFFT == 0) Plan2DFFT = fftw2d_create_plan(Ny, Nx, FFTW_BACKWARD, FFTW_IN_PLACE);
		else Plan2DFFT = *pPrecreatedPlan2DFFT;
		if (Plan2DFFT == 0) return ERROR_IN_FFT;
		RotateDataAfter2DFFT(DataToFFT);
		RepairSignAfter2DFFT(DataToFFT);
		fftwnd(Plan2DFFT, 1, DataToFFT, 1, 0, DataToFFT, 1, 0);
#endif
	}

	//double Mult = FFT2DInfo.xStep*FFT2DInfo.yStep;
	double Mult = FFT2DInfo.xStep * FFT2DInfo.yStep * FFT2DInfo.ExtraMult; //OC20112017
#ifdef _CUFFT
	if (DataToFFT != 0) {
		//cudaMemcpy(DataToFFT, DataToFFT_cufft, sizeof(cufftComplex) * Nx * Ny, cudaMemcpyDeviceToHost);
		//NormalizeDataAfter2DFFT(DataToFFT, Mult);
		//cudaMemcpy(DataToFFT_cufft, DataToFFT, sizeof(cufftComplex) * Nx * Ny, cudaMemcpyHostToDevice);
		NormalizeDataAfter2DFFT_CUDA((float*)DataToFFT_cufft, Nx, Ny, Mult);
	}
	else if (dDataToFFT != 0) {
		//cudaMemcpy(dDataToFFT, dDataToFFT_cufft, sizeof(cufftDoubleComplex) * Nx * Ny, cudaMemcpyDeviceToHost);
		//NormalizeDataAfter2DFFT(dDataToFFT, Mult);
		//cudaMemcpy(dDataToFFT_cufft, dDataToFFT, sizeof(cufftDoubleComplex) * Nx * Ny, cudaMemcpyHostToDevice);
		NormalizeDataAfter2DFFT_CUDA((double*)dDataToFFT_cufft, Nx, Ny, Mult);
	}
#else
	if (DataToFFT != 0) NormalizeDataAfter2DFFT(DataToFFT, Mult);

#ifdef _FFTW3 //OC27022019
	else if (dDataToFFT != 0) NormalizeDataAfter2DFFT(dDataToFFT, Mult);
#endif
#endif

	//if(NeedsShiftAfterX) FillArrayShift('x', t0SignMult*x0_After, FFT2DInfo.xStepTr);
	//if(NeedsShiftAfterY) FillArrayShift('y', t0SignMult*y0_After, FFT2DInfo.yStepTr);
	if (NeedsShiftAfterX)
	{//OC02022019
		if (m_ArrayShiftX != 0) FillArrayShift('x', t0SignMult * x0_After, FFT2DInfo.xStepTr, m_ArrayShiftX);
		else if (m_dArrayShiftX != 0) FillArrayShift('x', t0SignMult * x0_After, FFT2DInfo.xStepTr, m_dArrayShiftX);
	}
	if (NeedsShiftAfterY)
	{//OC02022019
		if (m_ArrayShiftY != 0) FillArrayShift('y', t0SignMult * y0_After, FFT2DInfo.yStepTr, m_ArrayShiftY);
		else if (m_dArrayShiftY != 0) FillArrayShift('y', t0SignMult * y0_After, FFT2DInfo.yStepTr, m_dArrayShiftY);
	}
	if (NeedsShiftAfterX || NeedsShiftAfterY)
	{
#ifdef _CUFFT
		if (DataToFFT != 0) {
			cudaMemcpy(DataToFFT, DataToFFT_cufft, sizeof(cufftComplex) * Nx * Ny, cudaMemcpyDeviceToHost);
			TreatShifts(DataToFFT);
			cudaMemcpy(DataToFFT_cufft, DataToFFT, sizeof(cufftComplex) * Nx * Ny, cudaMemcpyHostToDevice);
		}
		else if (dDataToFFT != 0) {
			cudaMemcpy(dDataToFFT, dDataToFFT_cufft, sizeof(cufftDoubleComplex) * Nx * Ny, cudaMemcpyDeviceToHost);
			TreatShifts(dDataToFFT); //OC02022019
			cudaMemcpy(dDataToFFT_cufft, dDataToFFT, sizeof(cufftDoubleComplex) * Nx * Ny, cudaMemcpyHostToDevice);
		}
#else
		if (DataToFFT != 0) TreatShifts(DataToFFT);

#ifdef _FFTW3 //OC27022019
		else if (dDataToFFT != 0) TreatShifts(dDataToFFT); //OC02022019
#endif
#endif
	}

	//OC_NERSC: to comment-out the following line for NERSC (to avoid crash with "python-mpi")
	//fftwnd_destroy_plan(Plan2DFFT);
	//OC27102018
	//SY: adopted for OpenMP

#ifdef _CUFFT
	if (DataToFFT != 0)
	{
		cudaMemcpy(DataToFFT, DataToFFT_cufft, sizeof(cufftComplex) * Nx * Ny, cudaMemcpyDeviceToHost);
		cudaFree(DataToFFT_cufft);
		if (pPrecreatedPlan2DFFT == 0) fftwf_destroy_plan(Plan2DFFT);
	}
	else if (dDataToFFT != 0) //OC03022019
	{
		cudaMemcpy(dDataToFFT, dDataToFFT_cufft, sizeof(cufftDoubleComplex) * Nx * Ny, cudaMemcpyDeviceToHost);
		cudaFree(dDataToFFT_cufft);
		if (pdPrecreatedPlan2DFFT == 0) fftw_destroy_plan(dPlan2DFFT);
	}
#elif _FFTW3 //OC28012019
	if (DataToFFT != 0)
	{
		if (pPrecreatedPlan2DFFT == 0) fftwf_destroy_plan(Plan2DFFT);
	}
	else if (dDataToFFT != 0) //OC03022019
	{
		if (pdPrecreatedPlan2DFFT == 0) fftw_destroy_plan(dPlan2DFFT);
	}
#else
	if (pPrecreatedPlan2DFFT == 0) fftwnd_destroy_plan(Plan2DFFT);
#endif

	//if(ArrayShiftX != 0) { delete[] ArrayShiftX; ArrayShiftX = 0;}
	//if(ArrayShiftY != 0) { delete[] ArrayShiftY; ArrayShiftY = 0;}
	if (m_ArrayShiftX != 0) { delete[] m_ArrayShiftX; m_ArrayShiftX = 0; }
	if (m_ArrayShiftY != 0) { delete[] m_ArrayShiftY; m_ArrayShiftY = 0; }
	if (m_dArrayShiftX != 0) { delete[] m_dArrayShiftX; m_dArrayShiftX = 0; } //OC02022019
	if (m_dArrayShiftY != 0) { delete[] m_dArrayShiftY; m_dArrayShiftY = 0; }

	return 0;
}