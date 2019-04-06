#include "Sum_Scan.cuh"

#include <iostream>

//https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html

__global__ void image_integral(int *a, int*b, int rowsTotal, int colsTotal)
{
	// Thread Ids equal to block Ids because the each blocks contains one thread only.
	int col = blockIdx.x;
	int row = blockIdx.y;
	int temp = 0;

	if (col < colsTotal && row < rowsTotal)
	{
		// The first loop iterates from zero to the Y index of the thread which represents the corresponding element of the output/input array.  
		for (int r = 0; r <= row; r++)
		{
			// The second loop iterates from zero to the X index of the thread which represents the corresponding element of the output/input array  
			for (int c = 0; c <= col; c++)
			{
				temp = temp + a[r*colsTotal + c];
			}
		}
	}

	//Transfer the final result to the output array
	b[row*colsTotal + col] = temp;
}


int main1()
{
	//M is number of rows
	//N is number of columns

	int M = 3, N = 2, m_e = 0;
	int total_e = M * N;
	int widthstep = total_e * sizeof(int);

	int* matrix_a = (int*)malloc(widthstep);
	int* matrix_b = (int*)malloc(widthstep);

	std::cout << "Enter elements for " << M << "x" << N << " matrix";

	for (int r = 0; r < M; r++)
	{
		for (int c = 0; c < N; c++)
		{
			std::cout << "Enter Matrix element [ " << r << "," << c << "]";
			std::cin >> m_e;
			matrix_a[r*N + c] = m_e;
			matrix_b[r*N + c] = 0;
		}
	}

	int * d_matrix_a, *d_matrix_b;

	std::cout << "Input:" << std::endl;

	for (int r = 0; r < M; r++)
	{
		for (int c = 0; c < N; c++)
		{
			std::cout << matrix_a[r*N + c] << " ";
		}
		std::cout << std::endl;
	}

	std::cout << std::endl;

	cudaMalloc(&d_matrix_a, widthstep);
	cudaMalloc(&d_matrix_b, widthstep);

	cudaMemcpy(d_matrix_a, matrix_a, widthstep, cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrix_b, matrix_b, widthstep, cudaMemcpyHostToDevice);

	//Creating a grid where the number of blocks are equal to the number of pixels or input matrix elements.

	//Each block contains only one thread.

	dim3 grid(N, M);

	image_integral << <grid, 1 >> > (d_matrix_a, d_matrix_b, M, N);

	cudaThreadSynchronize();

	cudaMemcpy(matrix_b, d_matrix_b, widthstep, cudaMemcpyDeviceToHost);

	std::cout << "The Summed Area table is: " << std::endl;

	for (int r = 0; r < M; r++)
	{
		for (int c = 0; c < N; c++)
		{
			std::cout << matrix_b[r*N + c] << " ";
		}
		std::cout << std::endl;
	}

	system("pause");

	cudaFree(d_matrix_a);
	cudaFree(d_matrix_b);
	free(matrix_a);
	free(matrix_b);

	return 0;
}