#include "deinterleave.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <stdio.h>
#include "summed_area_table.cuh"

__global__ void deinterleave_kernel(unsigned char* image, unsigned int NxN, unsigned int* r, unsigned int* g, unsigned int* b) 
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < NxN) {
		
		unsigned char* pixelOffset = image + tid * 3;
		r[tid] = pixelOffset[0];
		g[tid] = pixelOffset[1];
		b[tid] = pixelOffset[2];

		tid += blockDim.x * gridDim.x;
	}
}

void deinterleave(unsigned char* image, unsigned int NxN, unsigned int** r, unsigned int** g, unsigned int** b) 
{
	
	//input: img
	//output: r, g, b

	unsigned char* dev_img;
	cudaMalloc((void**)&dev_img, 3 * NxN * sizeof(char));

	unsigned int* dev_r, *dev_g, *dev_b;
	cudaMalloc((void**)&dev_r, NxN * sizeof(int));
	cudaMalloc((void**)&dev_g, NxN * sizeof(int));
	cudaMalloc((void**)&dev_b, NxN * sizeof(int));
	
	cudaMemcpy(dev_img, image, 3 * NxN * sizeof(char), cudaMemcpyHostToDevice);

	deinterleave_kernel << <512, 512>> > (dev_img, NxN, dev_r, dev_g, dev_b);
	
	cudaMemcpy(*r, dev_r, NxN * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(*g, dev_g, NxN * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(*b, dev_b, NxN * sizeof(int), cudaMemcpyDeviceToHost);
	
	/*for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			std::cout << *r[i * 512 + j] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;*/

	cudaFree(dev_r);
	cudaFree(dev_g);
	cudaFree(dev_b);
}


//int main()
//{
//	//input: img
//	//output: r, g, b
//
//	unsigned int N = 4; //N x N image
//
//	unsigned char* img = (unsigned char*)malloc(3 * N * N * sizeof(char));
//	for (int i = 0; i < 3 * N * N; i++)
//		img[i] = 1;
//	unsigned int* r = (unsigned int*)malloc(N * N * sizeof(int));
//	unsigned int* g = (unsigned int*)malloc(N * N * sizeof(int));
//	unsigned int* b = (unsigned int*)malloc(N * N *sizeof(int));
//
//	deinterleave(img, N * N, &r, &g, &b);
//
//	/*for (int i = 0; i < N; i++)
//	{
//		std::cout << r[i] << " " << g[i] << " " << b[i] << " ";
//	}*/
//
//	summed_area_table(g, N);
//
//	for (int i = 0; i < N; i++)
//	{
//		for (int j = 0; j < N; j++)
//		{
//			std::cout << g[i * N + j] << " ";
//		}
//		std::cout << std::endl;
//	}
//
//
//	return 0;
//}
