#include <stdio.h>

#include <iostream>
#include <math.h>
#include <functional>
#include <stdlib.h> 
#include <time.h>   
#include <iostream>
#include <chrono>
#include <unistd.h>   
#include <fstream>
#include <string>

using namespace std;

#define TILE_SIZE 32

// Matrix multiplication kernel
__global__ void matrixMultiply(float *A, float *B, float *C, int size) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0;

    if( row < size && col < size ){
    // do the multiplication for one row and col using local variable 
    for(int k = 0; k < size; k++){
      sum += A[row * size + k] * B[k * size + col];
    }
    // store result
    C[row * size + col] = sum;
  }
  
}


int main() {
    int size = 128;
    int matrixSize = size * size;

    // Allocate memory for matrices on the host
    float *h_A, *h_B, *h_C;
    h_A = (float *)malloc(matrixSize * sizeof(float));
    h_B = (float *)malloc(matrixSize * sizeof(float));
    h_C = (float *)malloc(matrixSize * sizeof(float));

    // Initialize matrices A and B
    for (int i = 0; i < matrixSize; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i);
    }

    // Allocate memory for matrices on the device
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, matrixSize * sizeof(float));
    cudaMalloc((void **)&d_B, matrixSize * sizeof(float));
    cudaMalloc((void **)&d_C, matrixSize * sizeof(float));

    // Copy matrices A and B from host to device
    cudaMemcpy(d_A, h_A, matrixSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, matrixSize * sizeof(float), cudaMemcpyHostToDevice);

    // Set grid and block sizes
    dim3 gridSize(size / TILE_SIZE, size / TILE_SIZE);
    dim3 blockSize(TILE_SIZE, TILE_SIZE);

  auto start = chrono::steady_clock::now();
    // Launch the matrix multiplication kernel
    matrixMultiply<<<gridSize, blockSize>>>(d_A, d_B, d_C, size);
  auto end = chrono::steady_clock::now();
  cout << "GPU Elapsed time in nanoseconds: "
        << chrono::duration_cast<chrono::nanoseconds>(end - start).count()
        << " ns" << endl;

    // Copy the result matrix from device to host
    cudaMemcpy(h_C, d_C, matrixSize * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}

