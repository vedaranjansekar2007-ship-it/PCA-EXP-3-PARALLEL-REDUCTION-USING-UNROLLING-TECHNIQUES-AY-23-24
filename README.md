# PCA-EXP-3-PARALLEL-REDUCTION-USING-UNROLLING-TECHNIQUES AY 23-24
<h3>AIM:</h3>
<h3>NAME VEDARANJAN S </h3>
<h3> REGISTER NO : 212225220119</h3>
<h3>EX. NO : 03</h3>
<h3>DATE</h3>
<h1> <align=center> PARALLEL REDUCTION USING UNROLLING TECHNIQUES </h3>
  Refer to the kernel reduceUnrolling8 and implement the kernel reduceUnrolling16, in which each thread handles 16 data blocks. Compare kernel performance with reduceUnrolling8 and use the proper metrics and events with nvprof to explain any difference in performance.</h3>

## AIM:
To implement the kernel reduceUnrolling16 and comapare the performance of kernal reduceUnrolling16 with kernal reduceUnrolling8 using nvprof.
## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler
## PROCEDURE:
1.	Initialization and Memory Allocation
2.	Define the input size n.
3.	Allocate host memory (h_idata and h_odata) for input and output data.
Input Data Initialization
4.	Initialize the input data on the host (h_idata) by assigning a value of 1 to each element.
Device Memory Allocation
5.	Allocate device memory (d_idata and d_odata) for input and output data on the GPU.
Data Transfer: Host to Device
6.	Copy the input data from the host (h_idata) to the device (d_idata) using cudaMemcpy.
Grid and Block Configuration
7.	Define the grid and block dimensions for the kernel launch:
8.	Each block consists of 256 threads.
9.	Calculate the grid size based on the input size n and block size.
10.	Start CPU Timer
11.	Initialize a CPU timer to measure the CPU execution time.
12.	Compute CPU Sum
13.	Calculate the sum of the input data on the CPU using a for loop and store the result in sum_cpu.
14.	Stop CPU Timer
15.	Record the elapsed CPU time.
16.	Start GPU Timer
17.	Initialize a GPU timer to measure the GPU execution time.
Kernel Execution
18.	Launch the reduceUnrolling16 kernel on the GPU with the specified grid and block dimensions.
Data Transfer: Device to Host
19.	Copy the result data from the device (d_odata) to the host (h_odata) using cudaMemcpy.
20.	Compute GPU Sum
21.	Calculate the final sum on the GPU by summing the elements in h_odata and store the result in sum_gpu.
22.	Stop GPU Timer
23.	Record the elapsed GPU time.
24.	Print Results
25.	Display the computed CPU sum, GPU sum, CPU elapsed time, and GPU elapsed time.
Memory Deallocation
26.	Free the allocated host and device memory using free and cudaFree.
27.	Exit
28.	Return from the main function.

## PROGRAM:
```
%%cuda

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <sys/time.h>

// CUDA Error Checking
#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        fprintf(stderr, "CUDA Error: %s\n", \
                cudaGetErrorString(error)); \
        exit(1); \
    } \
}

// Calculate execution time
inline double seconds()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

// GPU Reduction with Loop Unrolling 8
__global__ void reduceUnrolling8(int *g_idata,
                                  long long *g_odata,
                                  unsigned int n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    if (idx + 7 * blockDim.x < n)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int b1 = g_idata[idx + 4 * blockDim.x];
        int b2 = g_idata[idx + 5 * blockDim.x];
        int b3 = g_idata[idx + 6 * blockDim.x];
        int b4 = g_idata[idx + 7 * blockDim.x];

        g_idata[idx] = a1 + a2 + a3 + a4 +
                       b1 + b2 + b3 + b4;
    }

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}

// GPU Reduction with Loop Unrolling 16
__global__ void reduceUnrolling16(int *g_idata,
                                   long long *g_odata,
                                   unsigned int n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 16 + threadIdx.x;

    int *idata = g_idata + blockIdx.x * blockDim.x * 16;

    if (idx + 15 * blockDim.x < n)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];

        int b1 = g_idata[idx + 4 * blockDim.x];
        int b2 = g_idata[idx + 5 * blockDim.x];
        int b3 = g_idata[idx + 6 * blockDim.x];
        int b4 = g_idata[idx + 7 * blockDim.x];

        int c1 = g_idata[idx + 8 * blockDim.x];
        int c2 = g_idata[idx + 9 * blockDim.x];
        int c3 = g_idata[idx + 10 * blockDim.x];
        int c4 = g_idata[idx + 11 * blockDim.x];

        int d1 = g_idata[idx + 12 * blockDim.x];
        int d2 = g_idata[idx + 13 * blockDim.x];
        int d3 = g_idata[idx + 14 * blockDim.x];
        int d4 = g_idata[idx + 15 * blockDim.x];

        g_idata[idx] = a1 + a2 + a3 + a4 +
                       b1 + b2 + b3 + b4 +
                       c1 + c2 + c3 + c4 +
                       d1 + d2 + d3 + d4;
    }

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}

// CPU Recursive Reduction
long long recursiveReduce(long long *data, const int size)
{
    if (size == 1)
        return data[0];

    int stride = size / 2;

    for (int i = 0; i < stride; i++)
    {
        data[i] += data[i + stride];
    }

    return recursiveReduce(data, stride);
}

// Main Function
int main(int argc, char **argv)
{
    int dev = 0;
    cudaDeviceProp deviceProp;

    CHECK(cudaGetDeviceProperties(&deviceProp, dev));

    printf("%s starting reduction at ", argv[0]);
    printf("device %d: %s\n", dev, deviceProp.name);

    CHECK(cudaSetDevice(dev));

    // Array initialization
    int size = 1 << 24;

    printf("\nArray size: %d\n", size);

    // Block configuration
    int blocksize = 512;

    if (argc > 1)
        blocksize = atoi(argv[1]);

    if (blocksize != 512)
    {
        printf("This program uses block size 512.\n");
        return EXIT_FAILURE;
    }

    dim3 block(blocksize, 1);
    dim3 grid((size + block.x - 1) / block.x, 1);

    printf("Grid: %d  Block: %d\n", grid.x, block.x);

    // Allocate host memory
    size_t bytes = (size_t)size * sizeof(int);

    int *h_idata = (int *)malloc(bytes);
    long long *h_odata =
        (long long *)malloc((grid.x / 8) * sizeof(long long));

    long long *tmp =
        (long long *)malloc((size_t)size * sizeof(long long));

    if (!h_idata || !h_odata || !tmp)
    {
        printf("Host memory allocation failed!\n");
        return EXIT_FAILURE;
    }

    // Initialize input array
    for (int i = 0; i < size; i++)
    {
        h_idata[i] = (int)(rand() & 0xFF);
        tmp[i] = h_idata[i];
    }

    double iStart, iElaps;

    // CPU Reduction
    iStart = seconds();

    long long cpu_sum = recursiveReduce(tmp, size);

    iElaps = seconds() - iStart;

    printf("CPU Reduction      elapsed %f sec  CPU sum: %lld\n",
           iElaps, cpu_sum);

    // Allocate GPU memory
    int *d_idata = NULL;
    long long *d_odata = NULL;

    CHECK(cudaMalloc((void **)&d_idata, bytes));

    CHECK(cudaMalloc((void **)&d_odata,
                     (grid.x / 8) * sizeof(long long)));

    long long gpu_sum8 = 0;
    long long gpu_sum16 = 0;

    // GPU Reduction - Unrolling 8
    CHECK(cudaMemcpy(d_idata, h_idata, bytes,
                     cudaMemcpyHostToDevice));

    CHECK(cudaDeviceSynchronize());

    iStart = seconds();

    reduceUnrolling8<<<grid.x / 8, block>>>(
        d_idata, d_odata, size);

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    iElaps = seconds() - iStart;

    CHECK(cudaMemcpy(h_odata, d_odata,
                     (grid.x / 8) * sizeof(long long),
                     cudaMemcpyDeviceToHost));

    for (int i = 0; i < grid.x / 8; i++)
        gpu_sum8 += h_odata[i];

    printf("GPU Unrolling 8    elapsed %f sec  GPU sum: %lld\n",
           iElaps, gpu_sum8);

    // GPU Reduction - Unrolling 16
    CHECK(cudaMemcpy(d_idata, h_idata, bytes,
                     cudaMemcpyHostToDevice));

    CHECK(cudaDeviceSynchronize());

    iStart = seconds();

    reduceUnrolling16<<<grid.x / 16, block>>>(
        d_idata, d_odata, size);

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    iElaps = seconds() - iStart;

    CHECK(cudaMemcpy(h_odata, d_odata,
                     (grid.x / 16) * sizeof(long long),
                     cudaMemcpyDeviceToHost));

    for (int i = 0; i < grid.x / 16; i++)
        gpu_sum16 += h_odata[i];

    printf("GPU Unrolling 16   elapsed %f sec  GPU sum: %lld\n",
           iElaps, gpu_sum16);

    // Verify Results
    printf("\nChecking Results...\n");

    if (cpu_sum == gpu_sum8 && cpu_sum == gpu_sum16)
        printf("Test PASSED! All results match.\n");
    else
        printf("Test FAILED! Results do not match.\n");

    // Free host memory
    free(h_idata);
    free(h_odata);
    free(tmp);

    // Free GPU memory
    CHECK(cudaFree(d_idata));
    CHECK(cudaFree(d_odata));

    // Reset GPU
    CHECK(cudaDeviceReset());

    printf("\nExperiment 3 Completed Successfully!\n");

    return EXIT_SUCCESS;
} 
```

## OUTPUT:

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/1f49fbd9-a9da-4ae2-a343-246a6eb4695d" />


## RESULT:
Thus the program has been executed by unrolling by 8 and unrolling by 16. It is observed that _________ has executed with less elapsed time than _____________ with blocks_____,______.
