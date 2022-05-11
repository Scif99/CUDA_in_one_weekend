#include <iostream>
#include <cmath>


//Specify a function that runs on the GPu which can be called from CPU
//Kernel is executed for every thread
// Each thread does a single computation
__global__
void add(int n, float* x, float* y, float* out)
{
    int block_index = blockIdx.x; //Index of block in grid
    int thread_index = threadIdx.x; //Note that each block has the same indices for threads

    int curr = block_index*blockDim.x + thread_index; // Get the correct thread index over the whole grid
    int stride = blockDim.x*gridDim.x; //Note that the stride is the grid size
    for(int i = curr;i<n;i+=stride)
    {
        out[i] = x[i] + y[i];
    }
}

int main(void)
{
    int N = 1<<20; //1<<10 = 2^10 ~ 1000

    float* x;
    float *y;
    float* o;

    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));
    cudaMallocManaged(&o, N*sizeof(float));

    for(int i =0;i<N;++i)
    {
        x[i] = 1.f;
        y[i] = 2.f;
        o[i] = 0.f;
    }

    int blockSize = 256; //Threads per block
    int numBlocks = (N + blockSize -1 / blockSize); //Blocks in the kernel grid
    add<<<numBlocks,blockSize>>>(N,x,y,o);

    //Like calling join on threads?
    cudaDeviceSynchronize();

      // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i]-3.0f));
    std::cout << "Max error: " << maxError << std::endl;
    
    cudaFree(x);
    cudaFree(y);

    return 0;

}