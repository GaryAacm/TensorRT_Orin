#include "MyCustomPlugin.h"

// 这个 kernel 会并行地将两个向量相加
__global__ void cudaAdd(float* output, const float* input2, int numElements) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) 
    {
        output[idx] += input2[idx];  // 执行加法操作
    }
}

