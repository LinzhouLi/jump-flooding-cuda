#include <limits>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BLOCK_X 16
#define BLOCK_Y 16


__global__ void initKernel(float* input, int3* output, int2 size) {
    int x = int(blockIdx.x * blockDim.x + threadIdx.x);
	int y = int(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= size.x || y >= size.y) return;

    int pixelIdx = x * size.y + y;
    float inputValue = input[pixelIdx];
    if (inputValue < 0.0f) output[pixelIdx] = make_int3(x, y, 1); // interior
    else output[pixelIdx] = make_int3(-1, -1, -1);
}


__global__ void jumpFloodingKernel(int3* input, int3* output, int step, int2 size) {
	int x = int(blockIdx.x * blockDim.x + threadIdx.x);
	int y = int(blockIdx.y * blockDim.y + threadIdx.y);
	if (x >= size.x || y >= size.y) return;

    int selfIdx = x * size.y + y;
    int3 inputData = input[selfIdx];
    if (inputData.z == 1) { // interior
        output[selfIdx] = inputData;
        return;
    }

    int2 nearestCoord = make_int2(inputData.x, inputData.y);
    float minSquareDist = std::numeric_limits<float>::max();
    if (nearestCoord.x != -1 && nearestCoord.y != -1) {
        float offsetX = float(nearestCoord.x - x);
        float offsetY = float(nearestCoord.y - y);
        minSquareDist = offsetX * offsetX + offsetY * offsetY;
    }

    for (int gridY = -1; gridY < 2; ++gridY) {
        for (int gridX = -1; gridX < 2; ++gridX) {
            if (gridX == 0 && gridY == 0) continue; // skip self

            int xLookup = x + gridX * step;
            int yLookup = y + gridY * step;

            if (xLookup < 0 || xLookup >= size.x || yLookup < 0 || yLookup >= size.y) continue; // out of bounds

            int lookupIdx = xLookup * size.y + yLookup;
            inputData = input[lookupIdx];
            int2 coord = make_int2(inputData.x, inputData.y);

            if (coord.x != -1 && coord.y != -1) {
                float offsetX = float(coord.x - x);
                float offsetY = float(coord.y - y);
                float squareDist = offsetX * offsetX + offsetY * offsetY;
                if (squareDist < minSquareDist) {
                    nearestCoord = coord;
                    minSquareDist = squareDist;
                }
            }
        }
    }

    output[selfIdx] = make_int3(nearestCoord.x, nearestCoord.y, -1);
}


__global__ void signedDistanceKernel(int3* input, float2* output, int2 size) {
    int x = int(blockIdx.x * blockDim.x + threadIdx.x);
	int y = int(blockIdx.y * blockDim.y + threadIdx.y);
	if (x >= size.x || y >= size.y) return;

    int idx = x * size.y + y;
    int3 inputData = input[idx];
    if (inputData.z == 1) output[idx] = make_float2(0.0f, 0.0f);
    else {
        output[idx] =  make_float2(
            float(inputData.x - x),
            float(inputData.y - y)
        );
    }
}


void initDataCuda(float* input, int* output, int H, int W) {
    dim3 grid((H + BLOCK_X - 1) / BLOCK_X, (W + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);
    int2 size = make_int2(H, W);
    initKernel <<< grid, block >>> (input, (int3*)output, size);
}


bool jumpFloodingCuda(int* buffer1, int* buffer2, int H, int W) {
    dim3 grid((H + BLOCK_X - 1) / BLOCK_X, (W + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

    bool reverse = true;
    int step = max(H, W);
    int2 size = make_int2(H, W);

    while (step > 1) {
        reverse = !reverse;
        step = (step + 1) >> 1;

        if (reverse) jumpFloodingKernel <<< grid, block >>> ((int3*)buffer2, (int3*)buffer1, step, size);
        else jumpFloodingKernel <<< grid, block >>> ((int3*)buffer1, (int3*)buffer2, step, size);
    }
    return reverse;
}


void signedDistanceCuda(int* buffer, float* output, int H, int W) {
    dim3 grid((H + BLOCK_X - 1) / BLOCK_X, (W + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);
    int2 size = make_int2(H, W);
    signedDistanceKernel <<< grid, block >>> ((int3*)buffer, (float2*)output, size);
}