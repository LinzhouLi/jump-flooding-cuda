#pragma once

void initDataCuda(float* input, int* output, int H, int W);
bool jumpFloodingCuda(int* buffer1, int* buffer2, int H, int W);
void signedDistanceCuda(int* buffer, float* output, int H, int W);