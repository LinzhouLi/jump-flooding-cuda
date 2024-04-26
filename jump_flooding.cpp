#include <torch/extension.h>
#include "jump_flooding_cuda.h"

torch::Tensor jumpFlooding(const torch::Tensor& input) {
    if (input.ndimension() != 3 || input.size(2) != 1)
		AT_ERROR("input must have dimensions (H, W, 1)");

    int H = input.size(0);
    int W = input.size(1);

    torch::TensorOptions intOpt = input.options().dtype(torch::kInt32);
    torch::TensorOptions floatOpt = input.options().dtype(torch::kFloat32);
    torch::Tensor buffer1 = torch::zeros({H, W, 3}, intOpt);
    torch::Tensor buffer2 = torch::zeros({H, W, 3}, intOpt);
    torch::Tensor result = torch::zeros({H, W, 2}, floatOpt);

    initDataCuda(input.data_ptr<float>(), buffer1.data_ptr<int>(), H, W);
    bool reverse = jumpFloodingCuda(buffer1.data_ptr<int>(), buffer2.data_ptr<int>(), H, W);

    if (reverse) signedDistanceCuda(buffer1.data_ptr<int>(), result.data_ptr<float>(), H, W);
    else signedDistanceCuda(buffer2.data_ptr<int>(), result.data_ptr<float>(), H, W);

    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("jump_flooding", &jumpFlooding);
}