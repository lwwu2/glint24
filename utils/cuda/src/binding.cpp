#include <torch/extension.h>



void project_triangle(
    const torch::Tensor p0, const torch::Tensor p1, const torch::Tensor p2,
    const float c,
    torch::Tensor rets);

void vis_clip(
    const torch::Tensor p0, const torch::Tensor p1, const torch::Tensor p2,
    const float c,
    torch::Tensor rets);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("project_triangle", &project_triangle);
    m.def("vis_clip", &vis_clip);
    
}