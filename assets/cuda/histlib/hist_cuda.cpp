#include "hist.h"
#include "hist_cuda.h"

at::Tensor
hist(const at::Tensor &X, const at::Tensor &Y,
        const float min_x, const float min_y, const float min_z,
        const float max_x, const float max_y, const float max_z,
        const int len_x, const int len_y, const int len_z,
        const int mini_batch
        )
{

    if (X.type().is_cuda() && Y.type().is_cuda())
    {
        return hist_cuda(X, Y,
                        min_x, min_y, min_z, 
                        max_x, max_y, max_z, 
                        len_x, len_y, len_z,
                        mini_batch
                        );
    }
    AT_ERROR("Not implemented on the CPU");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hist", &hist, "hist");
}
