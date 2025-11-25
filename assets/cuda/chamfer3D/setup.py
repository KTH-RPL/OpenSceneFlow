from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

extra_compile_args = {
    'cxx': ['-DCCCL_IGNORE_DEPRECATED_CUDA_BELOW_12', '-DTHRUST_IGNORE_CUB_VERSION_CHECK'],
    'nvcc': ['-DCCCL_IGNORE_DEPRECATED_CUDA_BELOW_12', '-DTHRUST_IGNORE_CUB_VERSION_CHECK'],
}
setup(
    name='chamfer3D',
    ext_modules=[
        CUDAExtension(
            name='chamfer3D',
            sources=[
                "/".join(__file__.split('/')[:-1] + ['chamfer3D_cuda.cpp']), # must named as xxx_cuda.cpp
                "/".join(__file__.split('/')[:-1] + ['chamfer3D.cu']),
            ],
            extra_compile_args=extra_compile_args
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    version='1.0.5'
)