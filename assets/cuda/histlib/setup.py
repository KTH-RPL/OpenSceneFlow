from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='hist',
    ext_modules=[
        CUDAExtension('hist', [
            "/".join(__file__.split('/')[:-1] + ['hist_cuda.cpp']), # must named as xxx_cuda.cpp
            "/".join(__file__.split('/')[:-1] + ['hist.cu']),
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    version='1.0.1')
