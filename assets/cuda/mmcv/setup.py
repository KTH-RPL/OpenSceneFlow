import os
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

extra_compile_args = {
    'cxx': ['-DCCCL_IGNORE_DEPRECATED_CUDA_BELOW_12'],
    'nvcc': ['-DCCCL_IGNORE_DEPRECATED_CUDA_BELOW_12'],
}


setup(
    name='mmcv',
    version='1.0.2',
    ext_modules=[
        CUDAExtension(
            name='mmcv',
            sources=[
                "/".join(__file__.split("/")[:-1] + ["scatter_points_cuda.cu"]),
                "/".join(__file__.split("/")[:-1] + ["scatter_points.cpp"]),
                "/".join(__file__.split("/")[:-1] + ["voxelization_cuda.cu"]),
                "/".join(__file__.split("/")[:-1] + ["voxelization.cpp"]),
                "/".join(__file__.split("/")[:-1] + ["cudabind.cpp"]),
                "/".join(__file__.split("/")[:-1] + ["pybind.cpp"]),

            ],
            extra_compile_args=extra_compile_args
        ),
    ],
    cmdclass={'build_ext': BuildExtension},

    # no change below
    description='OpenMMLab Computer Vision Foundation',
    keywords='computer vision',
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Utilities',
    ],
    url='https://github.com/open-mmlab/mmcv',
    author='MMCV Contributors',
    author_email='openmmlab@gmail.com',
    python_requires='>=3.7',
    zip_safe=False)