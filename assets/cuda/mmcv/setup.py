import os
from pkg_resources import DistributionNotFound, get_distribution, parse_version
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='mmcv',
    version='1.0.1',
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
            # extra_compile_args={
            #     'cxx': ['-std=c++17'], 
            #     'nvcc': ['-std=c++17',
            #     '-D__CUDA_NO_HALF_OPERATORS__',
            #     '-D__CUDA_NO_HALF_CONVERSIONS__',
            #     '-D__CUDA_NO_HALF2_OPERATORS__',
            #              ],}
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