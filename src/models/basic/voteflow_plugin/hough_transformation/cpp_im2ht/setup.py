from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='im2ht',
    # ext_modules=[
    #     CUDAExtension(name='im2ht', sources=['im2ht.cpp', 'ht_cuda.cu'],
    #     extra_compile_args={'cxx': ['-g'], 'nvcc': ['-arch=sm_60']}),
    # ],
    ext_modules=[
        CUDAExtension(name='im2ht', sources=['im2ht.cpp', 'ht_cuda.cu']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })



# def load_cpp_ext(ext_name):
#     root_dir = os.path.join(os.path.split(__file__)[0])
#     src_dir = os.path.join(root_dir, "cpp")
#     tar_dir = os.path.join(src_dir, "build", ext_name)
#     os.makedirs(tar_dir, exist_ok=True)
#     srcs = glob(f"{src_dir}/*.cu") + glob(f"{src_dir}/*.cpp")

#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
#         from torch.utils.cpp_extension import load

#         ext = load(
#             name=ext_name,
#             sources=srcs,
#             extra_cflags=["-O3"],
#             extra_cuda_cflags=[],
#             build_directory=tar_dir,
#         )
#     return ext