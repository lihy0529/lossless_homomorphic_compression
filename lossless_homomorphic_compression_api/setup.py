from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='api',
    include_dirs=["include"],
    ext_modules=[
        CUDAExtension(
            'api',
            ['kernel/api.cpp', 'kernel/api_kernel.cu'],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)