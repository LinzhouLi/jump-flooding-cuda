from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name = "jump_flooding",
    packages = ['jump_flooding'],
    ext_modules = [
        CUDAExtension(
            name = "jump_flooding._C",
            sources = [
                "jump_flooding_cuda.cu",
                "jump_flooding.cpp"
            ]
        )
    ],
    cmdclass = {
        'build_ext': BuildExtension
    }
)