from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

import os

def find_cuda():
    cuda_path = os.environ.get('CUDA_PATH') or os.environ.get('CUDA_HOME')
    if cuda_path and os.path.exists(cuda_path):
        return cuda_path
    # Check common locations
    for path in ['C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4', 
                'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1',
                'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8']:
        if os.path.exists(path):
            return path
    return None

cuda_path = find_cuda()
extra_compile_args = []
include_dirs = [get_pybind_include(), get_pybind_include(user=True), "csrc"]
libraries = []
library_dirs = []

if cuda_path:
    print(f"CUDA found at {cuda_path}")
    extra_compile_args.append('/DUSE_CUDA')
    include_dirs.append(os.path.join(cuda_path, 'include'))
    library_dirs.append(os.path.join(cuda_path, 'lib', 'x64'))
    libraries.append('cuda')
    libraries.append('cudart')

ext_modules = [
    Extension(
        '_phoenix_backend',
        [
            'csrc/bindings.cpp',
            'csrc/core/tensor_data.cpp',
            'csrc/core/dispatcher.cpp',
            'csrc/cpu/cpu_backend.cpp',
            'csrc/cpu/math_cpu.cpp',
            'csrc/cuda/cuda_backend.cpp'
        ],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=extra_compile_args,
        language='c++'
    ),
]

def has_flag(compiler, flagname):
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True

def cpp_flag(compiler):
    flags = ['-std=c++17', '-std=c++14', '-std=c++11']
    for flag in flags:
        if has_flag(compiler, flag): return flag
    raise RuntimeError('Unsupported compiler -- at least C++11 support is needed!')

class BuildExt(build_ext):
    c_opts = {
        'msvc': ['/EHsc', '/std:c++17', '/O2'],
        'unix': ['-O3', '-Wall'],
    }
    l_opts = {
        'msvc': [],
        'unix': [],
    }

    if sys.platform == 'darwin':
        darwin_opts = ['-stdlib=libc++', '-mmacosx-version-min=10.14']
        c_opts['unix'] += darwin_opts
        l_opts['unix'] += darwin_opts

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args += opts
            ext.extra_link_args += link_opts
        build_ext.build_extensions(self)

setup(
    name='phoenix_engine',
    version='0.0.1',
    author='The_Last_King',
    description='Low-level C++ backend for Phoenix AI',
    ext_modules=ext_modules,
    setup_requires=['pybind11>=2.5.0'],
    install_requires=['pybind11>=2.5.0'],
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
    packages=['phoenix_engine'],
)

