from distutils.core import setup, Extension
import numpy
from Cython.Distutils import build_ext

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("SOMDisplay",
                 sources=["_SOMDisplay.pyx", "SOMDisplay.cpp"],
                 include_dirs=[numpy.get_include()],
                 extra_compile_args=['-O2', '-march=native'],
                 extra_link_args=['-O2', '-march=native'],
                 language="c++")],
)
