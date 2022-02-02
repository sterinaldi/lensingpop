import numpy
from setuptools import setup, find_packages
from setuptools import Extension
from setuptools.command.build_ext import build_ext as _build_ext
from codecs import open
from os import path
from distutils.extension import Extension
import os
try:
    from Cython.Build import cythonize
except:
    print('Cython not found. Please install it via\n\tpip install Cython')
    exit()

# see https://stackoverflow.com/a/21621689/1862861 for why this is here
class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        self.include_dirs.append(numpy.get_include())

ext_modules=[
             Extension("lensingpop.models",
                       sources=[os.path.join("lensingpop","models.pyx")],
                       libraries=["m"], # Unix-like specific
                       extra_compile_args=["-O3","-ffast-math"],
                       include_dirs=['lensingpop', numpy.get_include()]
                       )
             ]
             
ext_modules = cythonize(ext_modules)
setup(
      name = 'lensingpop/models',
      ext_modules = cythonize(ext_modules, language_level = "3"),
      include_dirs=[numpy.get_include()]
      )

setup(
    name = 'lensingpop',
    use_scm_version=True,
    description = 'Testing population assumptions in lensing',
    url = 'https://github.com/sterinaldi/lensingpop',
    python_requires = '>=3.7',
    packages = ['lensingpop'],
    include_dirs = [numpy.get_include()],
    setup_requires=['numpy', 'cython', 'setuptools_scm'],
    entry_points={},
    package_data={"": ['*.c', '*.pyx', '*.pxd']},
    ext_modules=ext_modules,
    )
