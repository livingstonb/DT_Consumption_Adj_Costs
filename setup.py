from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

from Cython.Compiler import Options

Options.buffer_max_dims = 10

extensions = [
				Extension("build.model",["code/model/model.pyx"],
							include_dirs=[np.get_include()])
				]

setup(	name="build",
		ext_modules=cythonize(extensions),
		packages=["build"])