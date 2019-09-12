from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

from Cython.Compiler import Options

Options.buffer_max_dims = 10

extensions = [
				Extension("model.model",["model/model.pyx"],
							include_dirs=[np.get_include()]),
				Extension("misc.functions",["misc/functions.pyx"],
							include_dirs=[np.get_include()]),
				]

setup(	name="DiscreTime",
		ext_modules=cythonize(extensions),
		packages=["model","misc"])
