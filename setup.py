from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

from Cython.Compiler import Options

Options.buffer_max_dims = 10

extensions = [
				Extension("model.model",["model/model.pyx"],
							include_dirs=[np.get_include()]),
				Extension("model.modelObjects",["model/modelObjects.pyx"],
							include_dirs=[np.get_include()]),
				Extension("misc.functions",["misc/functions.pyx"],
							include_dirs=[np.get_include()]),
				Extension("model.simulator",["model/simulator.pyx"],
							include_dirs=[np.get_include()],
							extra_compile_args=['-fopenmp'],
        					extra_link_args=['-fopenmp'],),
				]

setup(	name="DiscreTime",
		ext_modules=cythonize(extensions),
		packages=["model","misc"])
