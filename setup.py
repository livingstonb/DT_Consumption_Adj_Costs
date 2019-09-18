from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np
import platform

from Cython.Compiler import Options

Options.buffer_max_dims = 10

if platform.system() == "Linux":
	compileArgs = ['-fopenmp']
else:
	compileArgs = []


extensions = [
				Extension("model.cmodel",["model/cmodel.pyx"],
							include_dirs=[np.get_include(),"misc"],
							extra_compile_args=compileArgs,
        					extra_link_args=compileArgs,),

				Extension("model.modelObjects",["model/modelObjects.pyx"],
							include_dirs=[np.get_include()]),

				Extension("misc.functions",["misc/functions.pyx"],
							include_dirs=[np.get_include()]),

				Extension("misc.spline",["misc/spline.pyx"]),

				Extension("model.csimulator",["model/csimulator.pyx"],
							include_dirs=[np.get_include()],
							extra_compile_args=compileArgs,
        					extra_link_args=compileArgs,),

				Extension("misc.tester",["misc/tester.pyx"],
							include_dirs=[np.get_include()]),
				]

setup(	name="DiscreTime",
		ext_modules=cythonize(extensions),
		packages=["model","misc"])
