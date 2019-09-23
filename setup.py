from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os
import platform

from Cython.Compiler import Options

Options.buffer_max_dims = 10

if (os.environ.get('CC',None)=="gcc-9") or (platform.system()=="Linux"):
	compileArgs = ['-fopenmp']
else:
	compileArgs = []


extensions = [
				Extension("model.cmodel",["model/cmodel.pyx"],
							include_dirs=[np.get_include(),"misc"]),

				Extension("model.modelObjects",["model/modelObjects.pyx"],
							include_dirs=[np.get_include()]),

				Extension("misc.cfunctions",["misc/cfunctions.pyx"],
							include_dirs=[np.get_include()]),

				Extension("misc.spline",["misc/spline.pyx"]),

				Extension("misc.tester",["misc/tester.pyx"],
							include_dirs=[np.get_include()]),

				Extension("model.csimulator",["model/csimulator.pyx"],
							include_dirs=[np.get_include()],
							extra_compile_args=compileArgs,
        					extra_link_args=compileArgs,),
				]

setup(	name="DiscreTime",
		ext_modules=cythonize(extensions),
		packages=["model","misc"])
