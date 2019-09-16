from misc cimport functions
from misc cimport spline
import numpy as np
cimport numpy as np

from libc.stdlib cimport malloc, free

from matplotlib import pyplot as plt

def testGSS():
	pass
	# gr = (np.sqrt(5) + 1) / 2
	# grsq = gr ** 2

	# fn = lambda x: x ** 3 - x

	# print(functions.goldenSectionSearch(fn, -1, 0.5, 
	# 	gr, grsq, 1e-5, tuple()))

	# cgrid = np.array([0.001,0.05,1,2])
	# for cval in [-0.1,0.001,0.05,1.2,3]:
	# 	ind2 = functions.searchSortedSingleInput(cgrid, cval)
	# 	ind1 = ind2 - 1
	# 	weights = functions.getInterpolationWeights(cgrid, cval, ind2)
	# 	print(f'cval is {cval}')
	# 	print(f'index is {ind2}')
	# 	print(np.asarray(weights))

	# cvals = np.array([-0.1,0.001,0.05,1.2,3])
	# print(np.asarray(functions.searchSortedMultipleInput(cgrid,cvals)))

def testInterpolation():
	cdef double weights[2]
	grid = np.array([1,2,3.0])
	randNums = np.random.random(100) * 5 - 1

	for num in randNums:
		index = functions.searchSortedSingleInput(grid,num,grid.size)
		functions.getInterpolationWeights(grid,num,index,&weights[0])
		print(f'Random number = {num}')
		print(f'    index = {index}')
		print(f'    weights = {weights}')

def testCubicInterp():
	cdef double[:] yderivs
	cdef long nVals
	cdef double[:] grid
	cdef double[:] values

	nVals = 10
	nPts = 20

	yderivs = np.empty(nVals)

	grid = np.linspace(0,5,num=nVals)
	values = np.sort(np.random.random(nVals)) * 100

	xvals = np.random.random(nPts) * 5

	spline.spline(&grid[0], &values[0], nVals, 1.0e30, 1.0e30, &yderivs[0])

	for i in range(nPts):
		predicted = spline.splint(&grid[0], &values[0], &yderivs[0], nVals, xvals[i])
		print(f'At point {xvals[i]}, predicted value = {predicted}')

	plt.plot(grid,values)
	plt.show()
