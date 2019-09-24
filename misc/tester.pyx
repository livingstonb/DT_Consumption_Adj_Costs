from misc cimport cfunctions
from misc.cfunctions cimport FnArgs, objectiveFn
from misc cimport spline
import numpy as np
cimport numpy as np

cimport cython
from libc.stdlib cimport malloc, free

from matplotlib import pyplot as plt

cdef double myfun(double x, FnArgs fargs):
	return - x ** 2

def testGSS():
	cdef double out[2]
	cdef FnArgs fargs

	objective = <objectiveFn> myfun

	cfunctions.goldenSectionSearch(objective, -0.5, 0.5, 
		1e-8, &out[0], fargs)

	print(out)

	# cgrid = np.array([0.001,0.05,1,2])
	# for cval in [-0.1,0.001,0.05,1.2,3]:
	# 	ind2 = cfunctions.searchSortedSingleInput(cgrid, cval)
	# 	ind1 = ind2 - 1
	# 	weights = cfunctions.getInterpolationWeights(cgrid, cval, ind2)
	# 	print(f'cval is {cval}')
	# 	print(f'index is {ind2}')
	# 	print(np.asarray(weights))

	# cvals = np.array([-0.1,0.001,0.05,1.2,3])
	# print(np.asarray(cfunctions.searchSortedMultipleInput(cgrid,cvals)))

def testInterpolation():
	cdef double weights[2]
	cdef long indices[2]
	cdef double[:] grid

	grid = np.array([1,2,3.0])
	randNums = np.random.random(100) * 5 - 1

	for num in randNums:
		cfunctions.getInterpolationWeights(&grid[0],num,3,&indices[0],&weights[0])
		print(f'Random number = {num}')
		print(f'    indices = {indices}')
		print(f'    weights = {weights}')

# def testCubicInterp():
# 	cdef double[:] yderivs
# 	cdef long nVals
# 	cdef double[:] grid
# 	cdef double[:] values

# 	nVals = 10
# 	nPts = 20

# 	yderivs = np.empty(nVals)

# 	grid = np.linspace(0,5,num=nVals)
# 	values = np.sort(np.random.random(nVals)) * 100

# 	xvals = np.random.random(nPts) * 5

# 	spline.spline(&grid[0], &values[0], nVals, 1.0e30, 1.0e30, &yderivs[0])

# 	for i in range(nPts):
# 		predicted = spline.splint(&grid[0], &values[0], &yderivs[0], nVals, xvals[i])
# 		print(f'At point {xvals[i]}, predicted value = {predicted}')

# 	plt.plot(grid,values)
# 	plt.show()

@cython.boundscheck(False)
@cython.wraparound(False)
def testFastSearch(double[:] draws):
	cdef long i, nGrid
	cdef double[:] grid
	cdef long[:] indices

	indices = np.zeros(draws.size,dtype=int)

	nGrid = 800
	grid = np.linspace(0,5,num=nGrid)

	for i in range(draws.size):
		indices[i] = cfunctions.fastSearchSingleInput(&grid[0], draws[i], nGrid)

	return indices

@cython.boundscheck(False)
@cython.wraparound(False)
def testNaiveSearch(double[:] draws):
	cdef long i, nGrid
	cdef double[:] grid
	cdef long[:] indices

	indices = np.zeros(draws.size,dtype=int)

	nGrid = 800
	grid = np.linspace(0,5,num=nGrid)

	for i in range(draws.size):
		indices[i] = cfunctions.searchSortedSingleInput(grid, draws[i], nGrid)

	return indices

def checkSearch(runs):
	cdef long[:] indicesFast
	cdef long[:] indicesNaive
	draws = np.random.random(runs) * 5

	indicesFast = testFastSearch(draws)
	indicesNaive = testNaiveSearch(draws)

	nGrid = 800
	grid = np.linspace(0,5,num=nGrid)

	for i in range(runs):
		if not (indicesFast[i] == indicesNaive[i]):
			print(f'index from fast algorithm = {indicesFast[i]}')
			print(f'index from slow algorithm = {indicesNaive[i]}')

			print(f'grid value at {indicesFast[i]} = {grid[indicesFast[i]]}')
			print(f'grid value at {indicesNaive[i]} = {grid[indicesNaive[i]]}')
			print(f'draw value = {draws[i]}')