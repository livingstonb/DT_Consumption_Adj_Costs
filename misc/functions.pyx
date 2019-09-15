import numpy as np
cimport numpy as np
cimport cython

from libc.math cimport log, fabs, pow

cdef np.ndarray utilityMat(double riskaver, double[:,:,:,:] con):
	if riskaver == float(1):
		return np.log(con)
	else:
		return np.power(con,1-riskaver) / (1-riskaver)

cdef np.ndarray utilityVec(double riskaver, double[:] con):
	if riskaver == float(1):
		return np.log(con)
	else:
		return np.power(con,1-riskaver) / (1-riskaver)

cdef double utility(double riskaver, double con):
	if riskaver == float(1):
		return log(con)
	else:
		return pow(con,1-riskaver) / (1-riskaver)

cdef np.ndarray marginalUtility(double riskaver, np.ndarray con):
	"""
	Returns the first derivative of the utility function
	"""
	u = np.power(con,-riskaver)
	return u

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef long[:] searchSortedMultipleInput(double[:] grid, double[:] vals):
	"""
	This function finds the index i for which
	grid[i-1] <= val < grid[i]
	"""
	cdef long n, m, i, index
	cdef double currentVal
	cdef long[:] indices

	n = grid.size
	m = vals.size

	indices = np.zeros((m),dtype=int)

	for i in range(m):
		index = 1
		currentVal = vals[i]

		while index < n - 1:
			if currentVal >= grid[index]:
				index += 1
			else:
				break

		indices[i] = index

	return indices

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef long searchSortedSingleInput(double[:] grid, double val):
	"""
	This function finds the index i for which
	grid[i-1] <= val < grid[i]
	"""
	cdef long n, index = 1

	n = grid.size

	while index < n - 1:
		if val >= grid[index]:
			index += 1
		else:
			break

	return index

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:,:,:] interpolateTransitionProbabilities2D(double[:] grid, double[:,:] vals):
	cdef:
		long[:,:] gridIndices
		double[:,:,:] probabilities
		long i, j, igrid
		double Pi, gridPt1, gridPt2

	gridIndices = np.zeros((vals.shape[0],vals.shape[1]),dtype=int)
	probabilities = np.zeros((vals.shape[0],vals.shape[1],grid.shape[0]))

	nGrid = grid.shape[0]

	for j in range(vals.shape[1]):
		gridIndices[:,j] = searchSortedMultipleInput(grid,vals[:,j])

		for i in range(vals.shape[0]):
			igrid = gridIndices[i,j]

			gridPt1 = grid[igrid-1]
			gridPt2 = grid[igrid]
			Pi = (vals[i,j] - gridPt1) / (gridPt2 - gridPt1)

			if Pi < 0:
				probabilities[i,j,igrid-1] = 1
				probabilities[i,j,igrid] = 0
			elif Pi > 1:
				probabilities[i,j,igrid-1] = 0
				probabilities[i,j,igrid] = 1
			else:
				probabilities[i,j,igrid-1] = 1 - Pi
				probabilities[i,j,igrid] = Pi

	return probabilities

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple interpolate1D(double[:] grid, double pt):
	cdef:
		int gridIndex
		double gridPt1, gridPt2, proportion2
		list gridIndices, proportions

	# returns the two indices between which to interpolate
	# returns the proportions to place on each of the 2 indices
	gridIndex = searchSortedSingleInput(grid, pt)

	if gridIndex == 0:
		gridIndices = [0,1]
		proportions = [1,0]
	elif gridIndex == np.size(grid):
		gridIndices = [grid.shape[0]-2,grid.shape[0]-1]
		proportions = [0,1]
	else:
		gridIndices = [gridIndex-1,gridIndex]
		gridPt1 = grid[gridIndices[0]]
		gridPt2 = grid[gridIndices[1]]
		proportion2 = (pt - gridPt1) / (gridPt2 - gridPt1)
		proportions = [1-proportion2,proportion2]

	return (gridIndices, proportions)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:] getInterpolationWeights(double[:] grid, double pt, long rightIndex):
	cdef double[:] weights
	cdef double weight1

	weights = np.zeros((2))
	weight1 = (grid[rightIndex] - pt) / (grid[rightIndex] - grid[rightIndex-1])

	if weight1 < 0:
		weights[0] = 0
		weights[1] = 1
	elif weight1 > 1:
		weights[0] = 1
		weights[1] = 0
	else:
		weights[0] = weight1
		weights[1] = 1 - weight1

	return weights

cpdef tuple goldenSectionSearch(object f, double a, double b, 
	double invGoldenRatio, double invGoldenRatioSq, double tol):

	cdef double c, d, diff
	cdef double fc, fd

	diff = b - a

	c = a + diff * invGoldenRatioSq
	d = a + diff * invGoldenRatio 

	fc = f(c)
	fd = f(d)

	while fabs(c - d) > tol:
		if fc > fd:
			b = d
			d = c
			fd = fc
			diff = diff * invGoldenRatio
			c = a + diff * invGoldenRatioSq
			fc = f(c)
		else:
			a = c
			c = d
			fc = fd
			diff = diff * invGoldenRatio
			d = a + diff * invGoldenRatio
			fd = f(d)

	if fc > fd:
		return fc, c #(a + d) / 2
	else:
		return fd, d #(c + b) / 2