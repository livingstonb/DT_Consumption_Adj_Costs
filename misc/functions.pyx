import numpy as np
cimport numpy as np
cimport cython

from libc.math cimport log, fabs, pow

cdef double INV_GOLDEN_RATIO = 0.61803398874989479150
cdef double INV_GOLDEN_RATIO_SQ = 0.38196601125010509747


cdef np.ndarray utilityMat(double riskaver, double[:,:,:,:] con):
	"""
	Utility function for a 4D array.
	"""
	if riskaver == 1.0:
		return np.log(con)
	else:
		return np.power(con,1.0-riskaver) / (1.0-riskaver)

cdef inline double utility(double riskaver, double con) nogil:
	"""
	Utility function for a single value.
	"""
	if riskaver == 1.0:
		return log(con)
	else:
		return pow(con,1.0-riskaver) / (1.0-riskaver)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef long[:] searchSortedMultipleInput(double[:] grid, double[:] vals):
	"""
	For each value in vals, this function finds the index i for which
	grid[i-1] <= value < grid[i], with the exception that i = 0 is 
	taken to be i = 1 (i.e. 0 is never an output). Each element i
	of the output can be used as the index of the second value for
	interpolation.
	"""
	cdef long n, m, i, index
	cdef double currentVal
	cdef long[:] indices

	n = grid.size
	m = vals.size

	indices = np.empty((m),dtype=int)

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
cpdef long searchSortedSingleInput(double[:] grid, double val, long nGrid) nogil:
	"""
	For the value val, this function finds the index i for which
	grid[i-1] <= val < grid[i], with the exception that i = 0 is 
	taken to be i = 1 (i.e. 0 is never an output). The output can
	be used as the index of the second value for
	interpolation.
	"""
	cdef long index = 1

	while index < nGrid - 1:
		if val >= grid[index]:
			index += 1
		else:
			break

	return index

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:,:,:] interpolateTransitionProbabilities2D(double[:] grid, double[:,:] vals):
	"""
	This function interpolates vals onto grid. If grid is of dimension N and vals is
	of dimension (M,L), output is dimension (M,L,N). Element (m,l,n) of the output
	is the weight that should be placed on grid point n for element (m,l) of vals.
	No extrapolation is performed; for any input that is outside the bounds of grid,
	the interpolated value is taken to be either the lowest or the highest value of
	grid.
	"""
	cdef:
		long[:] gridIndices
		double[:,:,:] probabilities
		long i, j, igrid1, igrid2
		double Pi, gridPt1, gridPt2

	probabilities = np.zeros((vals.shape[0],vals.shape[1],grid.shape[0]))

	nGrid = grid.shape[0]

	for j in range(vals.shape[1]):
		gridIndices = searchSortedMultipleInput(grid,vals[:,j])

		for i in range(vals.shape[0]):
			igrid2 = gridIndices[i]
			igrid1 = igrid2 - 1

			gridPt1 = grid[igrid1]
			gridPt2 = grid[igrid2]
			Pi = (vals[i,j] - gridPt1) / (gridPt2 - gridPt1)

			if Pi < 0:
				probabilities[i,j,igrid1] = 1
				probabilities[i,j,igrid2] = 0
			elif Pi > 1:
				probabilities[i,j,igrid1] = 0
				probabilities[i,j,igrid2] = 1
			else:
				probabilities[i,j,igrid1] = 1 - Pi
				probabilities[i,j,igrid2] = Pi

	return probabilities

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void getInterpolationWeights(
	double[:] grid, double pt, long rightIndex, double *out) nogil:
	"""
	This function finds the weights placed on the grid value below pt
	and the grid value above pt when interpolating pt onto grid.

	rightIndex is the index in grid of the grid value above or equal to
	pt, and out is a pointer to a double array of length 2 where the
	weights will be stored.
	"""
	cdef double weight1

	weight1 = (grid[rightIndex] - pt) / (grid[rightIndex] - grid[rightIndex-1])

	if weight1 < 0:
		out[0] = 0
		out[1] = 1
	elif weight1 > 1:
		out[0] = 1
		out[1] = 0
	else:
		out[0] = weight1
		out[1] = 1 - weight1

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void goldenSectionSearch(objectiveFn f, double a, double b, 
	double tol, double* out, double[:] arg1, double[:] arg2,
	double *arg3, FnParameters fparams) nogil:
	"""
	This function iterates over the objective function f using
	the golden section search method in the interval (a,b).

	The maximum function value is supplied to out[0] and the
	maximizer is supplied to out[1]. Arguments of f must be arg1,
	arg2, and fparams.

	Algorithm taken from Wikipedia.
	"""
	cdef double c, d, diff
	cdef double fc, fd

	diff = b - a

	c = a + diff * INV_GOLDEN_RATIO_SQ
	d = a + diff * INV_GOLDEN_RATIO 

	fc = f(c,arg1,arg2,arg3,fparams)
	fd = f(d,arg1,arg2,arg3,fparams)

	while fabs(c - d) > tol:
		if fc > fd:
			b = d
			d = c
			fd = fc
			diff = diff * INV_GOLDEN_RATIO
			c = a + diff * INV_GOLDEN_RATIO_SQ
			fc = f(c,arg1,arg2,arg3,fparams)
		else:
			a = c
			c = d
			fc = fd
			diff = diff * INV_GOLDEN_RATIO
			d = a + diff * INV_GOLDEN_RATIO
			fd = f(d,arg1,arg2,arg3,fparams)

	if fc > fd:
		out[0] = fc
		out[1] = c
	else:
		out[0] = fd
		out[1] = d

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double cmax(double *vals, int nVals) nogil:
	"""
	Finds the maximum in 'vals'. Length of input
	must be supplied.
	"""
	cdef long i
	cdef double currentMax, currentVal

	currentMax = vals[0]

	for i in range(1,nVals):
		currentVal = vals[i]
		if currentVal > currentMax:
			currentMax = currentVal

	return currentMax

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double cmin(double *vals, int nVals) nogil:
	"""
	Finds the minimum in 'vals'. Length of input
	must be supplied.
	"""
	cdef long i
	cdef double currentMin, currentVal

	currentMin = vals[0]

	for i in range(1,nVals):
		currentVal = vals[i]
		if currentVal < currentMin:
			currentMin = currentVal

	return currentMin

@cython.boundscheck(False)
@cython.wraparound(False)
cdef long cargmax(double *vals, int nVals) nogil:
	"""
	Finds the argmax of 'vals'. Length of input
	must be supplied.
	"""
	cdef long i, currentArgMax = 0
	cdef double currentMax, currentVal

	currentMax = vals[0]

	for i in range(1,nVals):
		currentVal = vals[i]
		if currentVal > currentMax:
			currentMax = currentVal
			currentArgMax = i

	return currentArgMax