
import numpy as np
cimport numpy as np
cimport cython

from libc.math cimport log, fabs, pow

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
cdef long fastSearchSingleInput(double *grid, double val, long nGrid) nogil:
	cdef long lower, upper, midpt = 0
	cdef double valMidpt = 0.0

	if val >= grid[nGrid-1]:
		return nGrid - 1
	elif val <= grid[0]:
		return 1

	lower = -1
	upper = nGrid

	while (upper - lower) > 1:
		midpt = (upper + lower) >> 1
		valMidpt = grid[midpt]

		if val == valMidpt:
			return midpt + 1
		elif val > valMidpt:
			lower = midpt
		else:
			upper = midpt

	if val > valMidpt:
		return midpt + 1
	else:
		return midpt


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void getInterpolationWeights(
	double *grid, double pt, long nGrid, long *indices, double *weights) nogil:
	"""
	This function finds the weights placed on the grid value below pt
	and the grid value above pt when interpolating pt onto grid. Output
	is 'indices' and 'weights'.
	"""
	cdef double weight0

	indices[1] = fastSearchSingleInput(grid, pt, nGrid)
	indices[0] = indices[1] - 1

	weight0 = (grid[indices[1]] - pt) / (grid[indices[1]] - grid[indices[0]])

	if weight0 < 0:
		weights[0] = 0
		weights[1] = 1
	elif weight0 > 1:
		weights[0] = 1
		weights[1] = 0
	else:
		weights[0] = weight0
		weights[1] = 1 - weight0

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double interpolate(double *grid, double pt, double *vals, long nGrid) nogil:
	cdef:
		long indices[2]
		double weights[2]
		double out

	getInterpolationWeights(grid, pt, nGrid, &indices[0], &weights[0])
	out = weights[0] * vals[indices[0]] + weights[1] * vals[indices[1]]

	return out

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double cmax(double[:] vals) nogil:
	"""
	Finds the maximum in 'vals'. Length of input
	must be supplied.
	"""
	cdef long i
	cdef double currentMax, currentVal

	currentMax = vals[0]

	for i in range(1, vals.shape[0]):
		currentVal = vals[i]
		if currentVal > currentMax:
			currentMax = currentVal

	return currentMax

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double cmin(double[:] vals) nogil:
	"""
	Finds the minimum in 'vals'. Length of input
	must be supplied.
	"""
	cdef long i
	cdef double currentMin, currentVal

	currentMin = vals[0]

	for i in range(1, vals.shape[0]):
		currentVal = vals[i]
		if currentVal < currentMin:
			currentMin = currentVal

	return currentMin

@cython.boundscheck(False)
@cython.wraparound(False)
cdef long cargmax(double[:] vals) nogil:
	"""
	Finds the argmax of 'vals'. Length of input
	must be supplied.
	"""
	cdef long i, currentArgMax = 0
	cdef double currentMax, currentVal

	currentMax = vals[0]

	for i in range(1, vals.shape[0]):
		currentVal = vals[i]
		if currentVal > currentMax:
			currentMax = currentVal
			currentArgMax = i

	return currentArgMax

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double gini(double[:] vals):
	cdef:
		double temp, numerator = 0
		double denom = 0
		long i, n = vals.size
		double[:] vals_sorted = np.sort(vals)

	for i in range(n):
		numerator += <double> (i+1) * vals_sorted[i]
		denom += vals_sorted[i]

	temp = <double> n - (numerator/denom)

	return 1.0 - (2.0/(<double>n-1.0)) * temp

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void linspace(double lb, double ub, int num, double[:] out):
	cdef:
		int i
		double spacing

	spacing = (ub - lb) / (num - 1)
	for i in range(num):
		out[i] = lb + spacing * i