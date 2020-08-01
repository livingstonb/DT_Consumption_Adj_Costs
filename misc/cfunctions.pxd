import numpy as np
cimport numpy as np

# function pointer for golden section search
ctypedef double (*objectiveFn)(double x)

cdef double utility(double riskaver, double con) nogil

cdef long fastSearchSingleInput(double *grid, double val, long nGrid) nogil

cdef void getInterpolationWeights(
	double *grid, double pt, long nGrid, long *indices, double *weights) nogil

cdef double interpolate(double *grid, double pt, double *vals, long nGrid) nogil

cdef double cmax(double[:] vals) nogil

cdef double cmin(double[:] vals) nogil

cdef long cargmax(double[:] vals) nogil

cpdef double gini(double[:] vals)

cpdef void linspace(double lb, double ub, int num, double[:] out)