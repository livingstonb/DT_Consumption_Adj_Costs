import numpy as np
cimport numpy as np

cdef double utility(double riskaver, double con) nogil

cdef long fastSearchSingleInput(double *grid, double val, long nGrid) nogil

cdef double getInterpolationWeight(
	double *grid, double pt, long nGrid, long *indices) nogil

cdef double interpolate(double *grid, double pt, double *vals, long nGrid) nogil

cdef long cargmax(double[:] vals) nogil

cpdef double gini(double[:] vals)

cpdef void linspace(double lb, double ub, int num, double[:] out)