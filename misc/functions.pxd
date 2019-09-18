import numpy as np
cimport numpy as np

# structure to hold parameters for objective function
# used by golden section search
cdef struct FnParameters:
	long nx
	long nc
	long nz
	double cMin
	double cMax
	double riskAver
	double timeDiscount
	double deathProb
	bint cubicInterp

# function pointer for golden section search
ctypedef double (*objectiveFn)(double x, double[:] y, double[:] z, 
	FnParameters fparams) nogil

cpdef np.ndarray utilityMat(double riskaver, double[:,:,:,:] con)

cdef double utility(double riskaver, double con) nogil

cpdef long[:] searchSortedMultipleInput(double[:] grid, double[:] vals)

cdef long fastSearchSingleInput(double[:] grid, double val, long nGrid) nogil

cpdef long searchSortedSingleInput(double[:] grid, double val, long nGrid) nogil

cpdef double[:,:,:] interpolateTransitionProbabilities2D(double[:] grid, double[:,:] vals)

cdef void getInterpolationWeights(
	double[:] grid, double pt, long nGrid, long *indices, double *weights) nogil

cdef void goldenSectionSearch(objectiveFn f, double a, double b, 
	double tol, double* out, double[:] arg1, double[:] arg2, 
	FnParameters fparams) nogil

cdef double cmax(double *vals, int nVals) nogil

cdef double cmin(double *vals, int nVals) nogil

cdef long cargmax(double *vals, int nVals) nogil