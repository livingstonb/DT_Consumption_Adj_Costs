import numpy as np
cimport numpy as np

# structure to hold parameters for objective function
# used by golden section search
cdef struct FnArgs:
	double *cgrid
	double *emaxVec
	double *yderivs
	long error
	long ncValid
	double riskAver
	double timeDiscount
	double deathProb
	long nc

# function pointer for golden section search
ctypedef double (*objectiveFn)(double x, FnArgs fargs) nogil

cdef double utility(double riskaver, double con) nogil

cdef long fastSearchSingleInput(double *grid, double val, long nGrid) nogil

cdef void getInterpolationWeights(
	double *grid, double pt, long nGrid, long *indices, double *weights) nogil

cdef long goldenSectionSearch(objectiveFn f, double a, double b, 
	double tol, double* out, FnArgs args) nogil except -1

cdef double cmax(double *vals, int nVals) nogil

cdef double cmin(double *vals, int nVals) nogil

cdef long cargmax(double *vals, int nVals) nogil

cpdef double gini(double[:] vals)