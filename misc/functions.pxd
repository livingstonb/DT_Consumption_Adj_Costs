import numpy as np
cimport numpy as np

ctypedef double (*objectiveFn)(double x, double[:] y, double[:] z,
								long a1, double a2, double a3, double a4) nogil

cdef np.ndarray utilityMat(double riskaver, double[:,:,:,:] con)

cdef np.ndarray utilityVec(double riskaver, double[:] con)

cdef double utility(double riskaver, double con) nogil

cdef np.ndarray marginalUtility(double riskaver, np.ndarray con)

cpdef long[:] searchSortedMultipleInput(double[:] grid, double[:] vals)

cpdef long searchSortedSingleInput(double[:] grid, double val, long nGrid) nogil

cpdef double[:,:,:] interpolateTransitionProbabilities2D(double[:] grid, double[:,:] vals)

cpdef tuple interpolate1D(double[:] grid, double pt)

cdef void getInterpolationWeights(double[:] grid, double pt, long rightIndex, double *out) nogil

cdef void goldenSectionSearch(objectiveFn f, double a, double b, 
	double invGoldenRatio, double invGoldenRatioSq, double tol, double* out,
	double[:] arg1, double[:] arg2,
	long a1, double a2, double a3, double a4) nogil

cdef double cmax(double *vals, int nVals) nogil

cdef long cargmax(double *vals, int nVals) nogil