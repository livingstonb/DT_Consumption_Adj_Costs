import numpy as np
cimport numpy as np

cdef np.ndarray utilityMat(double riskaver, double[:,:,:,:] con)

cdef np.ndarray utilityVec(double riskaver, double[:] con)

cdef double utility(double riskaver, double con)

cdef np.ndarray marginalUtility(double riskaver, np.ndarray con)

cpdef long[:] searchSortedMultipleInput(double[:] grid, double[:] vals)

cpdef long searchSortedSingleInput(double[:] grid, double val, long nGrid) nogil

cpdef double[:,:,:] interpolateTransitionProbabilities2D(double[:] grid, double[:,:] vals)

cpdef tuple interpolate1D(double[:] grid, double pt)

cdef void getInterpolationWeights(double[:] grid, double pt, long rightIndex, double *out) nogil

cdef void goldenSectionSearch(object f, double a, double b, 
	double invGoldenRatio, double invGoldenRatioSq, double tol, double* out) nogil

cdef double cmax(double *vals, int nVals) nogil

cdef double cargmax(double *vals, int nVals) nogil