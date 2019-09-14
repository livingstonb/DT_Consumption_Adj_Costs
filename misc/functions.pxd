import numpy as np
cimport numpy as np

cdef np.ndarray utilityMat(double riskaver, double[:,:,:,:] con)

cdef np.ndarray utilityVec(double riskaver, double[:] con)

cdef double utility(double riskaver, double con)

cdef np.ndarray marginalUtility(double riskaver, np.ndarray con)

cpdef long[:] searchSortedMultipleInput(double[:] grid, double[:] vals)

cpdef long searchSortedSingleInput(double[:] grid, double val)

cpdef tuple interpolate1D(double[:] grid, double pt)

cpdef double[:] getInterpolationWeights(double[:] grid, double pt, long rightIndex)

cpdef tuple goldenSectionSearch(object f, double a, double b, 
	double goldenRatio, double goldenRatioSq, double tol, tuple args)
