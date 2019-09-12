import numpy as np
cimport numpy as np

cdef np.ndarray utility(double riskaver, np.ndarray con)

cdef np.ndarray marginalUtility(double riskaver, np.ndarray con)

cdef tuple interpolate1D(double[:] grid, double pt)

cpdef double goldenSectionSearch(object f, double a, double b, 
	double goldenRatio, double goldenRatioSq, double tol, tuple args)