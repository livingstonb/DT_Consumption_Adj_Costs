cdef void spline(double *x, double *y, long n, 
	double yp1, double yp2, double *y2) nogil

cdef double splint(double *xa, double *ya, double *y2a, long n,
	double x) nogil