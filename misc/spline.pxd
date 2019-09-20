cdef void spline(double *x, double *y, long n, 
	double yp1, double yp2, double *y2) nogil

cdef int splint(double *xa, double *ya, double *y2a, long n,
	double x, double *y) nogil except -1