from libc.stdlib cimport malloc, free
cimport cython

# Algorithm taken from Numerical Recipes in C, 2nd Edition.

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void spline(double *x, double *y, long n, 
	double yp1, double ypn, double *y2) nogil:
	"""
	Given a grid x with function values y, both of length n,
	this function approximates the second derivatives of the
	function, which are output to y2.

	Inputs yp1 and ypn are the first derivatives at the first
	point and the last point. If these are set to 1e30 or above,
	the second derivatives are assumed to be zero at these points.
	"""
	cdef:
		long i, k
		double p, qn, sig, un
		double *u

	u = <double *> malloc((n-1) * sizeof(double))

	if yp1 > 0.99e30:
		y2[0] = 0.0
		u[0] = 0.0
	else:
		y2[0] = -0.5
		u[0] = (3.0/(x[1]-x[0])) * ((y[1]-y[0])/(x[1]-x[0])-yp1)

	for i in range(1,n-1):
		sig = (x[i]-x[i-1]) / (x[i+1]-x[i-1])
		p = sig * y2[i-1] + 2.0
		y2[i] = (sig-1.0) / p
		u[i] = (y[i+1]-y[i]) / (x[i+1]-x[i]) - (y[i]-y[i-1])/(x[i]-x[i-1])
		u[i] = (6.0 * u[i] / (x[i+1]-x[i-1]) - sig * u[i-1]) / p

	if ypn > 0.99e30:
		qn = 0.0
		un = 0.0
	else:
		qn = 0.5
		un = (3.0 / (x[n-1]-x[n-2])) * (ypn-(y[n-1]-y[n-2]) / (x[n-1]-x[n-2]))

	y2[n-1] = (un-qn*u[n-2]) / (qn*y2[n-2]+1.0)

	for k in range(n-2,-1,-1):
		y2[k] = y2[k] * y2[k+1] + u[k]

	free(u)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double splint(double *xa, double *ya, double *y2a, long n,
	double x) nogil:
	"""
	Given a grid xa with function values ya, both of length n,
	this function outputs the interpolated function value at
	the point x. The input y2a is a vector of second derivatives
	computed in spline().
	"""
	cdef:
		long klo, khi, k
		double h, b, a, y

	klo = 1
	khi = n

	while (khi - klo) > 1:
		k = (khi+klo) >> 1
		if xa[k-1] > x:
			khi = k
		else:
			klo = k

	khi -= 1
	klo -= 1

	h = xa[khi] - xa[klo]
	if h == 0:
		raise Exception('Bad xa input to splint routine')

	a = (xa[khi]-x) / h
	b = (x-xa[klo]) / h
	y = a * ya[klo] + b * ya[khi] + \
		((a*a*a-a)*y2a[klo] + (b*b*b-b)*y2a[khi]) * (h*h) / 6.0

	return y
