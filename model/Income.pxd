from Params cimport Params

cdef class Income:
	"""
	This class stores income variables.

	Mean annual income is normalized to 1 by
	normalizing persistent income to have mean
	1 and transitory income to have mean 1 if
	frequency is annual and 1/4 if frequency
	is quarterly.
	"""

	cdef:
		Params p
		public long nyP, nyT
		public double[:] yPgrid, logyPgrid
		public double[:] yPdist, yPcumdist
		public double[:,:] yPtrans, ymat
		public object yPcumtrans
		public double[:] yTgrid, yTdist, logyTgrid
		public double[:] yTcumdist
		public object yTcumdistT, yPcumdistT
		public double ymin