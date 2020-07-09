from Params cimport Params

cdef class Income:
	cdef:
		Params p
		public long nyP, nyT
		public double[:] yPgrid, logyPgrid
		public double[:] yPdist, yPcumdist
		public double[:,:] yPtrans, ymat, ydist
		public object yPcumtrans
		public double[:] yTgrid, yTdist, logyTgrid
		public double[:] yTcumdist
		public object yTcumdistT, yPcumdistT
		public double ymin, meany
		public bint normalize