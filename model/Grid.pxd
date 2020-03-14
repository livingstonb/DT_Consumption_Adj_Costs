from Params cimport Params

cdef class Grid:
	cdef:
		Params p
		public tuple matrixDim
		public double[:] c_flat, x_flat
		public double[:,:,:,:] c_wide, x_matrix