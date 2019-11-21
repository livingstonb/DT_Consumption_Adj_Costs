from Params cimport Params

cdef class Grid:
	cdef:
		Params p
		public tuple matrixDim
		public double[:] c_flat, x_flat
		public double[:,:] c_vec, x_vec
		public double[:,:,:,:] c_wide, x_wide
		public double[:,:,:,:] c_matrix, x_matrix
		public long[:] z_flat
		public long[:,:] z_vec
		public long[:,:,:,:] z_wide, z_matrix
		public object mustSwitch