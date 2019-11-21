import numpy as np

cdef class GridDouble:
	cdef public double[:] flat
	cdef public double[:,:] vec
	cdef public double[:,:,:,:] wide
	cdef public double[:,:,:,:] matrix

cdef class GridInt:
	cdef public long[:] flat
	cdef public long[:,:] vec
	cdef public long[:,:,:,:] wide
	cdef public long[:,:,:,:] matrix


cdef class GridCreator:
	cdef:
		object p
		public tuple matrixDim
		public GridDouble c, x
		public GridInt z
		public object mustSwitch

	def __init__(self, params, income):
		self.p = params

		self.matrixDim = (	params.nx,
							params.nc,
							params.nz,
							params.nyP,
							)

		self.createCashGrid(income)

		self.createConsumptionGrid()

		self.mustSwitch = np.asarray(self.c.matrix) > np.asarray(self.x.matrix)

		self.create_zgrid()

	def createCashGrid(self, income):
		self.x  = GridDouble()

		xmin = self.p.borrowLim + income.ymin \
			+ self.p.govTransfer

		xgrid = np.linspace(0,1,num=self.p.nx)
		xgrid = xgrid.reshape((self.p.nx,1))
		xgrid = xgrid ** (1.0 / self.p.xGridCurv)
		xgrid = xmin \
			+ (self.p.xMax - xmin) * xgrid

		xgrid = self.enforceMinGridSpacing(xgrid)

		self.x.flat = xgrid.flatten()
		self.x.vec = xgrid
		self.x.wide = xgrid.reshape((-1,1,1,1))
		self.x.matrix = np.tile(self.x.wide,
			(1,self.p.nc,self.p.nz,self.p.nyP))

	def createConsumptionGrid(self):
		self.c = GridDouble()

		cgrid = np.linspace(0,1,num=self.p.nc)
		cgrid = cgrid.reshape((self.p.nc,1))
		cgrid = cgrid ** (1 / self.p.cGridCurv)
		cgrid = self.p.cMin \
			+ (self.p.cMax - self.p.cMin) * cgrid

		cgrid = self.enforceMinGridSpacing(cgrid)

		self.c.flat = cgrid.flatten()
		self.c.vec = cgrid
		self.c.wide = cgrid.reshape((1,self.p.nc,1,1))
		self.c.matrix = np.tile(self.c.wide,
			(self.p.nx,1,self.p.nz,self.p.nyP))

	def create_zgrid(self):
		self.z = GridInt()

		zgrid = np.arange(self.p.nz).reshape((-1,1))
		self.z.flat = zgrid.flatten()
		self.z.vec = zgrid
		self.z.wide = zgrid.reshape((1,1,self.p.nz,1))
		self.z.matrix = np.tile(self.z.wide,
			(self.p.nx,self.p.nc,1,self.p.nyP))

	def enforceMinGridSpacing(self, grid_in):
		grid_adj = grid_in
		for i in range(grid_in.size-1):
			if grid_adj[i+1] - grid_adj[i] < self.p.minGridSpacing:
				grid_adj[i+1] = grid_adj[i] + self.p.minGridSpacing

		return grid_adj