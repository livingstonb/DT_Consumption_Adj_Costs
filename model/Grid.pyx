import numpy as np
cimport numpy as np

cdef class Grid:
	def __init__(self, params, income):
		self.p = params

		self.matrixDim = (	params.nx,
							params.nc,
							params.nz,
							params.nyP,
							)

		self.createCashGrid(income)

		self.createConsumptionGrid()

		self.mustSwitch = np.asarray(self.c_matrix) > np.asarray(self.x_matrix)

		self.create_zgrid()

	def createCashGrid(self, income):
		xmin = self.p.borrowLim + income.ymin \
			+ self.p.govTransfer

		xgrid = np.linspace(0,1,num=self.p.nx)
		xgrid = xgrid.reshape((self.p.nx,1))
		xgrid = xgrid ** (1.0 / self.p.xGridCurv)
		xgrid = xmin \
			+ (self.p.xMax - xmin) * xgrid

		xgrid = self.enforceMinGridSpacing(xgrid)

		self.x_flat = xgrid.flatten()
		self.x_vec = xgrid
		self.x_wide = xgrid.reshape((-1,1,1,1))
		self.x_matrix = np.tile(self.x_wide,
			(1,self.p.nc,self.p.nz,self.p.nyP))

	def createConsumptionGrid(self):
		cgrid = np.linspace(0,1,num=self.p.nc)
		cgrid = cgrid.reshape((self.p.nc,1))
		cgrid = cgrid ** (1 / self.p.cGridCurv)
		cgrid = self.p.cMin \
			+ (self.p.cMax - self.p.cMin) * cgrid

		cgrid = self.enforceMinGridSpacing(cgrid)

		self.c_flat = cgrid.flatten()
		self.c_vec = cgrid
		self.c_wide = cgrid.reshape((1,self.p.nc,1,1))
		self.c_matrix = np.tile(self.c_wide,
			(self.p.nx,1,self.p.nz,self.p.nyP))

	def create_zgrid(self):
		zgrid = np.arange(self.p.nz).reshape((-1,1))
		self.z_flat = zgrid.flatten()
		self.z_vec = zgrid
		self.z_wide = zgrid.reshape((1,1,self.p.nz,1))
		self.z_matrix = np.tile(self.z_wide,
			(self.p.nx,self.p.nc,1,self.p.nyP))

	def enforceMinGridSpacing(self, grid_in):
		grid_adj = grid_in
		for i in range(grid_in.size-1):
			if grid_adj[i+1] - grid_adj[i] < self.p.minGridSpacing:
				grid_adj[i+1] = grid_adj[i] + self.p.minGridSpacing

		return grid_adj