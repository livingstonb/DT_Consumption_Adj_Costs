import numpy as np
cimport numpy as np

from misc.functions import constructCurvedGrid

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

		self.create_zgrid()

	def createCashGrid(self, income):
		xmin = self.p.borrowLim + self.p.cMin

		ymin = income.ymin + self.p.govTransfer
		xgrid1 = np.linspace(xmin, xmin+ymin, num=11)
		xgrid1 = xgrid1[0:10].reshape((10,1))
		xgrid2 = constructCurvedGrid(xmin + ymin,
			self.p.xMax, self.p.xGridCurv, self.p.nx-10)

		xgrid = np.concatenate((xgrid1, xgrid2), axis=0)
		xgrid = self.enforceMinGridSpacing(xgrid)

		self.x_flat = xgrid.flatten()
		self.x_vec = xgrid
		self.x_wide = xgrid.reshape((-1,1,1,1))
		self.x_matrix = np.tile(self.x_wide,
			(1,self.p.nc,self.p.nz,self.p.nyP))

	def createConsumptionGrid(self):
		cgrid = constructCurvedGrid(self.p.cMin,
			self.p.cMax, self.p.cGridCurv, self.p.nc)
		cgrid = self.enforceMinGridSpacing(cgrid)

		self.c_flat = cgrid.flatten()
		self.c_vec = cgrid
		self.c_wide = cgrid.reshape((1,self.p.nc,1,1))
		self.c_matrix = np.tile(self.c_wide,
			(self.p.nx,1,self.p.nz,self.p.nyP))

	# def getFineConsumptionGrid(self, curv, npts):


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

