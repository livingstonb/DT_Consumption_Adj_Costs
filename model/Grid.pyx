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

	def createCashGrid(self, income):
		xmin = self.p.R * self.p.borrowLim + self.p.govTransfer + income.ymin
		xgrid = constructCurvedGrid(xmin,
			self.p.xMax, self.p.xGridCurv, self.p.nx,
			self.p.xGridTerm1Wt, self.p.xGridTerm1Curv)

		self.x_flat = xgrid.flatten()

		xtmp = np.reshape(self.x_flat, (-1, 1, 1, 1))
		self.x_matrix = np.tile(xtmp,
			(1,self.p.nc,self.p.nz,self.p.nyP))

	def createConsumptionGrid(self):
		# cgrid = constructCurvedGrid(self.p.cMin,
		# 	self.p.cMax, self.p.cGridCurv, self.p.nc,
		# 	self.p.cGridTerm1Wt, self.p.cGridTerm1Curv)

		cgrid = self.genAdjustedCGrid(self.x_flat)

		self.c_flat = cgrid.flatten()
		self.c_wide = cgrid.reshape((1,self.p.nc,1,1))

	def genAdjustedCGrid(self, xgrid):
		c1 = np.asarray(xgrid).flatten()

		c0 = np.linspace(1.0e-6, c1[0], num=self.p.nc-self.p.nx+1).flatten()
		cgrid = np.append(c0[:-1], c1)

		return cgrid

	def genAdjustedXGrid(self, lb):
		if lb > self.x_flat[0]:
			new_grid = np.asarray(self.x_flat) + lb - self.x_flat[0]
		else:
			new_grid = np.asarray(self.x_flat)

		return new_grid