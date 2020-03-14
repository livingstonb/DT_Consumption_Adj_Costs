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
		xmin = self.p.borrowLim + self.p.cMin

		ymin = income.ymin + self.p.govTransfer
		# xgrid1 = np.linspace(xmin, xmin + ymin,
		# 	num=self.p.nxLow+1)
		# xgrid1 = xgrid1[0:self.p.nxLow].reshape((-1,1))
		# xgrid2 = constructCurvedGrid(xmin + ymin,
		# 	self.p.xMax, self.p.xGridCurv, self.p.nx-self.p.nxLow,
		# 	self.p.xGridTerm1Wt, self.p.xGridTerm1Curv)

		# xgrid = np.concatenate((xgrid1, xgrid2), axis=0)

		xgrid = constructCurvedGrid(xmin,
			self.p.xMax, self.p.xGridCurv, self.p.nx,
			self.p.xGridTerm1Wt, self.p.xGridTerm1Curv)

		self.x_flat = xgrid.flatten()

		xtmp = np.reshape(self.x_flat, (-1, 1, 1, 1))
		self.x_matrix = np.tile(xtmp,
			(1,self.p.nc,self.p.nz,self.p.nyP))

	def createConsumptionGrid(self):
		cgrid = constructCurvedGrid(self.p.cMin,
			self.p.cMax, self.p.cGridCurv, self.p.nc,
			self.p.cGridTerm1Wt, self.p.cGridTerm1Curv)

		self.c_flat = cgrid.flatten()
		self.c_wide = cgrid.reshape((1,self.p.nc,1,1))