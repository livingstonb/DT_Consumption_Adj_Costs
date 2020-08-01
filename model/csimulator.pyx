
import numpy as np
cimport numpy as np
import pandas as pd
from misc cimport cfunctions

cimport cython
from cython.parallel cimport prange, parallel
from misc cimport spline

from Params cimport Params
from Income cimport Income

from libc.math cimport log, fabs, fmin, fmax
from libc.stdlib cimport malloc, free

cdef class CSimulator:
	cdef:
		readonly Params p
		readonly Income income
		readonly object grids
		public double[:,:,:,:,:] inactionRegion, cSwitchingPolicy
		public int nCols
		readonly int periodsBeforeRedraw
		public int nSim, t, T, randIndex
		public bint initialized, news
		public long[:,:] switched
		public long[:] yPind, zind
		public double[:,:] ysim, csim, csim_adj, xsim, asim
		public list xgridCurr
		public list borrowLims, borrowLimsCurr

	def __init__(self, params, income, grids, simPeriods):
		self.p = params
		self.income = income
		self.grids = grids

		self.periodsBeforeRedraw = 10

		self.nSim = params.nSim
		self.T = simPeriods
		self.t = 1
		self.randIndex = 0

		self.initialized = False
		self.news = False

		np.random.seed(0)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def solveDecisions(self):
		cdef:
			long i, col, nc, nx
			double[:] cgrid
			double[:] xgrid
			double[:] discount_factor_grid
			double[:] risk_aver_grid
			double deathProb, adjustCost, blim
			long modelNum
		
		cgrid = self.grids.c_flat
		nc = self.p.nc
		nx = self.p.nx
		discount_factor_grid = self.p.discount_factor_grid
		deathProb = self.p.deathProb
		adjustCost = self.p.adjustCost
		risk_aver_grid = self.p.risk_aver_grid

		for col in range(self.nCols):
			xgrid = np.asarray(self.xgridCurr[col])
			blim = self.borrowLimsCurr[col]
			if self.news:
				modelNum = col
			else:
				modelNum = 0

			for i in prange(self.nSim, schedule='static', nogil=True):
				self.findIndividualPolicy(i, col, &cgrid[0], nc, &xgrid[0], nx,
					modelNum, &discount_factor_grid[0], deathProb, adjustCost,
					&risk_aver_grid[0], blim)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	cdef void findIndividualPolicy(self, long i, long col, double *cgrid, 
		long nc, double *xgrid, long nx, long modelNum, double *discount_factor_grid,
		double deathProb, double adjustCost, double *risk_aver_grid, double blim) nogil:
		cdef: 
			long iyP, iz
			double xWeights[2]
			double conInaction[2]
			long xIndices[2]
			bint switch
			double consumption, cash, copt

		iyP = self.yPind[i]
		iz = self.zind[i]

		consumption = self.csim[i,col]
		cash = self.xsim[i,col]
		
		cfunctions.getInterpolationWeights(xgrid, cash, nx, &xIndices[0], &xWeights[0])

		if cash - consumption < blim:
			# forced to switch consumption
			switch = True
		else:
			conInaction[0] = xWeights[0] * self.inactionRegion[xIndices[0],0,iz,iyP,modelNum] \
				+ xWeights[1] * self.inactionRegion[xIndices[1],0,iz,iyP,modelNum]
			conInaction[1] = xWeights[0] * self.inactionRegion[xIndices[0],1,iz,iyP,modelNum] \
				+ xWeights[1] * self.inactionRegion[xIndices[1],1,iz,iyP,modelNum] \

			if (consumption < conInaction[0]) or (consumption > conInaction[1]):
				switch = True
			else:
				switch = False

		if switch:
			if cash <= xgrid[0]:
				self.csim_adj[i,col] = \
					self.cSwitchingPolicy[0,0,iz,iyP,modelNum] \
					- (xgrid[0] - cash)
				self.csim[i,col] = self.cSwitchingPolicy[0,0,iz,iyP,modelNum]
			else:
				copt = xWeights[0] * self.cSwitchingPolicy[xIndices[0],0,iz,iyP,modelNum] \
					+ xWeights[1] * self.cSwitchingPolicy[xIndices[1],0,iz,iyP,modelNum]
				self.csim[i,col] = copt
				self.csim_adj[i,col] = copt

			self.switched[i,col] = 1
		else:
			self.csim_adj[i,col] = consumption
			self.switched[i,col] = 0