
import numpy as np
cimport numpy as np
import pandas as pd
from misc cimport cfunctions

cimport cython
from cython.parallel cimport prange, parallel
from misc cimport spline

from Params cimport Params
from Income cimport Income

from libc.math cimport log, fabs
from libc.stdlib cimport malloc, free

cdef class CSimulator:
	"""
	Serves as the base class for simulations.

	cSwitching policies contains the optimal policies
	conditional on switching, with an additional dimension
	for model. The last dimension is used for policies
	out of news.

	valueDiff is the difference between the value of
	switching and the value of not switching, with the
	last dimension being the model.
	"""
	cdef:
		readonly Params p
		readonly Income income
		readonly object grids
		readonly double[:,:,:,:,:] valueDiff, cSwitchingPolicy
		public int nCols
		readonly int periodsBeforeRedraw
		public int nSim, t, T, randIndex
		public bint initialized, news
		public long[:,:] switched
		public long[:] yPind, zind
		public double[:,:] ysim, csim, xsim, asim

	def __init__(self, params, income, grids, cSwitchingPolicies, valueDiffs, simPeriods):
		self.p = params
		self.income = income
		self.grids = grids

		self.cSwitchingPolicy = cSwitchingPolicies
		self.valueDiff = valueDiffs

		self.nCols = valueDiffs.shape[4]

		self.periodsBeforeRedraw = np.minimum(simPeriods,10)

		self.nSim = params.nSim
		self.T = simPeriods
		self.t = 1
		self.randIndex = 0

		self.initialized = False

		if valueDiffs.shape[4] > 1:
			self.news = True
		else:
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
			double deathProb, adjustCost
			long modelNum

		conIndices = np.zeros((self.nSim))
			
		cgrid = self.grids.c_flat
		nc = self.p.nc
		xgrid = self.grids.x_flat
		nx = self.p.nx
		discount_factor_grid = self.p.discount_factor_grid
		deathProb = self.p.deathProb
		adjustCost = self.p.adjustCost
		risk_aver_grid = self.p.risk_aver_grid

		for col in range(self.nCols):
			if self.news:
				modelNum = col
			else:
				modelNum = 0

			for i in prange(self.nSim, schedule='static', nogil=True):
				self.findIndividualPolicy(i, col, &cgrid[0], nc, &xgrid[0], nx,
					modelNum, &discount_factor_grid[0], deathProb, adjustCost,
					&risk_aver_grid[0])

		self.csim = np.minimum(self.csim,self.xsim)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	cdef void findIndividualPolicy(self, long i, long col, double *cgrid, 
		long nc, double *xgrid, long nx, long modelNum, double *discount_factor_grid,
		double deathProb, double adjustCost, double *risk_aver_grid) nogil:
		cdef: 
			long iyP, iz
			double xWeights[2]
			double conWeights[2]
			long xIndices[2]
			long conIndices[2]
			bint switch
			double consumption, cash, myValueDiff

		iyP = self.yPind[i]
		iz = self.zind[i]

		consumption = self.csim[i,col]
		cash = self.xsim[i,col]
		
		cfunctions.getInterpolationWeights(xgrid, cash, nx, &xIndices[0], &xWeights[0])

		if consumption > cash:
			# forced to switch consumption
			switch = True
		else:
			# check if switching is optimal
			cfunctions.getInterpolationWeights(cgrid, consumption, nc, &conIndices[0], &conWeights[0])

			myValueDiff = xWeights[0] * conWeights[0] * self.valueDiff[xIndices[0],conIndices[0],iz,iyP,modelNum] \
				+ xWeights[1] * conWeights[0] * self.valueDiff[xIndices[1],conIndices[0],iz,iyP,modelNum] \
				+ xWeights[0] * conWeights[1] * self.valueDiff[xIndices[0],conIndices[1],iz,iyP,modelNum] \
				+ xWeights[1] * conWeights[1] * self.valueDiff[xIndices[1],conIndices[1],iz,iyP,modelNum]

			switch = (myValueDiff > 0)

		if switch:
			self.csim[i,col] = xWeights[0] * self.cSwitchingPolicy[xIndices[0],0,iz,iyP,modelNum] \
					+ xWeights[1] * self.cSwitchingPolicy[xIndices[1],0,iz,iyP,modelNum]
			self.switched[i,col] = 1
		else:
			self.switched[i,col] = 0