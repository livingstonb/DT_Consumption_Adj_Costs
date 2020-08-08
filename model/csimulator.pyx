
import numpy as np
cimport numpy as np
import pandas as pd
from misc cimport cfunctions

cimport cython
from cython.parallel cimport prange, parallel

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
		public double[:,:] ysim, csim, xsim, asim
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
			long i, col
			double[:] xgrid
			double blim
			long modelNum

		for col in range(self.nCols):
			xgrid = np.asarray(self.xgridCurr[col])
			blim = self.borrowLimsCurr[col]
			if self.news:
				modelNum = col
			else:
				modelNum = 0

			for i in prange(self.nSim, schedule='static', nogil=True):
				self.findIndividualPolicy(i, col, &xgrid[0], modelNum, blim)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	cdef void findIndividualPolicy(self, long i, long col, double *xgrid,
		long modelNum, double blim) nogil:
		"""
		Simulates the consumption decision for a given simulated household
		in a given model.
		"""
		cdef: 
			long iyP, iz
			long xIndices[2]
			bint switch
			double consumption, cash, inactionLow, inactionHigh, w0

		iyP = self.yPind[i]
		iz = self.zind[i]

		consumption = self.csim[i,col]
		cash = self.xsim[i,col]
		
		w0 = cfunctions.getInterpolationWeight(xgrid, cash, self.p.nx, &xIndices[0])

		if cash - consumption < blim:
			# forced to switch consumption
			switch = True
		else:
			# check if household wants to switch
			inactionLow = w0 * self.inactionRegion[xIndices[0],0,iz,iyP,modelNum] \
				+ (1-w0) * self.inactionRegion[xIndices[1],0,iz,iyP,modelNum]
			inactionHigh = w0 * self.inactionRegion[xIndices[0],1,iz,iyP,modelNum] \
				+ (1-w0) * self.inactionRegion[xIndices[1],1,iz,iyP,modelNum] \

			switch = (consumption < inactionLow) or (consumption > inactionHigh)

		if switch:
			if cash <= xgrid[0]:
				# act as if the household DID have enough cash to incur the loss (i.e.
				# enough cash that we could use interpolation)
				self.csim[i,col] = \
					self.cSwitchingPolicy[0,0,iz,iyP,modelNum] \
					- (xgrid[0] - cash)
			else:
				self.csim[i,col] = w0 * self.cSwitchingPolicy[xIndices[0],0,iz,iyP,modelNum] \
					+ (1-w0) * self.cSwitchingPolicy[xIndices[1],0,iz,iyP,modelNum]

			self.switched[i,col] = 1
		else:
			self.switched[i,col] = 0