
import numpy as np
cimport numpy as np
import pandas as pd
from misc cimport functions

cimport cython
from cython.parallel import prange, parallel

from libc.math cimport log, fabs
from libc.stdlib cimport malloc, free

cdef class CSimulator:
	cdef:
		readonly object p, income, grids
		readonly double[:,:,:,:] valueDiff, cSwitchingPolicy
		public int nCols
		readonly int periodsBeforeRedraw
		public int nSim, t, T, randIndex
		public bint initialized
		public long[:,:] switched
		public long[:] yPind, zind
		public double[:,:] ysim, csim, xsim, asim

	def __init__(self, params, income, grids, model, simPeriods):
		self.p = params
		self.income = income
		self.grids = grids

		self.cSwitchingPolicy = model.cSwitchingPolicy
		self.valueDiff = np.asarray(model.valueSwitch) - np.asarray(model.valueNoSwitch)

		self.nCols = 1

		self.periodsBeforeRedraw = np.minimum(simPeriods,10)

		self.nSim = params.nSim
		self.T = simPeriods
		self.t = 1
		self.randIndex = 0

		self.initialized = False

		np.random.seed(0)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def solveDecisions(self):
		cdef:
			long i, col, nc, nx
			double[:] cgrid
			double[:] xgrid

		conIndices = np.zeros((self.nSim))
			
		cgrid = self.grids.c.flat
		nc = self.p.nc
		xgrid = self.grids.x.flat
		nx = self.p.nx

		for col in range(self.nCols):
			for i in prange(self.nSim,nogil=True):
				self.findIndividualPolicy(i, col, cgrid, nc, xgrid, nx)

		self.csim = np.minimum(self.csim,self.xsim)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	cdef void findIndividualPolicy(self, long i, long col, double[:] cgrid, 
		long nc, double[:] xgrid, long nx) nogil:
		cdef: 
			long iyP, iz
			double xWeights[2]
			double conWeights[2]
			long xIndices[2]
			long conIndices[2]
			bint switch
			double consumption, cash, myValueDiff, cSwitch

		iyP = self.yPind[i]
		iz = self.zind[i]

		consumption = self.csim[i,col]
		cash = self.xsim[i,col]
		
		functions.getInterpolationWeights(&xgrid[0], cash, nx, &xIndices[0], &xWeights[0])

		if consumption > cash:
			# forced to switch consumption
			switch = True
		else:
			# check if switching is optimal
			functions.getInterpolationWeights(&cgrid[0], consumption, nc, &conIndices[0], &conWeights[0])

			myValueDiff = xWeights[0] * conWeights[0] * self.valueDiff[xIndices[0],conIndices[0],iz,iyP] \
				+ xWeights[1] * conWeights[0] * self.valueDiff[xIndices[1],conIndices[0],iz,iyP] \
				+ xWeights[0] * conWeights[1] * self.valueDiff[xIndices[0],conIndices[1],iz,iyP] \
				+ xWeights[1] * conWeights[1] * self.valueDiff[xIndices[1],conIndices[1],iz,iyP]

			switch = myValueDiff > 0

		if switch:
			self.csim[i,col] = xWeights[0] * self.cSwitchingPolicy[xIndices[0],0,iz,iyP] \
					+ xWeights[1] * self.cSwitchingPolicy[xIndices[1],0,iz,iyP]
			self.switched[i,col] = 1
		else:
			self.switched[i,col] = 0

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def computeGini(self):
		cdef long i, j
		cdef double asim_i, giniNumerator = 0.0

		for i in prange(self.nSim, nogil=True):
			asim_i = self.asim[i,0]
			for j in range(self.nSim):
				giniNumerator += fabs(asim_i - self.asim[j,0])

		self.results['Gini coefficient (wealth)'] = \
			giniNumerator / (2 * self.nSim * np.sum(self.asim))