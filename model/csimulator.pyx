
import numpy as np
cimport numpy as np
import pandas as pd
from misc cimport cfunctions

cimport cython
from cython.parallel cimport prange, parallel
from misc cimport spline

from libc.math cimport log, fabs
from libc.stdlib cimport malloc, free

cdef class CSimulator:
	cdef:
		readonly object p, income, grids
		readonly double[:,:,:,:,:] valueDiff, cSwitchingPolicy
		readonly double[:,:,:,:,:] yderivs
		readonly double[:,:,:,:,:] EMAX
		public int nCols
		readonly int periodsBeforeRedraw
		public int nSim, t, T, randIndex
		public bint initialized, news
		public long[:,:] switched
		public long[:] yPind, zind
		public double[:,:] ysim, csim, xsim, asim

	def __init__(self, params, income, grids, models, simPeriods):
		self.p = params
		self.income = income
		self.grids = grids

		cSwitchingPolicy = np.zeros((params.nx,1,params.nz,params.nyP,len(models)),dtype=float)
		valueDiff = np.zeros((params.nx,params.nc,params.nz,params.nyP,len(models)),dtype=float)
		EMAX = np.zeros((params.nx,params.nc,params.nz,params.nyP,len(models)),dtype=float)

		for i in range(len(models)):
			cSwitchingPolicy[:,:,:,:,i] = np.asarray(models[i].cSwitchingPolicy)
			valueDiff[:,:,:,:,i] = np.asarray(models[i].valueSwitch) - np.asarray(models[i].valueNoSwitch)
			EMAX[:,:,:,:,i] = np.asarray(models[i].EMAX)
		self.cSwitchingPolicy = cSwitchingPolicy
		self.valueDiff = valueDiff
		self.EMAX = EMAX

		self.nCols = len(models)

		self.periodsBeforeRedraw = np.minimum(simPeriods,10)

		self.nSim = params.nSim
		self.T = simPeriods
		self.t = 1
		self.randIndex = 0

		self.initialized = False

		if len(models) > 1:
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
			
		cgrid = self.grids.c.flat
		nc = self.p.nc
		xgrid = self.grids.x.flat
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
			double consumption, cash, emaxNoSwitch, emaxSwitch, cSwitch

		iyP = self.yPind[i]
		iz = self.zind[i]

		consumption = self.csim[i,col]
		cash = self.xsim[i,col]
		
		cfunctions.getInterpolationWeights(xgrid, cash, nx, &xIndices[0], &xWeights[0])

		cSwitch = xWeights[0] * self.cSwitchingPolicy[xIndices[0],0,iz,iyP,modelNum] \
					+ xWeights[1] * self.cSwitchingPolicy[xIndices[1],0,iz,iyP,modelNum]

		if consumption > cash:
			# forced to switch consumption
			switch = True
		else:
			# check if switching is optimal
			uSwitch = cfunctions.utility(risk_aver_grid[iz], cSwitch);
			uNoSwitch = cfunctions.utility(risk_aver_grid[iz], consumption);

			cfunctions.getInterpolationWeights(cgrid, consumption, nc, &conIndices[0], &conWeights[0])

			emaxNoSwitch = xWeights[0] * conWeights[0] * self.EMAX[xIndices[0],conIndices[0],iz,iyP,modelNum] \
				+ xWeights[1] * conWeights[0] * self.EMAX[xIndices[1],conIndices[0],iz,iyP,modelNum] \
				+ xWeights[0] * conWeights[1] * self.EMAX[xIndices[0],conIndices[1],iz,iyP,modelNum] \
				+ xWeights[1] * conWeights[1] * self.EMAX[xIndices[1],conIndices[1],iz,iyP,modelNum]

			cfunctions.getInterpolationWeights(cgrid, cSwitch, nc, &conIndices[0], &conWeights[0])

			emaxSwitch = xWeights[0] * conWeights[0] * self.EMAX[xIndices[0],conIndices[0],iz,iyP,modelNum] \
				+ xWeights[1] * conWeights[0] * self.EMAX[xIndices[1],conIndices[0],iz,iyP,modelNum] \
				+ xWeights[0] * conWeights[1] * self.EMAX[xIndices[0],conIndices[1],iz,iyP,modelNum] \
				+ xWeights[1] * conWeights[1] * self.EMAX[xIndices[1],conIndices[1],iz,iyP,modelNum]

			switch = (uSwitch + discount_factor_grid[iz] * (1-deathProb) * emaxSwitch - adjustCost) \
				> (uNoSwitch + discount_factor_grid[iz] * (1-deathProb) * emaxNoSwitch)

		if switch:
			self.csim[i,col] = cSwitch
			self.switched[i,col] = 1
		else:
			self.switched[i,col] = 0

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def computeGini(self):
		cdef long i
		cdef double giniNumerator = 0.0

		for i in prange(self.nSim, schedule='static', nogil=True):
			self.giniHelper(&giniNumerator, i)

		self.results['Gini coefficient (wealth)'] = \
			giniNumerator / (2 * self.nSim * np.sum(self.asim))

	@cython.boundscheck(False)
	@cython.wraparound(False)
	cdef void giniHelper(self, double *giniNumerator, long i) nogil:
		cdef long j
		cdef double asim_i

		asim_i = self.asim[i,0]
		for j in range(self.nSim):
			giniNumerator[0] += fabs(asim_i - self.asim[j,0])