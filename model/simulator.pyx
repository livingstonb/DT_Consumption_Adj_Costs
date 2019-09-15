import numpy as np
cimport numpy as np
import pandas as pd
from misc cimport functions

cimport cython
from cython.parallel import prange, parallel

from libc.math cimport log, fabs
from libc.stdlib cimport malloc, free

cdef class Simulator:
	cdef:
		object p, income, grids
		double[:,:,:,:] valueDiff, cSwitchingPolicy
		int nCols
		int periodsBeforeRedraw
		int nSim, t, T, randIndex
		bint initialized
		long[:,:] switched
		long[:] yPind, zind
		double[:,:] ysim, csim, xsim, asim
		double[:,:] deathrand, yPrand, yTrand

	def __init__(self, params, income, grids, model, simPeriods):
		self.p = params
		self.income = income
		self.grids = grids

		self.cSwitchingPolicy = model.cSwitchingPolicy
		self.valueDiff = np.asarray(model.valueSwitch) \
			- params.adjustCost - np.asarray(model.valueNoSwitch)

		self.nCols = 1

		self.periodsBeforeRedraw = np.minimum(simPeriods,10)

		self.nSim = params.nSim
		self.T = simPeriods
		self.t = 1
		self.randIndex = 0

		self.initialized = False

		np.random.seed(0)

	def makeRandomDraws(self):
		self.yPrand = np.random.random(size=(self.nSim,self.periodsBeforeRedraw))
		self.yTrand = np.random.random(size=(self.nSim,self.periodsBeforeRedraw))

		if (self.p.deathProb > 0) and (not self.p.Bequests):
			self.deathrand = np.random.random(size=(self.nSim,self.periodsBeforeRedraw))

	def updateIncome(self):
		self.yPind = np.argmax(self.yPrand[:,self.randIndex,None]
					<= self.income.yPcumtrans[self.yPind,:],
					axis=1)
		yTind = np.argmax(self.yTrand[:,self.randIndex,None]
								<= self.income.yTcumdistT, axis=1)

		yPsim = self.income.yPgrid[self.yPind]
		yTsim = self.income.yTgrid[yTind]
		self.ysim = (yPsim * yTsim).reshape((-1,1))

	def updateCash(self):
		self.xsim = np.asarray(self.asim) + np.asarray(self.ysim)

	def updateAssets(self):
		self.asim = self.p.R * (np.asarray(self.xsim) - np.asarray(self.csim))
		self.asim = np.minimum(self.asim,self.p.xMax)

		if (self.p.deathProb > 0) and (not self.p.Bequests):
			self.asim[self.deathrand[:,self.randIndex]<self.p.deathProb,:] = 0

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def solveDecisions(self):
		cdef:
			long i, col, nc, nx
			double[:] cgrid
			double[:] xgrid
			double[:,:] csim
			long[:,:] switched

		conIndices = np.zeros((self.nSim))
			
		cgrid = self.grids.c.flat
		nc = self.p.nc
		xgrid = self.grids.x.flat
		nx = self.p.nx

		csim = np.zeros((self.nSim,self.nCols))
		switched = np.zeros((self.nSim,self.nCols),dtype=int)

		with nogil, parallel():
			csim[...] = self.csim
			switched[...] = self.switched

			for col in range(self.nCols):

				for i in prange(self.nSim):
					self.findIndividualPolicy(i, col, cgrid, nc, xgrid, nx, csim, switched)
		self.switched = switched
		self.csim = np.minimum(csim,self.xsim)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	cdef void findIndividualPolicy(self, long i, long col, double[:] cgrid, 
		long nc, double[:] xgrid, long nx, double[:,:] csim, long[:,:] switched) nogil:
		cdef: 
			long iyP, iz
			double *xWeights
			double *conWeights
			long xIndices[2]
			long conIndices[2]
			bint switch
			double consumption, cash, myValueDiff, cSwitch

		xWeights = <double *> malloc(2 * sizeof(double))


		iyP = self.yPind[i]
		iz = self.zind[i]

		consumption = self.csim[i,col]
		cash = self.xsim[i,col]

		xIndices[1] = functions.searchSortedSingleInput(xgrid,cash,nx)
		xIndices[0] = xIndices[1] - 1
		
		functions.getInterpolationWeights(xgrid,cash,xIndices[1],xWeights)

		if consumption > cash:
			# forced to switch consumption
			switch = True
		else:
			conWeights = <double *> malloc(2 * sizeof(double))

			# check if switching is optimal
			conIndices[1] = functions.searchSortedSingleInput(cgrid,consumption,nc)
			conIndices[0] = conIndices[1] - 1
			
			functions.getInterpolationWeights(cgrid,consumption,conIndices[1],conWeights)


			myValueDiff = xWeights[0] * conWeights[0] * self.valueDiff[xIndices[0],conIndices[0],iz,iyP] \
				+ xWeights[1] * conWeights[0] * self.valueDiff[xIndices[1],conIndices[0],iz,iyP] \
				+ xWeights[0] * conWeights[1] * self.valueDiff[xIndices[0],conIndices[1],iz,iyP] \
				+ xWeights[1] * conWeights[1] * self.valueDiff[xIndices[1],conIndices[1],iz,iyP]

			switch = myValueDiff > 0

			free(conWeights)

		if switch:
			self.csim[i,col] = xWeights[0] * self.cSwitchingPolicy[xIndices[0],0,iz,iyP] \
					+ xWeights[1] * self.cSwitchingPolicy[xIndices[1],0,iz,iyP]
			self.switched[i,col] = 1
		else:
			self.switched[i,col] = 0

		free(xWeights)

cdef class EquilibriumSimulator(Simulator):

	cdef public dict transitionStatistics
	cdef public object results
	cdef object incomeHistory

	def __init__(self, params, income, grids, model):
		super().__init__(params,income,grids,model,params.tSim)
		self.initialize()

		self.transitionStatistics = {}
		self.results = pd.Series()

	def simulate(self):
		if not self.initialized:
			raise Exception ('Simulator not initialized')

		while self.t <= self.T:

			if np.mod(self.t,25) == 0:
				print(f'    Simulating period {self.t}')

			if self.t > 1:
				# assets and income should already be initialized
				self.updateAssets()
				self.updateIncome()

			self.updateCash()
			self.solveDecisions()

			self.computeTransitionStatistics()

			if self.randIndex < self.periodsBeforeRedraw - 1:
				# use next column of random numbers
				self.randIndex += 1
			else:
				# need to redraw
				self.randIndex = 0
				self.makeRandomDraws()

			self.t += 1
		self.computeEquilibriumStatistics()

	def initialize(self):
		self.makeRandomDraws()

		self.asim = self.p.wealthTarget * np.ones((self.nSim,self.nCols))
		self.csim = np.ones((self.nSim,self.nCols))

		self.yPind = np.argmax(self.yPrand[:,self.randIndex,None]
					<= self.income.yPcumdistT,
					axis=1)
		yTind = np.argmax(self.yTrand[:,self.randIndex,None]
								<= self.income.yTcumdistT, axis=1)

		yPsim = self.income.yPgrid[self.yPind]
		yTsim = self.income.yTgrid[yTind]
		self.ysim = (yPsim * yTsim).reshape((-1,1))

		self.zind = np.zeros(self.nSim,dtype=int)

		self.switched = np.zeros((self.nSim,1),dtype=int)
		self.incomeHistory = np.zeros((self.nSim,4))

		self.initialized = True

	def computeTransitionStatistics(self):
		"""
		This method computes statistics that can be
		used to evaluate convergence to the equilibrium
		distribution.
		"""
		self.transitionStatistics = {	'E[a]': np.zeros((self.T,)),
										'Var[a]': np.zeros((self.T,))
									}
		self.transitionStatistics['E[a]'][self.t-1] = np.mean(self.asim,axis=0)
		self.transitionStatistics['Var[a]'][self.t-1] = np.var(self.asim,axis=0)

		if self.t >= self.T - 3:
			self.incomeHistory[:,self.t-self.T+3] = np.reshape(self.ysim,self.nSim)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def computeEquilibriumStatistics(self):
		cdef int i, j
		cdef double giniNumerator, asim_i

		# mean wealth
		self.results['Mean wealth'] = np.mean(self.asim)

		# mean cash-on-hand
		self.results['Mean cash-on-hand'] = np.mean(self.xsim)

		# fraction with wealth <= epsilon
		for threshold in self.p.wealthConstraints:
			constrained = np.mean(np.asarray(self.asim) <= threshold)
			self.results[f'Wealth <= {threshold:.2g}'] = constrained

		self.results['Wealth <= own quarterly income/6'] = np.mean(
			np.asarray(self.asim) <= (np.asarray(self.ysim) / 6))
		self.results['Wealth <= own quarterly income/12'] = np.mean(
			np.asarray(self.asim) <= (np.asarray(self.ysim) / 12))

		# wealth percentiles
		for pctile in self.p.wealthPercentiles:
			value = np.percentile(self.asim,pctile)
			if pctile <= 99:
				self.results[f'Wealth, {pctile:d}th percentile'] = value
			else:
				self.results[f'Wealth, {pctile:.2f}th percentile'] = value

		# top shares
		pctile90 = np.percentile(self.asim,90)
		pctile99 = np.percentile(self.asim,99)
		asimNumpy = np.asarray(self.asim)
		self.results['Top 10% wealth share'] = \
			asimNumpy[asimNumpy>=pctile90].sum() / asimNumpy.sum()
		self.results['Top 1% wealth share'] = \
			asimNumpy[asimNumpy>=pctile99].sum() / asimNumpy.sum()

		# gini
		giniNumerator = 0
		for i in prange(self.nSim, nogil=True):
			asim_i = self.asim[i,0]
			for j in range(self.nSim):
				giniNumerator += fabs(asim_i - self.asim[j,0])
		self.results['Gini coefficient (wealth)'] = \
			giniNumerator / (2 * self.nSim * np.sum(self.asim))

		# consumption percentiles
		for pctile in self.p.wealthPercentiles:
			value = np.percentile(self.csim,pctile)
			if pctile <= 99:
				self.results[f'Consumption, {pctile:d}th percentile'] = value
			else:
				self.results[f'Consumption, {pctile:.2f}th percentile'] = value

		# fraction of HHs that switched consumption in last period
		self.results['Probability of consumption change'] = np.mean(self.switched)

		# income statistics
		self.results['Mean annual income'] = self.incomeHistory.sum(axis=1).mean()
		self.results['Variance of annual income'] = self.incomeHistory.sum(axis=1).var()
		self.results['Stdev log annual income'] = np.log(self.incomeHistory.sum(axis=1)).std()

	def returnFinalStates(self):
		finalStates = {
			'yPind': self.yPind,
			'xsim': self.xsim,
			'csim': self.csim,
			'zind': self.zind,
		}
		return finalStates

cdef class MPCSimulator(Simulator):
	cdef public object mpcs
	cdef object responded, pushed_below_xgrid
	cdef list shockIndices
	cdef dict finalStates

	def __init__(self, params, income, grids, model, shockIndices, finalStates):
		Simulator.__init__(self,params,income,grids,model,4)
		self.nCols = len(shockIndices) + 1
		self.shockIndices = shockIndices
		self.mpcs = pd.Series()

		self.finalStates = finalStates

		self.initialize()

	def simulate(self):
		if not self.initialized:
			raise Exception ('Simulator not initialized')

		while self.t <= self.T:

			print(f'    Simulating MPCs, quarter {self.t}')

			if self.t > 1:
				# assets and income should already be initialized
				self.updateAssets()
				self.updateIncome()
				self.updateCash()

			self.solveDecisions()

			self.computeTransitionStatistics()

			if self.randIndex < self.periodsBeforeRedraw - 1:
				# use next column of random numbers
				self.randIndex += 1
			else:
				# need to redraw
				self.randIndex = 0
				self.makeRandomDraws()

			self.t += 1

	def initialize(self):
		self.xsim = np.repeat(self.finalStates['xsim'],self.nCols,axis=1)
		self.csim = np.repeat(self.finalStates['csim'],self.nCols,axis=1)
		self.yPind = self.finalStates['yPind'].copy()
		self.zind = self.finalStates['zind'].copy()

		self.pushed_below_xgrid = np.zeros((self.nSim,self.nCols-1),dtype=bool)

		col = 0
		for ishock in self.shockIndices:
			for i in range(self.p.nSim):
				self.xsim[i,col] += self.p.MPCshocks[ishock]
				if self.xsim[i,col] < self.grids.x.flat[0]:
					self.xsim[i,col] = self.grids.x.flat[0]
					self.pushed_below_xgrid[i,col] = True
			col += 1

		self.makeRandomDraws()

		self.responded = np.zeros((self.nSim,len(self.shockIndices)),dtype=bool)

		# statistics to compute very period
		self.mpcs = pd.Series(name=self.p.name)
		for ishock in range(6):
			for quarter in range(1,5):
				row = f'E[Q{quarter} MPC] out of {self.p.MPCshocks[ishock]}'
				self.mpcs[row] = np.nan
				row = f'E[Q{quarter} MPC | MPC > 0] out of {self.p.MPCshocks[ishock]}'
				self.mpcs[row] = np.nan

			row = f'E[Annual MPC] out of {self.p.MPCshocks[ishock]}'
			self.mpcs[row] = np.nan
			row = f'E[Annual MPC | MPC > 0] out of {self.p.MPCshocks[ishock]}'
			self.mpcs[row] = np.nan


		for ishock in range(6):
			row = f'P(Q1 MPC > 0) for shock of {self.p.MPCshocks[ishock]}'
			self.mpcs[row] = np.nan
			row = f'P(Annual MPC > 0) for shock of {self.p.MPCshocks[ishock]}'
			self.mpcs[row] = np.nan

		self.switched = np.zeros((self.nSim,self.nCols),dtype=int)

		self.initialized = True

	def computeTransitionStatistics(self):
		ii = 0
		for ishock in self.shockIndices:
			rowQuarterly = f'E[Q{self.t} MPC] out of {self.p.MPCshocks[ishock]}'
			rowQuarterlyCond = f'E[Q{self.t} MPC | MPC > 0] out of {self.p.MPCshocks[ishock]}'
			rowAnnual = f'E[Annual MPC] out of {self.p.MPCshocks[ishock]}'
			rowAnnualCond = f'E[Annual MPC | MPC > 0] out of {self.p.MPCshocks[ishock]}'

			csimQuarter = np.asarray(self.csim[:,ii])
			if self.t == 1:
				# adjust consumption response for households pushed below xMin
				# csim(x+delta) = csim(xmin) + x + delta - xmin
				indices = self.pushed_below_xgrid[:,ii]
				if np.any(indices):
					csimQuarter[indices] += np.asarray(self.finalStates['xsim'])[indices].flatten() \
						+ self.p.MPCshocks[ishock] - self.grids.x.flat[0]

			# quarterly mpcs
			allMPCS = (csimQuarter - np.asarray(self.csim[:,self.nCols-1])
					) / self.p.MPCshocks[ishock]
			self.mpcs[rowQuarterly] = allMPCS.mean()
			self.mpcs[rowQuarterlyCond] = allMPCS[allMPCS>0].mean()

			# add quarterly mpcs to annual mpcs
			if self.t == 1:
				self.mpcs[rowAnnual] = self.mpcs[rowQuarterly]
			else:
				self.mpcs[rowAnnual] += self.mpcs[rowQuarterly]

			# fraction of respondents in this quarter
			respondentsQ = (csimQuarter - np.asarray(self.csim[:,self.nCols-1])
				) / self.p.MPCshocks[ishock] > 0
			if self.t == 1:
				rowRespondentsQuarterly = f'P(Q1 MPC > 0) for shock of {self.p.MPCshocks[ishock]}'
				self.mpcs[rowRespondentsQuarterly] = respondentsQ.mean()

			# update if some households responded this period but not previous periods
			self.responded[:,ii] = np.logical_or(self.responded[:,ii],respondentsQ)

			# fraction of respondents (annual)
			if self.t == 4:
				rowRespondentsAnnual = f'P(Annual MPC > 0) for shock of {self.p.MPCshocks[ishock]}'
				self.mpcs[rowRespondentsAnnual] = self.responded[:,ii].mean()
		
			ii += 1
