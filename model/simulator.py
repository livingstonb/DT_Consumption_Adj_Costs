from model.csimulator import CSimulator

from misc.cfunctions import gini
from misc import functions

import numpy as np
import pandas as pd

class Simulator(CSimulator):
	"""
	Base class for simulations.
	"""
	def __init__(self, params, income, grids, simPeriods):
		super().__init__(params, income, grids, simPeriods)

	def initialize(self, cSwitchingPolicies, inactionRegions):
		self.cSwitchingPolicy = cSwitchingPolicies
		self.inactionRegion = inactionRegions

		self.borrowLimsCurr = [self.p.borrowLim] * self.nCols
		self.xgridCurr = [self.grids.x_flat] * self.nCols

	def makeRandomDraws(self):
		self.yPrand = np.random.random(size=(self.nSim,self.periodsBeforeRedraw))
		self.yTrand = np.random.random(size=(self.nSim,self.periodsBeforeRedraw))

		if (self.p.deathProb > 0) and (not self.p.Bequests):
			self.deathrand = np.random.random(size=(self.nSim,self.periodsBeforeRedraw))

	def updateIncome(self):
		self.yPind = np.argmax(self.yPrand[:,self.randIndex,None]
			<= self.income.yPcumtrans[self.yPind,:], axis=1)
		yTind = np.argmax(self.yTrand[:,self.randIndex,None]
			<= self.income.yTcumdistT, axis=1)

		if self.income.nyP > 1:
			yPsim = np.asarray(self.income.yPgrid)[self.yPind]
		else:
			yPsim = self.income.yPgrid[0] * np.ones(self.nSim)

		if self.income.nyT > 1:
			yTsim = np.asarray(self.income.yTgrid)[yTind]

		self.ysim = (yPsim * yTsim).reshape((-1,1))

	def updateCash(self):
		self.xsim = np.asarray(self.asim) + np.asarray(self.ysim) + self.p.govTransfer

	def updateAssets(self):
		self.asim = self.p.R * (np.asarray(self.xsim) - np.asarray(self.csim))
		
		if (self.p.deathProb > 0) and (not self.p.Bequests):
			self.asim[self.deathrand[:,self.randIndex]<self.p.deathProb,:] = 0

		for col in range(self.nCols):
			self.asim[:,col] = np.maximum(self.asim[:,col], self.p.R * self.borrowLimsCurr[col])

class EquilibriumSimulator(Simulator):
	"""
	This class is used to simulate statistics for the solved model.
	"""
	def __init__(self, params, income, grids):
		super().__init__(params, income, grids, params.tSim)

		self.transitionStatistics = {}
		self.results = pd.Series()
		self.nCols = 1

	def initialize(self, cSwitchingPolicies, inactionRegions):
		Simulator.initialize(self, cSwitchingPolicies, inactionRegions)

		self.makeRandomDraws()

		self.asim = 0.5 * np.ones((self.nSim,self.nCols))
		self.csim = 0.1 * np.ones((self.nSim,self.nCols))

		self.yPind = np.argmax(
			self.yPrand[:,self.randIndex,np.newaxis]
				<= self.income.yPcumdistT, axis=1)
		yTind = np.argmax(
			self.yTrand[:,self.randIndex,np.newaxis]
				<= self.income.yTcumdistT, axis=1)

		if self.income.nyP == 1:
			yPsim = self.income.yPgrid[0] * np.ones(self.nSim)
		else:
			yPsim = np.asarray(self.income.yPgrid)[np.asarray(self.yPind)]

		if self.income.nyT == 1:
			yTsim = self.income.yTgrid[0] * np.ones(self.nSim)
		else:
			yTsim = np.asarray(self.income.yTgrid)[yTind]
		self.ysim = (yPsim * yTsim).reshape((-1,1))

		chunkSize = self.nSim // self.p.nz
		chunks = [0]
		for iz in range(1,self.p.nz):
			chunks.append(chunks[iz-1]+chunkSize)
		chunks.append(self.nSim)

		self.zind = np.zeros(self.nSim,dtype=int)
		for iz in range(self.p.nz):
			self.zind[chunks[iz]:chunks[iz+1]] = iz * np.ones(
				chunks[iz+1]-chunks[iz], dtype=int)

		self.switched = np.zeros((self.nSim,1), dtype=int)
		self.incomeHistory = np.zeros((self.nSim,4))

		self.initialized = True

	def simulate(self):
		np.random.seed(1991)
		if not self.initialized:
			raise Exception ('Simulator not initialized')

		while self.t <= self.T:

			if np.mod(self.t, 100) == 0:
				print(f'    Simulating period {self.t}')

			if self.t > 1:
				# assets and income should already be initialized
				self.updateAssets()
				self.updateIncome()
			
			self.updateCash()
			self.solveDecisions()

			self.computeTransitionStatistics()
			self.updateRandIndex()

			self.t += 1

		self.computeEquilibriumStatistics()

	def updateRandIndex(self):
		if self.randIndex < self.periodsBeforeRedraw - 1:
			# use next column of random numbers
			self.randIndex += 1
		else:
			# need to redraw
			self.randIndex = 0
			self.makeRandomDraws()

	def computeTransitionStatistics(self):
		"""
		This method computes statistics that can be
		used to evaluate convergence to the equilibrium
		distribution.
		"""
		if self.t == 1:
			self.transitionStatistics = {	'E[a]': np.zeros((self.T,)),
											'Var[a]': np.zeros((self.T,))
										}

		self.transitionStatistics['E[a]'][self.t-1] = np.mean(self.asim, axis=0)
		self.transitionStatistics['Var[a]'][self.t-1] = np.var(self.asim, axis=0)

		if self.t >= self.T - 3:
			self.incomeHistory[:,self.t-self.T+3] = np.reshape(self.ysim,self.nSim)

		if self.t == self.T - 1:
			self.finalStates = {'csim': self.csim}
		elif self.t == self.T:
			self.finalStates.update({	'yPind': self.yPind,
										'xsim': self.xsim,
										'zind': self.zind,
										'asim': self.asim,
										})

	def computeEquilibriumStatistics(self):
		# mean wealth
		self.results['Mean wealth'] = np.mean(self.asim)

		# mean cash-on-hand
		self.results['Mean cash-on-hand'] = np.mean(self.xsim)

		# fraction with wealth <= epsilon
		for threshold in self.p.wealthConstraints:
			constrained = np.mean(np.asarray(self.asim) <= threshold)
			self.results[f'Wealth <= {threshold:.2g}'] = constrained

		# CDF over assets
		a_unique, a_counts = np.unique(self.asim, return_counts=True)
		self.cdf_a = np.zeros((a_unique.size, 2))
		self.cdf_a[:,0] = a_unique
		self.cdf_a[:,1] = np.cumsum(a_counts) / self.nSim

		self.results['Wealth <= $1000'] = np.mean(np.asarray(self.asim) <= 0.0162)
		self.results['Wealth <= $5000'] = np.mean(np.asarray(self.asim) <= 0.081)
		self.results['Wealth <= $10,000'] = np.mean(np.asarray(self.asim) <= 0.162)
		self.results['Wealth <= $25,000'] = np.mean(np.asarray(self.asim) <= 0.405)
		self.results['Wealth <= $250,000'] = np.mean(np.asarray(self.asim) <= 4.05)

		bounds = [0.005, 0.025]
		value = 0.0162
		self.results['Wealth <= $1000 interp'] = functions.eval_smoothed(
			self.cdf_a[:,0], self.cdf_a[:,1], bounds, value)

		self.results['Wealth <= own quarterly income/6'] = np.mean(
			np.asarray(self.asim) <= (np.asarray(self.ysim) / 6))
		self.results['Wealth <= own quarterly income/12'] = np.mean(
			np.asarray(self.asim) <= (np.asarray(self.ysim) / 12))

		# fraction of households who consume all of their cash-on-hand
		rowName = f'P(consumption=x)'
		self.results[rowName] = (np.asarray(self.xsim[:,0]) == np.asarray(self.csim[:,0])).mean()

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
			asimNumpy[asimNumpy>pctile90].sum() / asimNumpy.sum()
		self.results['Top 1% wealth share'] = \
			asimNumpy[asimNumpy>pctile99].sum() / asimNumpy.sum()

		self.results['Gini coefficient (wealth)'] = gini(self.asim[:,0])

		# consumption percentiles
		for pctile in self.p.wealthPercentiles:
			value = np.percentile(self.csim, pctile)
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

		print(f"Mean wealth = {self.results['Mean wealth']}")

class MPCSimulator(Simulator):
	"""
	Class for simulating MPCs. The first column of simulation variables (e.g. xsim) corresponds
	with the first shock index passed, the second column corresponds with the second shock
	index passed, etc. while the last column corresponds with the baseline, i.e. no shock.
	"""
	def __init__(self, params, income, grids, shockIndices):
		super().__init__(params, income, grids, 4)

		self.nCols = len(shockIndices) + 1
		self.shockIndices = shockIndices

		self.mpcs = pd.Series()
		self.initialize_results()

	def initialize(self, cSwitchingPolicies, inactionRegions,
		finalStates):
		Simulator.initialize(self, cSwitchingPolicies, inactionRegions)

		self.finalStates = finalStates
		self.initialize_variables()
		self.initialized = True

	def simulate(self):
		np.random.seed(1991)
		if not self.initialized:
			raise Exception ('Simulator not initialized')

		while self.t <= self.T:

			print(f'    Simulating MPCs, quarter {self.t}')
			self.updatePolicies()

			if self.t > 1:
				# assets and income should already be initialized
				self.updateAssets()
				self.updateIncome()
				self.updateCash()

			self.applyShocks()

			self.updateCashGrids()
			self.solveDecisions()

			self.computeTransitionStatistics()
			self.updateRandIndex()

			self.t += 1

	def updateCashGrids(self):
		pass

	def updatePolicies(self):
		pass

	def initialize_variables(self):
		self.xsim = np.repeat(self.finalStates['xsim'],self.nCols,axis=1)
		self.csim = np.repeat(self.finalStates['csim'],self.nCols,axis=1)
		self.yPind = self.finalStates['yPind'].copy()
		self.zind = self.finalStates['zind'].copy()

		self.makeRandomDraws()

		self.cumResponse = np.zeros((self.nSim,len(self.shockIndices)))

	def applyShocks(self):
		if self.t == 1:
			col = 0
			for ishock in self.shockIndices:
				self.xsim[:,col] = np.asarray(self.xsim[:,col]) + self.p.MPCshocks[ishock]
				col += 1

	def initialize_results(self):
		# statistics to compute very period
		self.results = pd.Series(name=self.p.name)
		rows = []
		for ishock in range(6):
			for quarter in range(1,5):
				rows.append(f'E[Q{quarter} MPC] out of {self.p.MPCshocks[ishock]}')
				rows.append(f'E[Q{quarter} MPC | MPC > 0] out of {self.p.MPCshocks[ishock]}')
				rows.append(f'Median(Q{quarter} MPC | MPC > 0) out of {self.p.MPCshocks[ishock]}')

			rows.append(f'E[Annual MPC] out of {self.p.MPCshocks[ishock]}')
			rows.append(f'E[Annual MPC | MPC > 0] out of {self.p.MPCshocks[ishock]}')
			rows.append(f'Median(Annual MPC | MPC > 0) out of {self.p.MPCshocks[ishock]}')

		for ishock in range(6):
			rows.append(f'P(Q1 MPC < 0) for shock of {self.p.MPCshocks[ishock]}')
			rows.append(f'P(Q1 MPC = 0) for shock of {self.p.MPCshocks[ishock]}')
			rows.append(f'P(Q1 MPC > 0) for shock of {self.p.MPCshocks[ishock]}')
			rows.append(f'P(Annual MPC < 0) for shock of {self.p.MPCshocks[ishock]}')
			rows.append(f'P(Annual MPC > 0) for shock of {self.p.MPCshocks[ishock]}')

		for row in rows:
			self.results[row] = np.nan

		self.mpcs = dict()
		self.switched = np.zeros((self.nSim,self.nCols), dtype=int)

	def computeTransitionStatistics(self):
		ii = 0
		for ishock in self.shockIndices:
			rowQuarterly = f'E[Q{self.t} MPC] out of {self.p.MPCshocks[ishock]}'
			rowQuarterlyCond = f'E[Q{self.t} MPC | MPC > 0] out of {self.p.MPCshocks[ishock]}'
			rowQuarterlyCondMedian = f'Median(Q{self.t} MPC | MPC > 0) out of {self.p.MPCshocks[ishock]}'
			rowAnnual = f'E[Annual MPC] out of {self.p.MPCshocks[ishock]}'
			rowAnnualCond = f'E[Annual MPC | MPC > 0] out of {self.p.MPCshocks[ishock]}'
			rowAnnualCondMedian = f'Median(Annual MPC | MPC > 0) out of {self.p.MPCshocks[ishock]}'

			csimQuarter = np.asarray(self.csim[:,ii])

			# quarterly mpcs
			allMPCS = (csimQuarter - np.asarray(self.csim[:,self.nCols-1])
					) / self.p.MPCshocks[ishock]


			self.cumResponse[:,ii] += allMPCS

			if self.t == 1:
				self.mpcs[ishock] = allMPCS

			self.results[rowQuarterly] = allMPCS.mean()

			if np.asarray(allMPCS[allMPCS>0]).size > 0:
				self.results[rowQuarterlyCond] = allMPCS[allMPCS>0].mean()
				self.results[rowQuarterlyCondMedian] = np.median(allMPCS[allMPCS>0]);

			# add quarterly mpcs to annual mpcs
			if self.t == 1:
				self.results[rowAnnual] = self.results[rowQuarterly]
			else:
				self.results[rowAnnual] += self.results[rowQuarterly]

			# fraction of respondents in this quarter
			respondentsQ = allMPCS > 0
			respondentsQ_neg = allMPCS < 0
			nonRespondents = (allMPCS == 0)
			if self.t == 1:
				rowRespondentsQuarterly = f'P(Q1 MPC < 0) for shock of {self.p.MPCshocks[ishock]}'
				self.results[rowRespondentsQuarterly] = respondentsQ_neg.mean()
				rowRespondentsQuarterly = f'P(Q1 MPC = 0) for shock of {self.p.MPCshocks[ishock]}'
				self.results[rowRespondentsQuarterly] =  nonRespondents.mean()
				rowRespondentsQuarterly = f'P(Q1 MPC > 0) for shock of {self.p.MPCshocks[ishock]}'
				self.results[rowRespondentsQuarterly] = respondentsQ.mean()

			# Fraction of respondents (annual)
			if self.t == 4:
				rowRespondentsAnnual = f'P(Annual MPC > 0) for shock of {self.p.MPCshocks[ishock]}'
				rowNegRespondentsAnnual = f'P(Annual MPC < 0) for shock of {self.p.MPCshocks[ishock]}'
				respondentsA = (self.cumResponse[:,ii] > 0)
				respondentsA_neg = (self.cumResponse[:,ii] < 0)
				self.results[rowRespondentsAnnual] = respondentsA.mean()
				self.results[rowNegRespondentsAnnual] = respondentsA_neg.mean()
				self.results[rowAnnualCond] = self.cumResponse[respondentsA,ii].mean()
				self.results[rowAnnualCondMedian] = np.median(self.cumResponse[respondentsA,ii])
		
			ii += 1

class MPCSimulatorNews(MPCSimulator):
	def __init__(self, params, income, grids, futureShockIndices, 
		currentShockIndices, periodsUntilShock=1, simPeriods=1):
		self.futureShockIndices = futureShockIndices
		self.periodsUntilShock = periodsUntilShock

		super().__init__(params, income, grids, currentShockIndices)
		
		self.nCols = len(futureShockIndices) + 1
		self.news = True
		self.T = simPeriods

	def initialize(self, cSwitchingPolicies, inactionRegions,
		finalStates):
		self.policies = cSwitchingPolicies
		self.inactions = inactionRegions
		self.finalStates = finalStates

		ymin = self.income.ymin + self.p.govTransfer
		self.borrowLims = []
		for ishock in self.futureShockIndices:
			shock = self.p.MPCshocks[ishock]
			self.borrowLims.append(functions.computeAdjBorrLims(shock,
				ymin, self.p.borrowLim, self.p.R, self.periodsUntilShock))
		self.borrowLims.append([self.p.borrowLim] * self.periodsUntilShock)

		self.initialize_variables()
		self.initialized = True

	def updateCashGrids(self):
		if self.t < self.periodsUntilShock + 1:
			self.xgridCurr = [None] * self.nCols
			self.borrowLimsCurr = [None] * self.nCols
			for col in range(self.nCols):
				self.borrowLimsCurr[col] = self.borrowLims[col].pop()
				self.xgridCurr[col] = self.grids.genAdjustedXGrid(self.borrowLimsCurr[col])
		else:
			for col in range(self.nCols):
				self.borrowLimsCurr[col] = self.p.borrowLim
				self.xgridCurr[col] = np.asarray(self.grids.x_flat)

	def updatePolicies(self):
		remPeriods = np.maximum(self.periodsUntilShock + 1 - self.t, 0)

		self.cSwitchingPolicy = self.policies[:,:,:,:,:,remPeriods]
		self.inactionRegion = self.inactions[:,:,:,:,:,remPeriods]

	def applyShocks(self):
		super().applyShocks()

		if self.t == self.periodsUntilShock + 1:
			col = 0
			for ishock in self.futureShockIndices:
				self.xsim[:,col] = np.asarray(self.xsim[:,col]) + self.p.MPCshocks[ishock]
				col += 1

	def initialize_results(self):
		# statistics to compute very period
		self.results = pd.Series(name=self.p.name)
		rows = []
		for ishock in range(6):
			shock = self.p.MPCshocks[ishock]
			for quarter in range(1,self.T+1):
				rows.append(f'E[Q{quarter} MPC] out of news of {shock} shock in {self.periodsUntilShock} quarter(s)')
				rows.append(f'E[Q{quarter} MPC | MPC > 0] out of news of {shock} shock in {self.periodsUntilShock} quarter(s)')
				rows.append(f'Median(Q{quarter} MPC | MPC > 0) out of news of {shock} shock in {self.periodsUntilShock} quarter(s)')

				rows.append(f'P(Q{quarter} MPC < 0) for news of {shock} shock in {self.periodsUntilShock} quarter(s)')
				rows.append(f'P(Q{quarter} MPC = 0) for news of {shock} shock in {self.periodsUntilShock} quarter(s)')
				rows.append(f'P(Q{quarter} MPC > 0) for news of {shock} shock in {self.periodsUntilShock} quarter(s)')

		for row in rows:
			self.results[row] = np.nan

		self.mpcs = dict()
		self.switched = np.zeros((self.nSim,self.nCols),dtype=int)

	def computeTransitionStatistics(self):
		ii = 0
		for ishock in self.futureShockIndices:
			futureShock = self.p.MPCshocks[ishock]
			rowQuarterly = f'E[Q{self.t} MPC] out of news of {futureShock} shock in {self.periodsUntilShock} quarter(s)'
			rowQuarterlyCond = f'E[Q{self.t} MPC | MPC > 0] out of news of {futureShock} shock in {self.periodsUntilShock} quarter(s)'
			rowQuarterlyCondMedian = f'Median(Q{self.t} MPC | MPC > 0) out of news of {futureShock} shock in {self.periodsUntilShock} quarter(s)'

			csimQuarter = np.asarray(self.csim[:,ii])

			# quarterly mpcs
			allMPCS = (csimQuarter - np.asarray(self.csim[:,self.nCols-1])
					) / self.p.MPCshocks[ishock]

			if self.t == 1:
				self.mpcs[ishock] = allMPCS

			self.results[rowQuarterly] = allMPCS.mean()
			self.results[rowQuarterlyCond] = allMPCS[allMPCS>0].mean()
			self.results[rowQuarterlyCondMedian] = np.median(allMPCS[allMPCS>0]);

			# fraction of respondents in this quarter
			respondentsQ = allMPCS > 0
			respondentsQ_neg = allMPCS < 0
			nonRespondents = (allMPCS == 0)
	
			rowRespondentsQuarterly = f'P(Q{self.t} MPC < 0) for news of {futureShock} shock in {self.periodsUntilShock} quarter(s)'
			self.results[rowRespondentsQuarterly] = respondentsQ_neg.mean()
			rowRespondentsQuarterly = f'P(Q{self.t} MPC = 0) for news of {futureShock} shock in {self.periodsUntilShock} quarter(s)'
			self.results[rowRespondentsQuarterly] = nonRespondents.mean()
			rowRespondentsQuarterly = f'P(Q{self.t} MPC > 0) for news of {futureShock} shock in {self.periodsUntilShock} quarter(s)'
			self.results[rowRespondentsQuarterly] = respondentsQ.mean()
		
			ii += 1

class MPCSimulatorNews_Loan(MPCSimulatorNews):
	def initialize_results(self):
		# statistics to compute very period
		self.results = pd.Series(name=self.p.name)
		rows = []
		for ishock in range(3):
			shock = self.p.MPCshocks[ishock]
			for quarter in range(1,2):
				rows.append(f'E[Q{quarter} MPC] out of {-shock} loan for {self.periodsUntilShock} quarter(s)')
				rows.append(f'E[Q{quarter} MPC | MPC > 0] out of {-shock} loan for {self.periodsUntilShock} quarter(s)')
				rows.append(f'Median(Q{quarter} MPC | MPC > 0) out of {-shock} loan for {self.periodsUntilShock} quarter(s)')

		for ishock in range(3):
			shock = self.p.MPCshocks[ishock]
			rows.append(f'P(Q1 MPC < 0) for {-shock} loan for {self.periodsUntilShock} quarter(s)')
			rows.append(f'P(Q1 MPC = 0) for {-shock} loan for {self.periodsUntilShock} quarter(s)')
			rows.append(f'P(Q1 MPC > 0) for {-shock} loan for {self.periodsUntilShock} quarter(s)')

		for row in rows:
			self.results[row] = np.nan

		self.mpcs = dict()
		self.switched = np.zeros((self.nSim,self.nCols),dtype=int)
		self.initialized = True

	def computeTransitionStatistics(self):
		quarter = 1
		ii = 0
		for ishock in self.futureShockIndices:
			loanSize = -self.p.MPCshocks[ishock]
			rowQuarterly = f'E[Q{quarter} MPC] out of {loanSize} loan for {self.periodsUntilShock} quarter(s)'
			rowQuarterlyCond = f'E[Q{quarter} MPC | MPC > 0] out of {loanSize} loan for {self.periodsUntilShock} quarter(s)'
			rowQuarterlyCondMedian = f'Median(Q{quarter} MPC | MPC > 0) out of {loanSize} loan for {self.periodsUntilShock} quarter(s)'

			csimQuarter = np.asarray(self.csim[:,ii])

			# quarterly mpcs
			allMPCS = (csimQuarter - np.asarray(self.csim[:,self.nCols-1])
					) / loanSize

			if self.t == 1:
				self.mpcs[ishock] = allMPCS

			self.results[rowQuarterly] = allMPCS.mean()
			self.results[rowQuarterlyCond] = allMPCS[allMPCS>0].mean()
			self.results[rowQuarterlyCondMedian] = np.median(allMPCS[allMPCS>0]);

			# fraction of respondents in this quarter
			respondentsQ = allMPCS > 0
			respondentsQ_neg = allMPCS < 0
			nonRespondents = allMPCS == 0
			if self.t == 1:
				rowRespondentsQuarterly = f'P(Q1 MPC < 0) for {loanSize} loan for {self.periodsUntilShock} quarter(s)'
				self.results[rowRespondentsQuarterly] = respondentsQ_neg.mean()
				rowRespondentsQuarterly = f'P(Q1 MPC = 0) for {loanSize} loan for {self.periodsUntilShock} quarter(s)'
				self.results[rowRespondentsQuarterly] =  nonRespondents.mean()
				rowRespondentsQuarterly = f'P(Q1 MPC > 0) for {loanSize} loan for {self.periodsUntilShock} quarter(s)'
				self.results[rowRespondentsQuarterly] = respondentsQ.mean()

			ii += 1