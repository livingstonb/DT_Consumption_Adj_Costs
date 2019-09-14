import numpy as np
cimport numpy as np
import pandas as pd
from misc cimport functions

cdef class Simulator:
	cdef:
		object p, income, grids
		double[:,:,:,:] valueSwitch, valueNoSwitch, cSwitchingPolicy
		int nCols
		int periodsBeforeRedraw
		int nSim, t, T, randIndex
		bint initialized
		long[:] yPind, zind
		double[:,:] ysim, csim, xsim, asim
		double[:,:] deathrand, yPrand, yTrand
		public dict results

	def __init__(self, params, income, grids, model, simPeriods):
		self.p = params
		self.income = income
		self.grids = grids

		self.results = {}

		self.cSwitchingPolicy = model.cSwitchingPolicy
		self.valueSwitch = model.valueSwitch
		self.valueNoSwitch = model.valueNoSwitch

		self.nCols = 1

		self.periodsBeforeRedraw = np.minimum(simPeriods,50)

		self.nSim = params.nSim
		self.T = simPeriods
		self.t = 1
		self.randIndex = 0

		self.initialized = False

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

	def makeRandomDraws(self):
		self.yPrand = np.random.random(size=(self.nSim,self.periodsBeforeRedraw))
		self.yTrand = np.random.random(size=(self.nSim,self.periodsBeforeRedraw))

		if (self.p.deathProb > 0) and self.p.Bequests:
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

		if not self.p.Bequests:
			self.asim[self.deathrand[:,self.randIndex]<self.p.deathProb,:] = 0


	def solveDecisions(self):
		cdef:
			long i, iyP, iz
			list conIndices, conSwitchIndices, conWeights, conSwitchWeights
			list xIndices, xWeights
			double[:] cgrid
			bint switch
			double valueSwitch, valueNoSwitch, cSwitch
			double consumption, cash

		cgrid = self.grids.c.flat
		for col in range(self.nCols):
			for i in range(self.nSim):

				iyP = self.yPind[i]
				iz = self.zind[i]

				consumption = self.csim[i,col]
				cash = self.xsim[i,col]

				xIndices, xWeights = functions.interpolate1D(
					self.grids.x.flat, cash)

				if consumption > cash:
					# forced to switch consumption
					switch = True
				else:
					# check if switching is optimal
					conIndices, conWeights = functions.interpolate1D(cgrid, consumption)

					valueSwitch = xWeights[0] * self.valueSwitch[xIndices[0],0,iz,iyP] \
						+ xWeights[1] * self.valueSwitch[xIndices[1],0,iz,iyP]

					valueNoSwitch = xWeights[0] * conWeights[0] * self.valueNoSwitch[xIndices[0],conIndices[0],iz,iyP] \
						+ xWeights[1] * conWeights[0] * self.valueNoSwitch[xIndices[1],conIndices[0],iz,iyP] \
						+ xWeights[0] * conWeights[1] * self.valueNoSwitch[xIndices[0],conIndices[1],iz,iyP] \
						+ xWeights[1] * conWeights[1] * self.valueNoSwitch[xIndices[1],conIndices[1],iz,iyP]

					switch = valueSwitch - self.p.adjustCost > valueNoSwitch

				if switch:
					self.csim[i,col] = xWeights[0] * self.cSwitchingPolicy[xIndices[0],0,iz,iyP] \
							+ xWeights[1] * self.cSwitchingPolicy[xIndices[1],0,iz,iyP]

cdef class EquilibriumSimulator(Simulator):

	def __init__(self, params, income, grids, model):
		super().__init__(params,income,grids,model,params.tSim)
		self.initialize()

	def simulate(self):
		super().simulate()
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

		# statistics to compute every period
		self.results['mean assets'] = np.zeros((self.T,))
		self.results['variance of assets'] = np.zeros((self.T,))

		self.initialized = True

	def computeTransitionStatistics(self):
		"""
		This method computes statistics that can be
		used to evaluate convergence to the equilibrium
		distribution.
		"""
		self.results['mean assets'][self.t-1] = np.mean(self.asim,axis=0)
		self.results['variance of assets'][self.t-1] = np.var(self.asim,axis=0)

	def computeEquilibriumStatistics(self):
		# fraction with wealth < epsilon
		self.results['constrained'] = []
		for threshold in self.p.wealthConstraints:
			constrained = np.mean(np.asarray(self.asim) <= threshold)
			self.results['constrained'].append(constrained)

		# wealth percentiles
		self.results['wpercentiles'] = []
		for pctile in self.p.wealthPercentiles:
			value = np.percentile(self.asim,pctile)
			self.results['wpercentiles'].append(value)

		# top shares
		pctile90 = np.percentile(self.asim,90)
		pctile99 = np.percentile(self.asim,99)
		asimNumpy = np.asarray(self.asim)
		self.results['top10share'] = \
			asimNumpy[asimNumpy>=pctile90].sum() / asimNumpy.sum()
		self.results['top1share'] = \
			asimNumpy[asimNumpy>=pctile99].sum() / asimNumpy.sum()

class MPCSimulator(Simulator):
	def __init__(self, params, income, grids, policies, shockIndices):
		Simulator.__init__(self,params,income,grids,policies,4)
		self.nCols = len(shockIndices) + 1
		self.shockIndices = shockIndices

	def simulate(self):
		Simulator.simulate(self)

	def initialize(self, initialAssets, initialCon, initialyPind, initialyTind):
		self.asim = np.repeat(initialAssets,self.nCols,axis=1)
		self.csim = np.repeat(initialCon,self.nCols,axis=1)
		self.yPind = initialyPind
		self.yTind = initialyTind
		yPsim = self.income.yPgrid[self.yPind]
		yTsim = self.income.yTgrid[self.yTind]
		self.ysim = (yPsim * yTsim).reshape((-1,1))
		self.responded = np.zeros((self.nSim,len(self.shockIndices)),dtype=bool)

		# statistics to compute very period
		self.mpcs = pd.Series(name=self.p.name)
		for ishock in range(6):
			for quarter in range(1,5):
				row = f'Quarter {quarter} MPC out of {self.p.MPCshocks[ishock]}'
				self.mpcs[row] = np.nan

			row = f'Annual MPC out of {self.p.MPCshocks[ishock]}'
			self.mpcs[row] = np.nan

		for ishock in range(6):
			row = f'Fraction with Q1 MPC > 0 for shock of {self.p.MPCshocks[ishock]}'
			self.mpcs[row] = np.nan
			row = f'Fraction with Annual MPC > 0 for shock of {self.p.MPCshocks[ishock]}'
			self.mpcs[row] = np.nan

	def computeTransitionStatistics(self):
		ii = 0
		for ishock in self.shockIndices:
			rowQuarterly = f'Quarter {self.t} MPC out of {self.p.MPCshocks[ishock]}'
			rowAnnual = f'Annual MPC out of {self.p.MPCshocks[ishock]}'

			# quarterly mpcs
			self.mpcs[rowQuarterly] = np.mean(
				(self.csim[:,ii] - self.csim[:,self.nCols]) / self.p.MPCshocks[ishock])

			# add quarterly mpcs to annual mpcs
			if self.t == 1:
				self.mpcs[rowAnnual] = self.mpcs[rowQuarterly]
			elif self.t > 1:
				self.mpcs[rowAnnual] += self.mpcs[rowQuarterly]

			# fraction of respondents in this quarter
			respondentsQ = (self.csim[:,ii] - self.csim[:,self.nCols]) / self.p.MPCshocks[ishock] > 0
			if self.t == 1:
				rowRespondentsQuarterly = f'Fraction with Q1 MPC > 0 for shock of {self.p.MPCshocks[ishock]}'
				self.mpcs[rowRespondentsQuarterly] = respondentsQ.mean()

			# update if some households responded this period but not previous periods
			self.responded[:,ii] = self.responded[:,ii] or respondentsQ

			# fraction of respondents (annual)
			if self.t == 4:
				rowRespondentsAnnual = f'Fraction with Annual MPC > 0 for shock of {self.p.MPCshocks[ishock]}'
				self.mpcs[rowRespondentsAnnual] = np.self.responded[:,ii].mean()
		
			ii += 1
