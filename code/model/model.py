import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import RegularGridInterpolator
import numpy.matlib as matlib
from misc import functions
import pandas as pd

class Model:
	def __init__(self, params, income, grids, nextModel=None, nextMPCShock=0):
		self.p = params
		self.grids = grids
		self.income = income

		self.nextMPCShock = nextMPCShock

		self.dims = (params.nx, params.nc, params.nz, params.nyP)
		self.dims_yT = self.dims + (self.p.nyT,)

		self.xp_s = None
		self.xp_s_T = None

		self.nextModel = nextModel

		self.stats = dict()

	def solve(self):
		# interpolant for updating no-switch case
		self.constructInterpolantForV()

		# guess conditional on not switching consumption
		self.valueNoSwitch = functions.utility(self.p.riskAver,self.grids.c['matrix'])

		# guess conditional on switching
		self.valueSwitch = self.valueNoSwitch.max(axis=1,keepdims=True)

		# update value function
		self.updateValueFunction()

		iteration = 0
		while True:

			# update value function of not switching
			self.updateValueNoSwitch()

			# update value function
			self.updateValueFunction()

			# update value function of switching
			self.updateValueSwitch()

			Vprevious = self.valueFunction.copy()
			self.updateValueFunction()

			distance = np.abs(self.valueFunction - Vprevious).flatten().sum()
			if distance < self.p.tol:
				break
			elif iteration > self.p.maxIters:
				raise Exception(f'No convergence after {iteration+1} iterations...')

			iteration += 1

		
		print('done')


		it = 0
		cdiff = 1e5
		while (it < self.p.maxIterVFI) and (cdiff > self.p.tolIterVFI):
			it += 1

	def constructInterpolantForV(self):
		yTvec = self.income.yTgrid.reshape((1,-1))
		yTdistvec = self.income.yTdist.reshape((1,-1))

		interpMat = np.zeros((self.p.nx,self.p.nc,self.p.nz,self.p.nyP,self.p.nx,self.p.nc,self.p.nz,self.p.nyP))
		for iyP1 in range(self.p.nyP):
			xgrid = self.grids.x['wide'][:,0,0,iyP1]
			for iyP2 in range(self.p.nyP):
				# compute E_{yT}[V()]
				yP = self.income.yPgrid[iyP2]
				PyP1yP2 = self.income.yPtrans[iyP1,iyP2]
				for ic in range(self.p.nc):
					xprime = self.p.R * (xgrid[:,None] - self.grids.c['vec'][ic]) + yP * yTvec
					interpWithyT = functions.interpolateTransitionProbabilities2D(xgrid,xprime)
					newBlock = PyP1yP2 * np.dot(yTdistvec,interpWithyT)
					for iz in range(self.p.nz):
						# first dim of interpMat is x
						# fourth dimension is x'
						
						# take expectation wrt yT
						interpMat[:,ic,iz,iyP1,:,ic,iz,iyP2] = newBlock

		interpMat = interpMat.reshape((self.p.nx*self.p.nc*self.p.nz*self.p.nyP,
			self.p.nx*self.p.nc*self.p.nz*self.p.nyP))

		self.interpMat = interpMat

	def updateValueFunction(self):
		self.valueFunction = np.maximum(self.valueNoSwitch,self.valueSwitch)

	def updateValueNoSwitch(self):
		self.valueNoSwitch = functions.utility(self.p.riskAver,self.grids.c['matrix']) \
			+ self.p.timeDiscount * np.matmul(self.interpMat,self.valueFunction.reshape((-1,1))
				).reshape(self.grids.matrixDim)

		self.valueNoSwitch = self.valueNoSwitch.reshape(self.grids.matrixDim)

	def updateValueSwitch(self):
		# compute EMAX
		EMAX = np.matmul(self.interpMat,self.valueFunction.reshape((-1,1))
				).reshape(self.grids.matrixDim)

		util = functions.utility(self.p.riskAver,self.grids.c['vec']).flatten()

		self.valueSwitch = np.zeros((self.p.nx,1,self.p.nz,self.p.nyP))
		for iyP in range(self.p.nyP):
			EMAXInterpolant = RegularGridInterpolator(
				(self.grids.x['wide'][:,0,0,iyP],self.grids.c['vec'].flatten(),self.grids.z['vec'].flatten()),EMAX[:,:,:,iyP],bounds_error=False)
			print(f'income value {iyP}')

			for ix in range(self.p.nx):
				xval = self.grids.x['wide'][ix,0,0,iyP]
				print(xval)
				print(f'asset value {ix}')

				for iz in range(self.p.nz):
					if ix == 0:
						x0 = xval / 2
					iteratorFn = lambda c: self.findValueFromSwitching(c,EMAXInterpolant,util,xval,iz)
					self.valueSwitch[ix,0,iz,iyP] = minimize(iteratorFn,x0,method='SLSQP',bounds=((1e-5,xval),)).x

	def findValueFromSwitching(self, cSwitch, EMAXInterpolant, util, xval, iz):
		# utilSumVec = interpolateTransitionProbabilities(self.grids.c['vec'],cSwitch)
		utility = np.interp(cSwitch,self.grids.c['vec'].flatten(),util)
		EV = EMAXInterpolant((xval,cSwitch,iz))

		return - utility - self.p.timeDiscount * EV


	def makePolicyGuess(self):
		# to avoid a degenerate guess, adjust for low r...
		returns = self.p.r + 0.001 * (self.p.r<0.0001)
		conGuess = returns * self.grids.x['matrix']
		return conGuess

	def makeValueGuess(self, conGuess):
		vGuess = functions.utility(self.p.riskAver,conGuess)
		return vGuess


class Simulator:
	def __init__(self, params, income, grids, policies, simPeriods):
		self.p = params
		self.income = income
		self.grids = grids
		self.policies = policies

		self.nCols = 1

		self.periodsBeforeRedraw = np.min(simPeriods,50)

		self.nSim = params.nSim
		self.T = simPeriods
		self.t = 1
		self.randIndex = 0

		self.initialized = False

	def simulate(self):
		if not self.initialized:
			raise Exception ('Simulator not initialized')

		self.makeRandomDraws()
		while self.t <= self.T:

			if self.t > 1:
				# assets and income should already be initialized
				self.updateAssets()
				self.updateIncome()

			self.updateCash()
			self.updateConsumption()

			self.solveDecisions()

			self.computeTransitionStatistics()

			if self.randIndex < self.periodsBeforeRedraw - 1:
				# use next column of random numbers
				self.randIndex += 1
			else:
				# need to redraw
				self.randIndex = 0
				self.makeRandomDraws()

			t += 1

	def makeRandomDraws(self):
		self.yPrand = np.random(size=(self.nSim,self.periodsBeforeRedraw))
		self.yTrand = np.random(size=(self.nSim,self.periodsBeforeRedraw))

		if self.p.deathProb > 0:
			self.deathrand = np.random(size=(self.nSim,self.periodsBeforeRedraw))

	def updateIncome(self):
		self.yPind = np.argmax(self.yPrand[:,self.randIndex]
					<= self.income.yPcumtrans[self.yPind,:],
					axis=1)
		self.yTind = np.argmax(self.yTrand[:,self.randIndex]
								<= self.income.yTcumdistT, axis=1)

		yPsim = self.income.yPgrid[self.yPind]
		yTsim = self.income.yTgrid[self.yTind]
		self.ysim = (yPsim * yTsim).reshape((-1,1))

	def updateCash(self):
		self.xsim = self.asim + self.ysim

	def updateAssets(self):
		if not self.Bequests:
			self.asim[self.deathrand[:,self.randIndex]<self.p.deathProb,:] = 0

		self.asim = self.p.R * self.ssim

class EquilbriumSimulator(Simulator):
	def __init__(self, params, income, grids, policies):
		Simulator.__init__(self,params,income,grids,policies,params.tSim)

	def simulate(self):
		Simulator.simulate(self)
		self.computeEquilibriumStatistics()

	def initialize(self):
		self.asim = self.p.wealthTarget * np.ones((self.nSim,self.nCols))
		self.csim = np.ones((self.nSim,self.nCols))

		# statistics to compute every period
		self.aMean = np.zeros((self.T,)) # mean assets
		self.aVariance = np.zeros((self.T,)) # variance of assets

		self.initialized = True

	def computeTransitionStatistics(self):
		"""
		This method computes statistics that can be
		used to evaluate convergence to the equilibrium
		distribution.
		"""
		self.aMean[self.t-1] = np.mean(self.asim,axis=0)
		self.aVariance[self.t-1] = np.var(self.asim,axis=0)

	def computeEquilibriumStatistics(self):
		# fraction with wealth < epsilon
		self.stats['constrained'] = []
		for threshold in self.p.wealthConstraints:
			constrained = np.mean(self.asim <= threshold)
			self.stats['constrained'].append(constrained)

		# wealth percentiles
		self.stats['wpercentiles'] = []
		for pctile in self.p.wealthPercentiles:
			value = np.percentile(self.asim,pctile)
			self.stats['wpercentiles'].append(value)

		# top shares
		pctile90 = np.percentile(self.asim,0.9)
		pctile99 = np.percentile(self.asim,0.99)
		self.results['top10share'] = \
			np.sum(self.asim[self.asim >= pctile90]) / self.asim.sum()
		self.results['top10share'] = \
			np.sum(self.asim[self.asim >= pctile99]) / self.asim.sum()

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
		self.responded = np.zeros((self.nSim,numel(self.shockIndices)),dtype=bool)

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
