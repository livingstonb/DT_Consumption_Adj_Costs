import numpy as np
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

		self.dims = (params.nx, params.nc, params.nh, params.nyP)
		self.dims_yT = self.dims + (self.p.nyT,)

		self.xp_s = None
		self.xp_s_T = None

		self.nextModel = nextModel

		# create x, c vectors of length (nx*nc) for interpolator
		self.xgridLong = self.grids.x['matrix'][:,:,1].reshape((-1,1))

		self.stats = dict()

	def solve(self):
		# guess conditional on not switching consumption
		valueNoSwitch = functions.utility(self.p.riskaver,self.grids.c['matrix'])

		# guess conditional on switching
		valueSwitch = valueNoSwitch.max(axis=1,keepdims=True)

		# value function
		value = np.maximum(valueNoSwitch,valueSwitch)

		# update value function of not switching
		# not this simple, need to evaluate EmaxV at x' which means summing over
		# all yT as well...
		valueNoSwitch = functions.utility(self.p.riskaver,self.grids.c['matrix']) \
			+ self.p.timeDiscount * np.matmul(self.Emat,value.reshape((-1,1)))
		valueNoSwitch = valueNoSwitch.reshape(self.grids.matrixDim)



		# update value function of switching
		# first get Emax
		EmaxVals = self.Emat * np.maximum(valueSwitch-self.p.adjustCost,valueNoSwitch)
		EmaxInterp = []
		for iP in range(self.p.nyP):
			# interpGrids = (	self.grids.x['matrix'][:,0,0,iP].flatten(),
			# 				self.grids.c['vec'],)
			# if self.nz > 1:
			# 	interpGrids = interpGrids + (self.grids.z['vec'],)
			# interpObj = RegularGridInterpolator(interpGrids,EmaxVals[:,:,:,iP])
			# EmaxInterp.append(interpObj)

			gridPts = self.grids.x['matrix'][:,0,0,iP]
			for ix in range(self.p.nx):
				# interpolate Emax at 


		utilSwitch = np.max(utilNoSwitch,axis=2).reshape((self.p.nx,1,self.p.nyP))
		utilSwitch = np.repeat(utilSwitch,self.p.nc,axis=1) - self.p.adjustCost

		it = 0
		cdiff = 1e5
		while (it < self.p.maxIterVFI) and (cdiff > self.p.tolIterVFI):
			it += 1

			

		self.compute_xprime_s()
		self.compute_Emat()

		self.iterateBackward(conGuess)


	def makePolicyGuess(self):
		# to avoid a degenerate guess, adjust for low r...
		returns = self.p.r + 0.001 * (self.p.r<0.0001)
		conGuess = returns * self.grids.x['matrix']
		return conGuess

	def makeValueGuess(self, conGuess):
		vGuess = functions.utility(self.p.riskAver,conGuess)
		return vGuess

	def iterateBackward(self, conGuess):
		it = 0
		cdiff = 1e5
		con = conGuess

		dimTuple = (self.p.nx, self.p.nc, self.p.nyP)

		while (it < self.p.maxIterEGP) and (cdiff > self.p.tolIterEGP):
			it += 1

			# c(x)
			con = con.reshape(dimTuple)

			# c(x')
			con_xp = self.get_c_xprime(con).reshape((-1,self.p.nyT))

			# MUC in current period, from Euler equation
			muc_s = self.getMUC(con_xp)

			# c(s)
			con_s = functions.marginalUtility(self.p.riskAver,muc_s)

			# x(s) = s + c(s)
			x_s = self.grids.x['matrix'] + con_s

			# interpolate from x(s) to get s(x)
			sav = self.getSavingPolicy(x_s)

			conUpdate = self.grids.x['matrix'] - sav

			cdiff = np.max(np.abs(conUpdate.flatten()-con.flatten()))
			if np.mod(it,50) == 0:
				print(f'    EGP iteration {it}, distance = {cdiff}')

			con = conUpdate

		if cdiff > self.p.tolIterEGP:
			raise Exception ('EGP failed to converge')

	def compute_xprime_s(self):
		# find xprime as a function of saving
		income_x = np.kron(self.income.ymat,np.ones((self.p.nx*self.p.nc,1)))
		income_x = income_x.reshape(self.dims_yT)

		self.xp_s = self.p.R * self.grids.s['matrix'][...,None] \
			+ income_x + self.nextMPCShock

		self.xp_s_T = self.xp_s.reshape((-1,self.p.nyT))

	def compute_Emat(self):
		self.Emat = np.kron(self.income.yPtrans,np.eye(self.p.nx*self.p.nc*self.p.nz))

	def get_c_xprime(self,con):
		con_xprime = np.zeros(self.dims_yT)

		for iy in range(self.p.nyP):
			for ic in range(self.p.nc):
				# coninterp = RegularGridInterpolator(
				# 	(self.grids.x['matrix'][:,ic,iy]),
				# 	con[:,ic,iy],method='linear',bounds_error=True)

				# con_xprime[:,ic,iy] = coninterp(self.xp_s[:,ic,iy])
				con_xprime[:,ic,iy,:] = np.interp(self.xp_s[:,ic,iy,:],
					self.grids.x['matrix'][:,ic,iy],
					con[:,ic,iy])

		return con_xprime

	def getMUC(self, c_xp):
		# first get MUC of consumption next period over all states
		mucnext = functions.marginalUtility(self.p.riskAver,c_xp)
		Emucnext = np.matmul(self.Emat,
						np.matmul(mucnext,self.income.yTdist)).reshape(
						self.dims)
		muc_s = (1 - self.p.deathProb) * Emucnext
		return muc_s

	def getSavingPolicy(self,x_s):
		"""
		Finds s(x), the saving policy function on the cash-on-hand
		grid.
		"""
		sav = np.zeros(self.dims)
		for iy in range(self.p.nyP):
			for ic in range(self.p.nc):
				sav[:,ic,iy] = np.interp(	self.grids.x['matrix'][:,ic,iy],
											x_s[:,ic,iy],
											self.grids.s['vec'][:,0])

		# impose borrowing limit
		sav = np.maximum(sav,self.p.borrowLim)
		return sav

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
			self.responded[:,ii] = self.responded[:,ii] || respondentsQ

			# fraction of respondents (annual)
			if self.t == 4:
				rowRespondentsAnnual = f'Fraction with Annual MPC > 0 for shock of {self.p.MPCshocks[ishock]}'
				self.mpcs[rowRespondentsAnnual] = np.self.responded[:,ii].mean()
		
			ii += 1
