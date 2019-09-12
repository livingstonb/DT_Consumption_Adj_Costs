import numpy as np
cimport numpy as np

from scipy.optimize import minimize
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp1d
import numpy.matlib as matlib
from build import functions
from build cimport functions
import pandas as pd

cdef class Model:
	cdef:
		object p, grids, income
		double nextMPCShock
		tuple dims, dims_yT
		dict nextModel, stats
		# np.ndarray[np.float64_t, ndim=4] valueNoSwitch, valueSwitch, valueFunction
		# np.ndarray[np.float64_t, ndim=2] interpMat
		double[:,:,:,:] valueNoSwitch, valueSwitch, valueFunction
		double[:,:] interpMat

	def __init__(self, params, income, grids, nextModel=None, nextMPCShock=0):

		self.p = params
		self.grids = grids
		self.income = income

		self.nextMPCShock = nextMPCShock

		self.dims = (params.nx, params.nc, params.nz, params.nyP)
		self.dims_yT = self.dims + (self.p.nyT,)

		self.nextModel = nextModel

		self.stats = dict()

	def solve(self):
		# construct interpolant for EMAX
		self.constructInterpolantForV()

		# make initial guess for value function
		valueGuess = functions.utility(self.p.riskAver,self.grids.c['matrix']
			) / (1 - self.p.timeDiscount * (1 - self.p.deathProb * self.p.Bequests))

		# subtract the adjustment cost for states with c > x
		self.valueFunction = valueGuess - self.p.adjustCost * self.grids.mustSwitch

		iteration = 0
		while True:
			Vprevious = self.valueFunction.copy()

			# update value function of not switching
			self.updateValueNoSwitch()

			# update value function of switching
			self.updateValueSwitch()

			# compute V = max(VSwitch,VNoSwitch)
			self.updateValueFunction()

			distance = np.abs(np.asarray(self.valueFunction) - np.asarray(Vprevious)).flatten().max()
			if distance < self.p.tol:
				print('Value function converged')
				break
			elif iteration > self.p.maxIters:
				raise Exception(f'No convergence after {iteration+1} iterations...')

			if np.mod(iteration,1) == 0:
				print(f'    Iteration {iteration}, norm of |V1-V| = {distance}')

			iteration += 1

	def constructInterpolantForV(self):
		"""
		This method constructs an interpolant ndarray 'interpMat' such that
		interpMat @ valueFunction(:) = E[V]. That is, the expected continuation 
		value of being in state (x_i,ctilde,z_k,yP_k) and choosing to consume c_j, 
		the (i,j,k,l)-th element of the matrix product of interpMat and 
		valueFunction(:) is the quantity below:

		E[V(R(x_i-c_j)+yP'yT', c_j, z_k, yP')|x_i,c_j,z_k,yP_l]

		where the expectation is taken over yP' and yT'.

		Output is stored in self.interpMat.
		"""
		cdef:
			int iyP1, iyP2, ic, iz
			double yP, PyP1yP2
			np.ndarray[np.float64_t, ndim=8] interpMat
			np.ndarray[np.float64_t, ndim=3] newBlock

		yTvec = self.income.yTgrid.reshape((1,-1))
		yTdistvec = self.income.yTdist.reshape((1,-1))

		interpMat = np.zeros((self.p.nx,self.p.nc,self.p.nz,self.p.nyP,self.p.nx,self.p.nc,self.p.nz,self.p.nyP))
		for iyP1 in range(self.p.nyP):
			xgrid1 = self.grids.x['wide'][:,0,0,iyP1]
			for iyP2 in range(self.p.nyP):
				# compute E_{yT}[V()]
				xgrid2 = self.grids.x['wide'][:,0,0,iyP2]
				yP = self.income.yPgrid[iyP2]
				PyP1yP2 = self.income.yPtrans[iyP1,iyP2]
				for ic in range(self.p.nc):
					xprime = self.p.R * (xgrid1[:,None] - self.grids.c['vec'][ic]) + yP * yTvec
					interpWithyT = functions.interpolateTransitionProbabilities2D(xgrid2,xprime)
					newBlock = PyP1yP2 * np.dot(yTdistvec,interpWithyT)
					for iz in range(self.p.nz):
						interpMat[:,ic,iz,iyP1,:,ic,iz,iyP2] = newBlock

		self.interpMat = interpMat.reshape((self.p.nx*self.p.nc*self.p.nz*self.p.nyP,
			self.p.nx*self.p.nc*self.p.nz*self.p.nyP))

	def updateValueFunction(self):
		"""
		This method updates self.valueFunction by finding max(valueSwitch,valueNoSwitch),
		where valueSwitch is used wherever c > x in the state space.
		"""
		self.valueFunction = np.where(self.grids.mustSwitch,
			np.asarray(self.valueSwitch)-self.p.adjustCost,
			np.maximum(self.valueNoSwitch,np.asarray(self.valueSwitch)-self.p.adjustCost)
			)

	def updateValueNoSwitch(self):
		"""
		Updates self.valueNoSwitch via valueNoSwitch(c) = u(c) + beta * E[V(c)]
		"""
		self.valueNoSwitch = functions.utility(self.p.riskAver,self.grids.c['matrix']) \
			+ self.p.timeDiscount * (1 - self.p.deathProb * self.p.Bequests) \
			* np.matmul(self.interpMat,np.reshape(self.valueFunction,(-1,1))
				).reshape(self.grids.matrixDim)

		self.valueNoSwitch = np.reshape(self.valueNoSwitch,self.grids.matrixDim)

	def updateValueSwitch(self):
		"""
		Updates self.valueSwitch by first approximating EMAX(x,c,z,yP), defined as
		E[V(R(x-c)+yP'yT', c, z, yP')|x,c,z,yP]. This approximation is done by
		interpolating the sums over income transitions using interpMat. Then a
		maximization is done over u(c) + beta * EMAX at each point in the
		(x,z,yP)-space by interpolating u(c) and EMAX(c) at each iteration.
		"""
		cdef:
			int iyP, ix, iz, ii
			double xval, maxAdmissibleC, goldenRatio, goldenRatioSq
			np.ndarray[np.float64_t, ndim=1] candidateC, funVals
			double[:] cgrid, util, em
			np.ndarray[np.float64_t, ndim=4] EMAX
			tuple cBounds

		# compute EMAX
		EMAX = np.matmul(self.interpMat,np.reshape(self.valueFunction,(-1,1))
				).reshape(self.grids.matrixDim)

		util = functions.utility(self.p.riskAver,self.grids.c['vec']).flatten()
		cgrid = self.grids.c['vec'].flatten()

		goldenRatio = (np.sqrt(5) + 1) / 2
		goldenRatioSq = goldenRatio ** 2

		self.valueSwitch = np.zeros((self.p.nx,1,self.p.nz,self.p.nyP))
		for iyP in range(self.p.nyP):
			# EMAXInterpolant = RegularGridInterpolator(
			# 	(self.grids.x['wide'][:,0,0,iyP],self.grids.c['vec'].flatten(),self.grids.z['vec'].flatten()),EMAX[:,:,:,iyP],bounds_error=False)

			for ix in range(self.p.nx):
				xval = self.grids.x['wide'][ix,0,0,iyP]
				maxAdmissibleC = np.minimum(xval,self.p.cMax)
				cBounds = ((self.p.cMin,maxAdmissibleC),)
				bounds = [	(self.p.cMin,maxAdmissibleC/4),
							(maxAdmissibleC/4,maxAdmissibleC/2),
							(maxAdmissibleC/2,3*maxAdmissibleC/4),
							(3*maxAdmissibleC/4,maxAdmissibleC),]

				for iz in range(self.p.nz):
					em = EMAX[ix,:,iz,iyP].flatten()
					iteratorFn = lambda c: self.findValueFromSwitching(c,cgrid,em,util)

					# candidateC = np.zeros((5,))
					funVals = np.zeros((6,))
					ii = 0
					# for x0 in [self.p.cMin+1e-4,maxAdmissibleC/3,2*maxAdmissibleC/3,maxAdmissibleC]:
					for bound in bounds:
						# optimResult = minimize(iteratorFn,x0,method='SLSQP',bounds=cBounds)
						# candidateC[ii] = optimResult.x
						# funVals[ii] = optimResult.fun
						funVals[ii] = functions.goldenSectionSearch(iteratorFn,
							bound[0],bound[1],goldenRatio,goldenRatioSq,1e-8,tuple())
						ii += 1

					# try consumting cmin
					funVals[ii] = iteratorFn(self.p.cMin)
					ii += 1
					# try consuming xval (or cmax)
					funVals[ii] = iteratorFn(maxAdmissibleC)

					self.valueSwitch[ix,0,iz,iyP] = - funVals.min()

	cdef findValueFromSwitching(self, double cSwitch, double[:] cgrid, double[:] em, double[:] util):
		"""
		Output is - [u(cSwitch) + beta * EMAX(cSwitch)]
		"""
		cdef list indices, weights
		cdef int ind1, ind2
		cdef double weight1, weight2, u, emOut

		(indices, weights) = functions.interpolate1D(cgrid, cSwitch)
		ind1 = indices[0]
		ind2 = indices[1]
		weight1 = weights[0]
		weight2 = weights[1]
		u = weight1 * util[ind1] + weight2 * util[ind2]
		emOUT = weight1 * em[ind1] + weight2 * em[ind2]

		return - u - self.p.timeDiscount * (1 - self.p.deathProb * self.p.Bequests) * emOUT


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
