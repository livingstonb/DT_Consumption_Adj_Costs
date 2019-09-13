import numpy as np
cimport numpy as np

from scipy.optimize import minimize
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp1d
import numpy.matlib as matlib
from misc import functions
from misc cimport functions
import pandas as pd
from matplotlib import pyplot as plt
from scipy import sparse

cdef class Model:
	cdef:
		object p, grids, income
		double nextMPCShock
		tuple dims, dims_yT
		dict nextModel, stats
		# np.ndarray[np.float64_t, ndim=4] valueNoSwitch, valueSwitch, valueFunction
		# np.ndarray[np.float64_t, ndim=2] interpMat
		public double[:,:,:,:] valueNoSwitch, valueSwitch, valueFunction
		public double[:,:,:,:] EMAX
		object interpMat
		public double[:,:,:,:] cSwitchingPolicy

	def __init__(self, params, income, grids, nextModel=None, nextMPCShock=0):

		self.p = params
		self.grids = grids
		self.income = income

		self.nextMPCShock = nextMPCShock

		self.dims = (params.nx, params.nc, params.nz, params.nyP)
		self.dims_yT = self.dims + (self.p.nyT,)

		self.nextModel = nextModel

		self.stats = dict()

		self.initialize()

	def initialize(self):
		print('Constructing interpolant for EMAX')
		self.constructInterpolantForV()

		# make initial guess for value function
		valueGuess = functions.utilityVec(self.p.riskAver,self.grids.c['matrix']
			) / (1 - self.p.timeDiscount * (1 - self.p.deathProb))

		# subtract the adjustment cost for states with c > x
		self.valueFunction = valueGuess - self.p.adjustCost * self.grids.mustSwitch

	def solve(self):
		print('Beginning value function iteration...')
		distance = 1e5
		iteration = 0
		while distance > self.p.tol:

			if iteration > self.p.maxIters:
				raise Exception(f'No convergence after {iteration+1} iterations...')

			Vprevious = self.valueFunction.copy()

			# update EMAX = E[V|x,c,z,yP], where c is CHOSEN c not state variable
			self.updateEMAX()

			# update value function of not switching
			self.updateValueNoSwitch()

			# update value function of switching
			self.maximizeValueFromSwitching()

			# compute V = max(VSwitch,VNoSwitch)
			self.updateValueFunction()

			distance = np.abs(
				np.asarray(self.valueFunction) - np.asarray(Vprevious)
				).flatten().max()

			if np.mod(iteration,10) == 0:
				print(f'    Iteration {iteration}, norm of |V1-V| = {distance}')

			iteration += 1

		# value function found, now re-compute valueSwitch and valueNoSwitch
		# self.updateEMAX()
		# self.updateValueNoSwitch()
		# self.maximizeValueFromSwitching()

		# compute c-policy function conditional on switching
		self.maximizeValueFromSwitching(findPolicy=True)

		print('Value function converged')

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
			np.ndarray[np.float64_t, ndim=1] xgrid1, xgrid2
			np.ndarray[np.float64_t, ndim=2] xprime, yTvec, yTdist
			np.ndarray[float, ndim=8] interpMat
			np.ndarray[np.float64_t, ndim=3] newBlock
			double[:,:,:] interpWithyT

		yTvec = self.income.yTgrid.reshape((1,-1))
		yTdistvec = self.income.yTdist.reshape((1,-1))

		interpMat = np.zeros((self.p.nx,self.p.nc,self.p.nz,self.p.nyP,self.p.nx,self.p.nc,self.p.nz,self.p.nyP),dtype='float32')
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


		self.interpMat = sparse.csr_matrix(interpMat.reshape((self.p.nx*self.p.nc*self.p.nz*self.p.nyP,
			self.p.nx*self.p.nc*self.p.nz*self.p.nyP)))

	def updateValueFunction(self):
		"""
		This method updates self.valueFunction by finding max(valueSwitch,valueNoSwitch),
		where valueSwitch is used wherever c > x in the state space.
		"""
		self.valueFunction = np.where(self.grids.mustSwitch,
			np.asarray(self.valueSwitch)-self.p.adjustCost,
			np.maximum(self.valueNoSwitch,np.asarray(self.valueSwitch)-self.p.adjustCost)
			)

	def updateEMAX(self):
		self.EMAX = self.interpMat.dot(np.reshape(self.valueFunction,(-1,1))
				).reshape(self.grids.matrixDim)

	def updateValueNoSwitch(self):
		"""
		Updates self.valueNoSwitch via valueNoSwitch(c) = u(c) + beta * E[V(c)]
		"""
		self.valueNoSwitch = functions.utilityVec(self.p.riskAver,self.grids.c['matrix']) \
			+ self.p.timeDiscount * (1 - self.p.deathProb) \
			* np.asarray(self.EMAX)

		self.valueNoSwitch = np.reshape(self.valueNoSwitch,self.grids.matrixDim)

	def maximizeValueFromSwitching(self, findPolicy=False):
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
			np.ndarray[np.float64_t, ndim=1] cVals, funVals
			double[:] cgrid, util, em
			np.ndarray[np.float64_t, ndim=4] EMAX
			tuple cBounds, bounds, bound

		if findPolicy:
			self.cSwitchingPolicy = np.zeros((self.p.nx,1,self.p.nz,self.p.nyP))

		util = functions.utilityVec(self.p.riskAver,self.grids.c['vec']).flatten()
		cgrid = self.grids.c['vec'].flatten()

		goldenRatio = (np.sqrt(5) + 1) / 2
		goldenRatioSq = goldenRatio ** 2

		nSections = 10
		sections = np.linspace(1/nSections,1,num=nSections)

		cVals = np.zeros((nSections+2,))
		funVals = np.zeros((nSections+2,))

		if not findPolicy:
			self.valueSwitch = np.zeros((self.p.nx,1,self.p.nz,self.p.nyP))

		for iyP in range(self.p.nyP):
			# EMAXInterpolant = RegularGridInterpolator(
			# 	(self.grids.x['wide'][:,0,0,iyP],self.grids.c['vec'].flatten(),self.grids.z['vec'].flatten()),EMAX[:,:,:,iyP],bounds_error=False)

			for ix in range(self.p.nx):
				xval = self.grids.x['wide'][ix,0,0,iyP]
				maxAdmissibleC = np.minimum(xval,self.p.cMax)
				cBounds = ((self.p.cMin,maxAdmissibleC),)
				# bounds = (	(self.p.cMin,maxAdmissibleC/4),
				# 			(maxAdmissibleC/4,maxAdmissibleC/2),
				# 			(maxAdmissibleC/2,3*maxAdmissibleC/4),
				# 			(3*maxAdmissibleC/4,maxAdmissibleC),
				# 			)
				bounds = tuple((maxAdmissibleC*sections[i],maxAdmissibleC*sections[i+1]) for i in range(nSections-1))
				bounds = ((self.p.cMin,maxAdmissibleC*sections[0]),) + bounds

				for iz in range(self.p.nz):
					em = self.EMAX[ix,:,iz,iyP]
					iteratorFn = lambda c: self.findValueFromSwitching(c,cgrid,em,util)

					ii = 0
					# for x0 in [self.p.cMin+1e-4,maxAdmissibleC/3,2*maxAdmissibleC/3,maxAdmissibleC]:
					for bound in bounds:
						# optimResult = minimize(iteratorFn,x0,method='SLSQP',bounds=cBounds)
						# candidateC[ii] = optimResult.x
						# funVals[ii] = optimResult.fun
						funVals[ii], cVals[ii] = functions.goldenSectionSearch(iteratorFn,
							bound[0],bound[1],goldenRatio,goldenRatioSq,1e-8,tuple())
						ii += 1

					# try consuming cmin
					cVals[ii] = self.p.cMin
					funVals[ii] = iteratorFn(self.p.cMin)
					ii += 1

					# try consuming xval (or cmax)
					cVals[ii] = maxAdmissibleC
					funVals[ii] = iteratorFn(maxAdmissibleC)

					if findPolicy:
						self.cSwitchingPolicy[ix,0,iz,iyP] = cVals[funVals.argmax()]
					else:
						self.valueSwitch[ix,0,iz,iyP] = funVals.max()

	cdef findValueFromSwitching(self, double cSwitch, double[:] cgrid, double[:] em, double[:] util):
		"""
		Output is u(cSwitch) + beta * EMAX(cSwitch)
		"""
		cdef list indices
		cdef int ind1, ind2
		cdef double weight1, weight2, u, emOut
		cdef double[:] weights

		# (indices, weights) = functions.interpolate1D(cgrid, cSwitch)
		ind2 = functions.searchSortedSingleInput(cgrid, cSwitch)
		ind1 = ind2 - 1
		weights = functions.getInterpolationWeights(cgrid, cSwitch, ind2)
		weight1 = weights[0]
		weight2 = weights[1]
		u = functions.utility(self.p.riskAver,cSwitch)
		# u = weight1 * util[ind1] + weight2 * util[ind2]
		emOUT = weight1 * em[ind1] + weight2 * em[ind2]

		return u + self.p.timeDiscount * (1 - self.p.deathProb) * emOUT

	def resetParameters(self, newParams):
		self.p = newParams