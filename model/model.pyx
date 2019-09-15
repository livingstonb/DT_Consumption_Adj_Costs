import numpy as np
cimport numpy as np

cimport cython

from misc cimport functions
from misc.functions cimport objectiveFn, FnParameters

import pandas as pd
from scipy import sparse

from cython.parallel import prange, parallel
from libc.math cimport fmin, pow, sqrt


cdef enum:
	# number of sections to use in golden section search
	NSECTIONS = 5

	# number of sections + boundaries
	NVALUES = NSECTIONS + 2

cdef class Model:
	"""
	This extension class solves for the value function of a heterogenous
	agent model with disutility of consumption adjustment.
	"""
	cdef:
		object p, grids, income
		double nextMPCShock
		tuple dims, dims_yT
		bint EMAXPassedAsArgument
		dict nextModel, stats
		# np.ndarray[np.float64_t, ndim=4] valueNoSwitch, valueSwitch, valueFunction
		# np.ndarray[np.float64_t, ndim=2] interpMat
		public double[:,:,:,:] valueNoSwitch, valueSwitch, valueFunction
		public double[:,:,:,:] EMAX
		object interpMat
		public double[:,:,:,:] cSwitchingPolicy

	def __init__(self, params, income, grids, EMAX=None, nextMPCShock=0):

		self.p = params
		self.grids = grids
		self.income = income

		self.nextMPCShock = nextMPCShock

		self.dims = (params.nx, params.nc, params.nz, params.nyP)
		self.dims_yT = self.dims + (self.p.nyT,)

		if EMAX is None:
			self.EMAXPassedAsArgument = False
		else:
			# EMAX is based on expectation of a future shock,
			# do not update with value function of current model
			self.EMAXPassedAsArgument = True
			self.EMAX = EMAX
			
		self.stats = dict()

		self.initialize()

	def initialize(self):
		if not self.EMAXPassedAsArgument:
			print('Constructing interpolant for EMAX')
			self.constructInterpolantForEMAX()

		# make initial guess for value function
		valueGuess = functions.utilityMat(self.p.riskAver,self.grids.c.matrix
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

			if not self.EMAXPassedAsArgument:
				# update EMAX = E[V|x,c,z,yP], where c is chosen c
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

		# compute c-policy function conditional on switching
		self.maximizeValueFromSwitching(findPolicy=True)

		print('Value function converged')

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def constructInterpolantForEMAX(self):
		"""
		This method constructs an interpolant array interpMat such that
		interpMat * valueFunction(:) = E[V]. That is, the expected continuation 
		value of being in state (x_i,~,z_k,yP_k) and choosing to consume c_j, 
		the (i,j,k,l)-th element of the matrix product of interpMat and 
		valueFunction(:) is the quantity below:

		E[V(R(x_i-c_j)+yP'yT', c_j, z_k, yP')|x_i,c_j,z_k,yP_l]

		where the expectation is taken over yP' and yT'.

		Output is stored in self.interpMat. Note that this array is constructed
		in Fortran style, must use order='F' for matrix multiplication.
		"""
		cdef:
			int iyP1, iyP2, ic, iz, i
			long iblock
			double yP, PyP1yP2
			np.ndarray[np.float64_t, ndim=1] xgrid
			np.ndarray[np.float64_t, ndim=2] xprime, yTvec, yTdist
			np.ndarray[float, ndim=8] interpMat
			np.ndarray[np.float64_t, ndim=2] newBlock
			double[:,:,:] interpWithyT
			list blocks

		yTvec = self.income.yTgrid.reshape((1,-1))
		yTdistvec = self.income.yTdist.reshape((1,-1))

		xgrid = np.asarray(self.grids.x.flat)

		blockMats = [[None] * self.p.nyP] * self.p.nyP
		
		for iyP1 in range(self.p.nyP):
			for iyP2 in range(self.p.nyP):
				iblock = 0
				blocks = [None] * (self.p.nz*self.p.nc)
				
				yP = self.income.yPgrid[iyP2]
				PyP1yP2 = self.income.yPtrans[iyP1,iyP2]

				for iz in range(self.p.nz):

					for ic in range(self.p.nc):
						xprime = self.p.R * (xgrid[:,None] - self.grids.c.flat[ic]) \
							+ yP * yTvec + self.nextMPCShock
						interpWithyT = functions.interpolateTransitionProbabilities2D(xgrid,xprime)
						newBlock = PyP1yP2 * np.squeeze(np.dot(yTdistvec,interpWithyT))

						blocks[iblock] = sparse.csr_matrix(newBlock)

						iblock += 1

				blockMats[iyP1][iyP2] = sparse.block_diag(blocks)

		self.interpMat = sparse.bmat(blockMats)

	def updateValueFunction(self):
		"""
		This method updates self.valueFunction by finding max(valueSwitch-adjustCost,valueNoSwitch),
		where valueSwitch is used wherever c > x in the state space.
		"""
		self.valueFunction = np.where(self.grids.mustSwitch,
			np.asarray(self.valueSwitch)-self.p.adjustCost,
			np.maximum(self.valueNoSwitch,np.asarray(self.valueSwitch)-self.p.adjustCost)
			)

	def updateEMAX(self):
		"""
		This method computes E[V] from the most recent value function iteration.
		"""
		self.EMAX = self.interpMat.dot(np.reshape(self.valueFunction,(-1,1),order='F')
				).reshape(self.grids.matrixDim,order='F')

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def updateEMAXslow(self):
		"""
		This method can be used to double check the construction of the
		interpolation matrix for updating EMAT.
		"""
		cdef:
			double emax, Pytrans, assets, cash, yP2
			double[:] xgrid, cgrid
			double xWeights[2]
			double *xWeights_ptr = xWeights
			long[:] xIndices
			int ix, ic, iz, iyP1, iyP2, iyT

		xgrid = self.grids.x.flat
		cgrid = self.grids.c.flat

		xIndices = np.zeros(2,dtype=int)

		self.EMAX = np.zeros((self.p.nx,self.p.nc,self.p.nz,self.p.nyP))
		
		for ix in range(self.p.nx):

			for ic in range(self.p.nc):
				assets = self.p.R * (xgrid[ix] - cgrid[ic])

				for iz in range(self.p.nz):

					for iyP1 in range(self.p.nyP):
						emax = 0

						for iyP2 in range(self.p.nyP):
							Pytrans = self.income.yPtrans[iyP1,iyP2]
							yP2 = self.income.yPgrid[iyP2]

							for iyT in range(self.p.nyT):
								cash = assets + yP2 * self.income.yTgrid[iyT] + self.nextMPCShock
								xIndices[1] = functions.searchSortedSingleInput(xgrid,cash,self.p.nx)
								xIndices[0] = xIndices[1] - 1
								functions.getInterpolationWeights(xgrid,cash,xIndices[1],xWeights_ptr)
				

								emax += Pytrans * self.income.yTdist[iyT] * (
									xWeights[0] * self.valueFunction[xIndices[0],ic,iz,iyP2]
									+ xWeights[1] * self.valueFunction[xIndices[1],ic,iz,iyP2]
									)

						self.EMAX[ix,ic,iz,iyP1] = emax

	def updateValueNoSwitch(self):
		"""
		Updates self.valueNoSwitch via valueNoSwitch(c) = u(c) + beta * EMAX(c)
		"""
		self.valueNoSwitch = functions.utilityMat(self.p.riskAver,self.grids.c.matrix) \
			+ self.p.timeDiscount * (1 - self.p.deathProb) \
			* np.asarray(self.EMAX)

		self.valueNoSwitch = np.reshape(self.valueNoSwitch,self.grids.matrixDim)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	cdef maximizeValueFromSwitching(self, bint findPolicy=False):
		"""
		Updates self.valueSwitch via a maximization over u(c) + beta * EMAX(c)
		at each point in the (x,z,yP)-space by computing u(c) and interpolating 
		EMAX(c) at each iteration.
		"""
		cdef:
			long iyP
			double invGoldenRatio, invGoldenRatioSq
			double[:] sections
			double[:] xgrid, cgrid
			long nyP
			FnParameters fparams

		if findPolicy:
			self.cSwitchingPolicy = np.zeros((self.p.nx,1,self.p.nz,self.p.nyP))

		xgrid = self.grids.x.flat
		cgrid = self.grids.c.flat

		fparams.nx = self.p.nx
		fparams.cMin = self.p.cMin
		fparams.cMax = self.p.cMax
		fparams.nz = self.p.nz
		fparams.nc = self.p.nc
		fparams.riskAver = self.p.riskAver
		fparams.timeDiscount = self.p.timeDiscount
		fparams.deathProb = self.p.deathProb

		sections = np.linspace(1/<double>NSECTIONS, 1, num=NSECTIONS)

		if not findPolicy:
			self.valueSwitch = np.zeros((self.p.nx,1,self.p.nz,self.p.nyP))

		nyP = self.p.nyP
		with nogil, parallel():
			
			for iyP in prange(nyP):
				self.valueForOneIncomeBlock(xgrid, cgrid, sections, 
					findPolicy, iyP, fparams)
			

	@cython.boundscheck(False)
	@cython.wraparound(False)
	cdef void valueForOneIncomeBlock(self, double[:] xgrid, double[:] cgrid,
		double[:] sections, bint findPolicy, long iyP, FnParameters fparams) nogil:
		"""
		Updates self.valueSwitch for one income block.
		"""
		cdef:
			long ix, ii, iz
			double xval, maxAdmissibleC
			double[:] emaxVec
			double bounds[NSECTIONS][2]
			double gssResults[2]
			double cVals[NVALUES]
			double funVals[NVALUES]
			objectiveFn iteratorFn

		bounds[0][0] = fparams.cMin

		for ix in range(fparams.nx):
			xval = xgrid[ix]
			maxAdmissibleC = fmin(xval,fparams.cMax)

			bounds[0][1] = maxAdmissibleC * sections[0]
			for ii in range(1,NSECTIONS):
				bounds[ii][0] = maxAdmissibleC * sections[ii-1]
				bounds[ii][1] = maxAdmissibleC * sections[ii]

			for iz in range(fparams.nz):
				iteratorFn = <objectiveFn> self.findValueFromSwitching

				emaxVec = self.EMAX[ix,:,iz,iyP]

				for ii in range(NSECTIONS):
					functions.goldenSectionSearch(iteratorFn, bounds[ii][0],
						bounds[ii][1],1e-8, &gssResults[0], cgrid, emaxVec, fparams)
					funVals[ii] = gssResults[0]
					cVals[ii] = gssResults[1]
				ii = NSECTIONS

				# try consuming cmin
				cVals[ii] = fparams.cMin
				funVals[ii] = iteratorFn(fparams.cMin, cgrid, emaxVec, fparams)
				ii += 1

				# try consuming xval (or cmax)
				cVals[ii] = maxAdmissibleC
				funVals[ii] = iteratorFn(maxAdmissibleC, cgrid, emaxVec, fparams)

				if findPolicy:
					self.cSwitchingPolicy[ix,0,iz,iyP] = cVals[functions.cargmax(funVals,NVALUES)]
				else:
					self.valueSwitch[ix,0,iz,iyP] = functions.cmax(funVals,NVALUES)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	cdef double findValueFromSwitching(self, double cSwitch, double[:] cgrid, double[:] emaxVec,
		FnParameters fparams) nogil:
		"""
		Outputs the value u(cSwitch) + beta * EMAX(cSwitch) for a given cSwitch.
		"""
		cdef long ind1, ind2
		cdef double weight1, weight2, u, emax, value
		cdef double em1, em2
		cdef double weights[2]

		ind2 = functions.searchSortedSingleInput(cgrid, cSwitch, fparams.nc)
		ind1 = ind2 - 1
		functions.getInterpolationWeights(cgrid, cSwitch, ind2, &weights[0])

		weight1 = weights[0]
		weight2 = weights[1]

		em1 = emaxVec[ind1]
		em2 = emaxVec[ind2]

		u = functions.utility(fparams.riskAver,cSwitch)

		emax = weight1 * em1 + weight2 * em2
		value = u + fparams.timeDiscount * (1 - fparams.deathProb) * emax

		return value

	def resetDiscountRate(self, newTimeDiscount):
		self.p.timeDiscount = newTimeDiscount