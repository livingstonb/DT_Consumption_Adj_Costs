import numpy as np
cimport numpy as np

cimport cython

from misc import functions
from misc cimport functions
import pandas as pd
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

		# compute c-policy function conditional on switching
		self.maximizeValueFromSwitching(findPolicy=True)

		print('Value function converged')

	@cython.boundscheck(False)
	@cython.wraparound(False)
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
						xprime = self.p.R * (xgrid[:,None] - self.grids.c.flat[ic]) + yP * yTvec
						interpWithyT = functions.interpolateTransitionProbabilities2D(xgrid,xprime)
						newBlock = PyP1yP2 * np.squeeze(np.dot(yTdistvec,interpWithyT))

						blocks[iblock] = sparse.csr_matrix(newBlock)

						iblock += 1

				blockMats[iyP1][iyP2] = sparse.block_diag(blocks)

		self.interpMat = sparse.bmat(blockMats)

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
			double[:] xgrid, cgrid, xWeights
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
								cash = assets + yP2 * self.income.yTgrid[iyT]
								xIndices[1] = functions.searchSortedSingleInput(xgrid,cash)
								xIndices[0] = xIndices[1] - 1
								xWeights = functions.getInterpolationWeights(xgrid,cash,xIndices[1])
				

								emax += Pytrans * self.income.yTdist[iyT] * (
									xWeights[0] * self.valueFunction[xIndices[0],ic,iz,iyP2]
									+ xWeights[1] * self.valueFunction[xIndices[1],ic,iz,iyP2]
									)

						self.EMAX[ix,ic,iz,iyP1] = emax

	def updateValueNoSwitch(self):
		"""
		Updates self.valueNoSwitch via valueNoSwitch(c) = u(c) + beta * E[V(c)]
		"""
		self.valueNoSwitch = functions.utilityMat(self.p.riskAver,self.grids.c.matrix) \
			+ self.p.timeDiscount * (1 - self.p.deathProb) \
			* np.asarray(self.EMAX)

		self.valueNoSwitch = np.reshape(self.valueNoSwitch,self.grids.matrixDim)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def maximizeValueFromSwitching(self, findPolicy=False):
		"""
		Updates self.valueSwitch by first approximating EMAX(x,c,z,yP), defined as
		E[V(R(x-c)+yP'yT', c, z, yP')|x,c,z,yP]. This approximation is done by
		interpolating the sums over income transitions using interpMat. Then a
		maximization is done over u(c) + beta * EMAX at each point in the
		(x,z,yP)-space by interpolating u(c) and EMAX(c) at each iteration.
		"""
		cdef:
			int iyP, ix, iz, ii, nSections, i
			double xval, maxAdmissibleC, goldenRatio, goldenRatioSq
			np.ndarray[np.float64_t, ndim=1] cVals, funVals
			double[:] cgrid, util, sections
			np.ndarray[np.float64_t, ndim=4] EMAX
			tuple cBounds, bounds, bound

		if findPolicy:
			self.cSwitchingPolicy = np.zeros((self.p.nx,1,self.p.nz,self.p.nyP))

		cgrid = self.grids.c.flat

		goldenRatio = (np.sqrt(5) + 1) / 2
		goldenRatioSq = goldenRatio ** 2

		nSections = 5
		sections = np.linspace(1/nSections,1,num=nSections)

		cVals = np.zeros((nSections+2,))
		funVals = np.zeros((nSections+2,))

		if not findPolicy:
			self.valueSwitch = np.zeros((self.p.nx,1,self.p.nz,self.p.nyP))

		for iyP in range(self.p.nyP):

			for ix in range(self.p.nx):
				xval = self.grids.x.flat[ix]
				maxAdmissibleC = np.minimum(xval,self.p.cMax)
				cBounds = ((self.p.cMin,maxAdmissibleC),)
				bounds = tuple((maxAdmissibleC*sections[i],maxAdmissibleC*sections[i+1]) for i in range(nSections-1))
				bounds = ((self.p.cMin,maxAdmissibleC*sections[0]),) + bounds

				for iz in range(self.p.nz):
					iteratorFn = lambda c: self.findValueFromSwitching(c,cgrid,self.EMAX[ix,:,iz,iyP])

					ii = 0
					for bound in bounds:
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

	@cython.boundscheck(False)
	@cython.wraparound(False)
	cdef findValueFromSwitching(self, double cSwitch, double[:] cgrid, double[:] em):
		"""
		Output is u(cSwitch) + beta * EMAX(cSwitch)
		"""
		cdef list indices
		cdef int ind1, ind2
		cdef double weight1, weight2, u, emOut
		cdef double[:] weights

		ind2 = functions.searchSortedSingleInput(cgrid, cSwitch)
		ind1 = ind2 - 1
		weights = functions.getInterpolationWeights(cgrid, cSwitch, ind2)
		weight1 = weights[0]
		weight2 = weights[1]
		u = functions.utility(self.p.riskAver,cSwitch)
		emOUT = weight1 * em[ind1] + weight2 * em[ind2]

		return u + self.p.timeDiscount * (1 - self.p.deathProb) * emOUT

	def resetDiscountRate(self, newTimeDiscount):
		self.p.timeDiscount = newTimeDiscount