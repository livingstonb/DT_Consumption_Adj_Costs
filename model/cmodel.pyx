IF UNAME_SYSNAME == "Linux":
	cdef enum:
		OPENMP = True
ELSE:
	cdef enum:
		OPENMP = False

import numpy as np
cimport numpy as np

cimport cython

from misc cimport functions
from misc.functions cimport objectiveFn, FnParameters
from misc cimport spline

import pandas as pd
from scipy import sparse

from cython.parallel import prange
from libc.math cimport fmin
from libc.stdlib cimport malloc, free


cdef enum:
	# number of sections to try in golden section search
	NSECTIONS = 50

	# number of sections + boundaries
	NVALUES = NSECTIONS + 2

cdef class CModel:
	"""
	This extension class solves for the value function of a heterogenous
	agent model with disutility of consumption adjustment.
	"""
	cdef:
		readonly object p, grids, income
		public double nextMPCShock
		readonly tuple dims, dims_yT
		public double[:,:,:,:] valueNoSwitch, valueSwitch, valueFunction
		public double[:,:,:,:] EMAX
		readonly object interpMat
		public double[:,:,:,:] cSwitchingPolicy

	def __init__(self, params, income, grids):
		self.p = params
		self.grids = grids
		self.income = income

		self.dims = (params.nx, params.nc, params.nz, params.nyP)
		self.dims_yT = self.dims + (self.p.nyT,)

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

	# @cython.boundscheck(False)
	# @cython.wraparound(False)
	# def updateEMAXslow(self):
	# 	"""
	# 	This method can be used to double check the construction of the
	# 	interpolation matrix for updating EMAT.
	# 	"""
	# 	cdef:
	# 		double emax, Pytrans, assets, cash, yP2
	# 		double[:] xgrid, cgrid, valueVec
	# 		double xWeights[2]
	# 		double *xWeights_ptr = xWeights
	# 		double *yderivs
	# 		long[:] xIndices
	# 		int ix, ic, iz, iyP1, iyP2, iyT

	# 	xgrid = self.grids.x.flat
	# 	cgrid = self.grids.c.flat

	# 	yderivs = <double *> malloc(self.p.nx * sizeof(double))

	# 	xIndices = np.zeros(2,dtype=int)

	# 	self.EMAX = np.zeros((self.p.nx,self.p.nc,self.p.nz,self.p.nyP))
		
	# 	for ix in range(self.p.nx):

	# 		for ic in range(self.p.nc):
	# 			assets = self.p.R * (xgrid[ix] - cgrid[ic])

	# 			for iz in range(self.p.nz):

	# 				for iyP1 in range(self.p.nyP):
	# 					emax = 0

	# 					for iyP2 in range(self.p.nyP):
	# 						Pytrans = self.income.yPtrans[iyP1,iyP2]
	# 						yP2 = self.income.yPgrid[iyP2]

	# 						for iyT in range(self.p.nyT):
	# 							cash = assets + yP2 * self.income.yTgrid[iyT] + self.nextMPCShock

	# 							xIndices[1] = functions.fastSearchSingleInput(xgrid,cash,self.p.nx)
	# 							xIndices[0] = xIndices[1] - 1
	# 							functions.getInterpolationWeights(xgrid,cash,xIndices[1],xWeights_ptr)
				

	# 							emax += Pytrans * self.income.yTdist[iyT] * (
	# 								xWeights[0] * self.valueFunction[xIndices[0],ic,iz,iyP2]
	# 								+ xWeights[1] * self.valueFunction[xIndices[1],ic,iz,iyP2]
	# 								)

	# 					self.EMAX[ix,ic,iz,iyP1] = emax

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def maximizeValueFromSwitching(self, bint findPolicy=False):
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

		if OPENMP:
			for iyP in prange(nyP, nogil=True):
				self.valueForOneIncomeBlock(xgrid, cgrid, sections, 
					findPolicy, iyP, fparams)
		else:
			for iyP in range(nyP):
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
				funVals[ii] = self.findValueFromSwitching(fparams.cMin, cgrid, emaxVec, fparams)
				ii += 1

				# try consuming xval (or cmax)
				cVals[ii] = maxAdmissibleC
				funVals[ii] = self.findValueFromSwitching(maxAdmissibleC, cgrid, emaxVec, fparams)

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
		cdef double u, emax, value
		cdef double weights[2]
		cdef long indices[2]
		
		functions.getInterpolationWeights(cgrid, cSwitch, fparams.nc, &indices[0], &weights[0])

		emax = weights[0] * emaxVec[indices[0]] + weights[1] * emaxVec[indices[1]]

		u = functions.utility(fparams.riskAver,cSwitch)
		value = u + fparams.timeDiscount * (1 - fparams.deathProb) * emax

		return value