
import numpy as np
cimport numpy as np

cimport cython

from misc cimport functions
from misc.functions cimport FnArgs, objectiveFn
from misc cimport spline

import pandas as pd
from scipy import sparse

from libc.math cimport fmin
from libc.stdlib cimport malloc, free


cdef enum:
	# number of sections to try in golden section search
	NSECTIONS = 20

	# number of sections + boundaries
	NVALUES = NSECTIONS + 2

cdef class CModel:
	"""
	This extension class solves for the value function of a heterogenous
	agent model with disutility of consumption adjustment.
	"""
	cdef:
		public object p
		readonly object grids, income
		public double nextMPCShock
		readonly tuple dims, dims_yT
		public double[:,:,:,:] valueNoSwitch, valueSwitch, valueFunction
		public double[:,:,:,:] EMAX
		readonly object interpMat
		public double[:,:,:,:] cSwitchingPolicy
		public double[:,:,:] inactionRegionLower, inactionRegionUpper

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

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def updateEMAXslow(self):
		"""
		This method can be used to double check the construction of the
		interpolation matrix for updating EMAT.
		"""
		cdef:
			double emax, Pytrans, assets, cash, yP2
			double vInterp, xval
			double[:] xgrid, cgrid, valueVec
			double[:,:] yPtrans
			double[:] yPgrid, yTdist, yTgrid
			double xWeights[2]
			double *yderivs
			double *value
			long xIndices[2]
			int ix, ic, iz, iyP1, iyP2, iyT, ix2

		xgrid = self.grids.x.flat
		cgrid = self.grids.c.flat
		yPgrid = self.income.yPgrid.flatten()
		yPtrans = self.income.yPtrans
		yTgrid = self.income.yTgrid.flatten()
		yTdist = self.income.yTdist.flatten()

		yderivs = <double *> malloc(self.p.nx * sizeof(double))
		value = <double *> malloc(self.p.nx * sizeof(double))

		self.EMAX = np.zeros((self.p.nx,self.p.nc,self.p.nz,self.p.nyP))
		
		for ix in range(self.p.nx):
			xval = xgrid[ix]

			for ic in range(self.p.nc):
				assets = self.p.R * (xval - cgrid[ic])

				for iz in range(self.p.nz):

					for iyP1 in range(self.p.nyP):
						emax = 0

						for iyP2 in range(self.p.nyP):
							Pytrans = yPtrans[iyP1,iyP2]
							yP2 = yPgrid[iyP2]

							for ix2 in range(self.p.nx):
								value[ix2] = self.valueFunction[ix2,ic,iz,iyP2]

							if self.p.cubicEMAXInterp:
								spline.spline(&xgrid[0], value, self.p.nx, 
									1.0e30, 1.0e30, yderivs)

							for iyT in range(self.p.nyT):
								cash = assets + yP2 * yTgrid[iyT] + self.nextMPCShock

								if self.p.cubicEMAXInterp:
									spline.splint(&xgrid[0], value, yderivs, self.p.nx, cash, &vInterp)

									if vInterp > value[self.p.nx]:
										vInterp = value[self.p.nx]
									elif vInterp < value[0]:
										vInterp = value[0]
									emax += Pytrans * yTdist[iyT] * vInterp
								else:
									functions.getInterpolationWeights(&xgrid[0],cash,self.p.nx,&xIndices[0],&xWeights[0])

									emax += Pytrans * yTdist[iyT] * (
										xWeights[0] * self.valueFunction[xIndices[0],ic,iz,iyP2]
										+ xWeights[1] * self.valueFunction[xIndices[1],ic,iz,iyP2]
										)

						self.EMAX[ix,ic,iz,iyP1] = emax

		free(yderivs)
		free(value)

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
			int error
			double invGoldenRatio, invGoldenRatioSq
			double[:] sections
			double[:] xgrid, cgrid
			long nyP
			FnArgs fargs

		if findPolicy:
			self.cSwitchingPolicy = np.zeros((self.p.nx,1,self.p.nz,self.p.nyP))

		xgrid = self.grids.x.flat
		cgrid = self.grids.c.flat

		fargs.cgrid = &cgrid[0]
		fargs.cubicValueInterp = self.p.cubicValueInterp
		fargs.nx = self.p.nx
		fargs.cMin = self.p.cMin
		fargs.cMax = self.p.cMax
		fargs.nz = self.p.nz
		fargs.nc = self.p.nc
		fargs.riskAver = self.p.riskAver
		fargs.timeDiscount = self.p.timeDiscount
		fargs.deathProb = self.p.deathProb

		sections = np.linspace(1/<double>NSECTIONS, 1, num=NSECTIONS)

		if not findPolicy:
			self.valueSwitch = np.zeros((self.p.nx,1,self.p.nz,self.p.nyP))

		nyP = self.p.nyP

		for iyP in range(nyP):
			error = self.valueForOneIncomeBlock(xgrid, cgrid, sections, 
				findPolicy, iyP, fargs)
			if error == 1:
				raise Exception('Exception in findValueFromSwitching')
			elif error == 2:
				raise Exception('Exception in valueForOneIncomeBlock')
			elif error == -1:
				raise Exception('Exception in golden section search')
		

	@cython.boundscheck(False)
	@cython.wraparound(False)
	cdef int valueForOneIncomeBlock(self, double[:] xgrid, double[:] cgrid,
		double[:] sections, bint findPolicy, long iyP, FnArgs fargs_in) nogil except 2:
		"""
		Updates self.valueSwitch for one income block.
		"""
		cdef:
			long ix, ii, iz, ic
			double xval, maxAdmissibleC
			double *emaxVec
			double *yderivs
			long errorGSS, error = 0
			double bounds[NSECTIONS][2]
			double gssResults[2]
			double cVals[NVALUES]
			double funVals[NVALUES]
			FnArgs fargs
			objectiveFn iteratorFn
			
		bounds[0][0] = fargs_in.cMin
		fargs = fargs_in
		fargs.error = &error

		emaxVec = <double *> malloc(fargs.nc * sizeof(double))
		fargs.emaxVec = emaxVec

		yderivs = <double *> malloc(fargs.nc * sizeof(double))
		fargs.yderivs = yderivs

		for ix in range(fargs.nx):
			xval = xgrid[ix]
			maxAdmissibleC = fmin(xval,fargs.cMax)

			bounds[0][1] = maxAdmissibleC * sections[0]
			for ii in range(1,NSECTIONS):
				bounds[ii][0] = maxAdmissibleC * sections[ii-1]
				bounds[ii][1] = maxAdmissibleC * sections[ii]

			for iz in range(fargs.nz):
				iteratorFn = <objectiveFn> self.findValueFromSwitching

				fargs.ncValid = 0
				for ic in range(fargs.nc):
					emaxVec[ic] = self.EMAX[ix,ic,iz,iyP]
					
					if xval >= cgrid[ic]:
						fargs.ncValid += 1

				if fargs.cubicValueInterp:
					spline.spline(&cgrid[0], emaxVec, fargs.nc, 1.0e30, 1.0e30, yderivs)

				for ii in range(NSECTIONS):
					errorGSS = functions.goldenSectionSearch(iteratorFn, bounds[ii][0],
						bounds[ii][1],1e-8, &gssResults[0], fargs)
					if errorGSS != 0:
						return errorGSS

					funVals[ii] = gssResults[0]
					cVals[ii] = gssResults[1]
				ii = NSECTIONS

				# try consuming cmin
				cVals[ii] = fargs.cMin
				funVals[ii] = self.findValueFromSwitching(fargs.cMin, fargs)
				if error == 1:
					return error
				ii += 1

				# try consuming xval (or cmax)
				cVals[ii] = maxAdmissibleC
				funVals[ii] = self.findValueFromSwitching(maxAdmissibleC, fargs)
				if error == 1:
					return error

				if findPolicy:
					self.cSwitchingPolicy[ix,0,iz,iyP] = cVals[functions.cargmax(funVals,NVALUES)]
				else:
					self.valueSwitch[ix,0,iz,iyP] = functions.cmax(funVals,NVALUES)

		free(emaxVec)
		free(yderivs)

		return 0

	@cython.boundscheck(False)
	@cython.wraparound(False)
	cdef double findValueFromSwitching(self, double cSwitch, FnArgs fargs) nogil:
		"""
		Outputs the value u(cSwitch) + beta * EMAX(cSwitch) for a given cSwitch.
		"""
		cdef double u, emax, value
		cdef double weights[2]
		cdef long indices[2]
	
		if (fargs.ncValid > 4) and fargs.cubicValueInterp:
			fargs.error[0] = spline.splint(fargs.cgrid, fargs.emaxVec, fargs.yderivs, fargs.nc, cSwitch, &emax)
		else:
			functions.getInterpolationWeights(fargs.cgrid, cSwitch, fargs.nc, &indices[0], &weights[0])

			emax = weights[0] * fargs.emaxVec[indices[0]] + weights[1] * fargs.emaxVec[indices[1]]

		u = functions.utility(fargs.riskAver,cSwitch)
		value = u + fargs.timeDiscount * (1 - fargs.deathProb) * emax

		return value

	@cython.boundscheck(False)
	def doComputations(self):
		cdef np.ndarray[np.uint8_t, ndim=4, cast=True] cSwitch
		cdef np.ndarray[np.int64_t, ndim=1] inactionRegion
		cdef long ix, iz, iyP

		cSwitch = np.asarray(self.valueFunction) == (np.asarray(self.valueSwitch) - self.p.adjustCost)

		self.inactionRegionLower = np.zeros((self.p.nx,self.p.nz,self.p.nyP))
		self.inactionRegionUpper = np.zeros((self.p.nx,self.p.nz,self.p.nyP))

		for ix in range(self.p.nx):
			for iz in range(self.p.nz):
				for iyP in range(self.p.nyP):
					inactionRegion = np.flatnonzero(~cSwitch[ix,:,iz,iyP])
					if inactionRegion.size > 0:
						self.inactionRegionLower[ix,iz,iyP] = self.grids.c.flat[inactionRegion[0]]
						self.inactionRegionUpper[ix,iz,iyP] = self.grids.c.flat[inactionRegion[-1]]
					else:
						self.inactionRegionLower[ix,iz,iyP] = np.nan
						self.inactionRegionUpper[ix,iz,iyP] = np.nan

	def resetParams(self, newParams):
		self.p = newParams