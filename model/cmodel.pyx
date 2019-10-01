
import numpy as np
cimport numpy as np

cimport cython
from cython.parallel cimport prange

from misc cimport cfunctions
from misc.cfunctions cimport FnArgs, objectiveFn
from misc cimport spline

import pandas as pd
from scipy import sparse

from libc.math cimport fmin
from libc.stdlib cimport malloc, free

ctypedef struct interpMatArgs:
	long nx, nc, nz, nyP, nyT
	double R, govTransfer, xmin, timeDiscountAdj
	double *cgrid
	double *xgrid
	double *yPgrid
	double *yPtrans
	double *yTgrid
	double *yTdist

cdef enum:
	# number of sections to try in golden section search
	NSECTIONS = 20

	# number of sections + boundaries
	NVALUES = NSECTIONS + 2

cdef class CModel:
	"""
	Base class which serves as the workhorse of the model.
	"""
	cdef:
		public object p
		readonly object grids, income
		public double nextMPCShock
		readonly tuple dims, dims_yT
		public double[:,:,:,:] valueNoSwitch, valueSwitch, valueFunction
		public double[:,:,:,:] EMAX, EMAXnext, add_to_EMAX
		public double[:,:,:] inactionRegionLower, inactionRegionUpper
		long [:] I, J
		double [:] V
		public object valueDiff, cSwitchingPolicy

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
		This method constructs the sparse matrix A such that
		EMAX(x,c,z,yP) = A * E[V(R(x-c),c,z',yP')|z,yP] where
		the expectation on the right is a flattened vector.
		"""
		cdef:
			double[:] xgrid, cgrid
			double[:] yPgrid, yTdist, yTgrid, yPtrans
			long ix
			interpMatArgs interp_args

		self.add_to_EMAX = np.zeros((self.p.nx,self.p.nc,self.p.nz,self.p.nyP))

		# (I,J) indicate the (row,column) for the value in V
		self.I = np.zeros((self.p.nx*self.p.nc*self.p.nz*self.p.nyP*self.p.nyP*self.p.nyT*2),dtype=int)
		self.J = np.zeros((self.p.nx*self.p.nc*self.p.nz*self.p.nyP*self.p.nyP*self.p.nyT*2),dtype=int)
		self.V = np.zeros((self.p.nx*self.p.nc*self.p.nz*self.p.nyP*self.p.nyP*self.p.nyT*2))

		xgrid = self.grids.x.flat
		cgrid = self.grids.c.flat
		yPgrid = self.income.yPgrid.flatten()
		yPtrans = self.income.yPtrans.flatten(order='F')
		yTgrid = self.income.yTgrid.flatten()
		yTdist = self.income.yTdist.flatten()

		interp_args.xmin = xgrid[0]
		interp_args.xgrid = &xgrid[0]
		interp_args.cgrid = &cgrid[0]
		interp_args.yPgrid = &yPgrid[0]
		interp_args.yPtrans = &yPtrans[0]
		interp_args.yTgrid = &yTgrid[0]
		interp_args.yTdist = &yTdist[0]
		interp_args.nx = self.p.nx
		interp_args.nc = self.p.nc
		interp_args.nz = self.p.nz
		interp_args.nyP = self.p.nyP
		interp_args.nyT = self.p.nyT
		interp_args.R = self.p.R
		interp_args.govTransfer = self.p.govTransfer
		interp_args.timeDiscountAdj = self.p.timeDiscount * (1-self.p.deathProb)

		length = self.p.nx * self.p.nc * self.p.nz * self.p.nyP
		for ix in prange(interp_args.nx, num_threads=4, schedule='static', nogil=True):
			self.findInterpMatOneX(ix, interp_args)

		self.interpMat = sparse.coo_matrix((self.V,(self.I,self.J)),shape=(length,length)).tocsr()

	@cython.boundscheck(False)
	@cython.wraparound(False)
	cdef void findInterpMatOneX(self, long ix, interpMatArgs args) nogil:
		"""
		Constructs the rwos of the interpolant matrix
		corresponding with initial cash holdings of
		xgrid[ix]
		"""
		cdef: 
			double xWeights[2]
			double cWeights[2]
			long xIndices[2]
			long cIndices[2]
			double xval, assets, cash, Pytrans, yP2
			long ic, iz, iyP1, iyP2, iyT, ii, ii2, row

		xval = args.xgrid[ix]

		ii = ix * 2 * args.nc * args.nz * args.nyP * args.nyP * args.nyT
		ii2 = ii + 1

		for ic in range(args.nc):
			assets = args.R * (xval - args.cgrid[ic])

			for iz in range(args.nz):

				for iyP1 in range(args.nyP):

					for iyP2 in range(args.nyP):
						Pytrans = args.yPtrans[iyP1+args.nyP*iyP2]
						yP2 = args.yPgrid[iyP2]

						for iyT in range(args.nyT):

							cash = assets + yP2 * args.yTgrid[iyT] + self.nextMPCShock + args.govTransfer
							cfunctions.getInterpolationWeights(args.xgrid,cash,args.nx,&xIndices[0],&xWeights[0])

							if cash < args.xmin:
								cfunctions.getInterpolationWeights(args.cgrid,cash,args.nc,&cIndices[0],&cWeights[0])
								self.add_to_EMAX[ix,ic,iz,iyP1] += Pytrans * args.yTdist[iyT] * (
									cfunctions.utility(fargs.riskAver, cash)
									+ args.timeDiscountAdj
										* 	( 	xWeights[0] * cWeights[0] * self.EMAXnext[xIndices[0],cIndices[0],iz,iyP2]
												+ xWeights[1] * cWeights[0] * self.EMAXnext[xIndices[1],cIndices[0],iz,iyP2]
												+ xWeights[0] * cWeights[1] * self.EMAXnext[xIndices[0],cIndices[1],iz,iyP2]
												+ xWeights[1] * cWeights[1] * self.EMAXnext[xIndices[1],cIndices[1],iz,iyP2]
											)
									)

							else
								row = ix + args.nx*ic + args.nx*args.nc*iz + args.nx*args.nc*args.nz*iyP1

								self.I[ii] = row
								self.I[ii2] = row

								self.J[ii] = xIndices[0] + args.nx * ic + args.nx * args.nc * iz + args.nx * args.nc * args.nz * iyP2
								self.J[ii2] = xIndices[1] + args.nx * ic + args.nx * args.nc * iz + args.nx * args.nc * args.nz * iyP2

								self.V[ii] = Pytrans * args.yTdist[iyT] * xWeights[0]
								self.V[ii2] = Pytrans * args.yTdist[iyT] * xWeights[1]

								ii += 2
								ii2 += 2

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def maximizeValueFromSwitching(self, bint findPolicy=False):
		"""
		Updates valueSwitch via a maximization over u(c) + beta * EMAX(c)
		at each point in the (x,z,yP)-space by computing u(c) and interpolating 
		EMAX(c) at each iteration.
		"""
		cdef:
			long iyP, ix, ii, iz, ic
			double xval, maxAdmissibleC,
			double[:] emaxVec, yderivs
			double[:] risk_aver_grid, discount_factor_grid
			double[:] sections, xgrid, cgrid
			long errorGSS
			double bounds[NSECTIONS][2]
			double gssResults[2]
			double cVals[NVALUES]
			double funVals[NVALUES]
			long hetType
			FnArgs fargs
			objectiveFn iteratorFn

		xgrid = self.grids.x.flat
		cgrid = self.grids.c.flat

		fargs.error = 0
		fargs.cgrid = &cgrid[0]
		fargs.cubicValueInterp = self.p.cubicValueInterp
		fargs.nc = self.p.nc
		fargs.deathProb = self.p.deathProb
		fargs.adjustCost = self.p.adjustCost

		if self.p.risk_aver_grid.size > 1:
			hetType = 1
			fargs.timeDiscount = self.p.timeDiscount
			risk_aver_grid = self.p.risk_aver_grid
		elif self.p.discount_factor_grid.size > 1:
			hetType = 2
			fargs.riskAver = self.p.riskAver
			discount_factor_grid = self.p.discount_factor_grid
		else:
			hetType = 0
			fargs.timeDiscount = self.p.timeDiscount
			fargs.riskAver = self.p.riskAver


		sections = np.linspace(1/<double>NSECTIONS, 1, num=NSECTIONS)

		emaxVec = np.zeros(self.p.nc)
		yderivs = np.zeros(self.p.nc)

		fargs.emaxVec = &emaxVec[0]
		fargs.yderivs = &yderivs[0]

		iteratorFn = <objectiveFn> self.findValueFromSwitching

		if findPolicy:
			self.cSwitchingPolicy = np.zeros((self.p.nx,1,self.p.nz,self.p.nyP))
		else:
			self.valueSwitch = np.zeros((self.p.nx,1,self.p.nz,self.p.nyP))

		for iyP in range(self.p.nyP):
			bounds[0][0] = self.p.cMin

			for ix in range(self.p.nx):
				xval = xgrid[ix]
				maxAdmissibleC = fmin(xval,self.p.cMax)

				bounds[0][1] = maxAdmissibleC * sections[0]
				for ii in range(1,NSECTIONS):
					bounds[ii][0] = maxAdmissibleC * sections[ii-1]
					bounds[ii][1] = maxAdmissibleC * sections[ii]

				for iz in range(self.p.nz):
					if hetType == 1:
						fargs.riskAver = risk_aver_grid[iz]
					elif hetType == 2:
						fargs.timeDiscount = discount_factor_grid[iz]

					fargs.ncValid = 0
					for ic in range(self.p.nc):
						emaxVec[ic] = self.EMAX[ix,ic,iz,iyP]
						
						if xval >= cgrid[ic]:
							fargs.ncValid += 1

					if fargs.cubicValueInterp:
						spline.spline(&cgrid[0], &emaxVec[0], self.p.nc, 1.0e30, 1.0e30, &yderivs[0])

					for ii in range(NSECTIONS):
						errorGSS = cfunctions.goldenSectionSearch(iteratorFn, bounds[ii][0],
							bounds[ii][1],1e-8, &gssResults[0], fargs)
						if errorGSS == -1:
							raise Exception('Exception in golden section search')
						elif errorGSS == 1:
							raise Exception('Exception in fn called by golden section search')

						funVals[ii] = gssResults[0]
						cVals[ii] = gssResults[1]
					ii = NSECTIONS

					# try consuming cmin
					cVals[ii] = self.p.cMin
					funVals[ii] = self.findValueFromSwitching(self.p.cMin, fargs)
					ii += 1

					# try consuming xval (or cmax)
					cVals[ii] = maxAdmissibleC
					funVals[ii] = self.findValueFromSwitching(maxAdmissibleC, fargs)

					if findPolicy:
						self.cSwitchingPolicy[ix,0,iz,iyP] = cVals[cfunctions.cargmax(funVals,NVALUES)]
					else:
						self.valueSwitch[ix,0,iz,iyP] = cfunctions.cmax(funVals,NVALUES) - self.p.adjustCost

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
			fargs.error = spline.splint(fargs.cgrid, fargs.emaxVec, fargs.yderivs, fargs.nc, cSwitch, &emax)
		else:
			cfunctions.getInterpolationWeights(fargs.cgrid, cSwitch, fargs.nc, &indices[0], &weights[0])
			emax = weights[0] * fargs.emaxVec[indices[0]] + weights[1] * fargs.emaxVec[indices[1]]

		u = cfunctions.utility(fargs.riskAver,cSwitch)
		value = u + fargs.timeDiscount * (1 - fargs.deathProb) * emax

		return value

	@cython.boundscheck(False)
	def doComputations(self):
		cdef np.ndarray[np.uint8_t, ndim=4, cast=True] cSwitch
		cdef np.ndarray[np.int64_t, ndim=1] inactionRegion
		cdef long ix, iz, iyP

		cSwitch = np.asarray(self.valueSwitch) > np.asarray(self.valueNoSwitch)

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

		self.valueDiff = (np.asarray(self.valueSwitch) - np.asarray(self.valueNoSwitch)
			).reshape((self.p.nx,self.p.nc,self.p.nz,self.p.nyP,1))
		self.cSwitchingPolicy = self.cSwitchingPolicy.reshape((self.p.nx,1,self.p.nz,self.p.nyP,1))

	def resetParams(self, newParams):
		self.p = newParams