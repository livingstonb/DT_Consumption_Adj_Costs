
import numpy as np
cimport numpy as np

cimport cython
from cython.parallel cimport prange
from cython.operator cimport dereference

from misc import functions
from misc cimport cfunctions
from misc.cfunctions cimport FnArgs, objectiveFn
from misc cimport spline

from Grid cimport Grid
from Params cimport Params
from Income cimport Income

import pandas as pd
from scipy import sparse

from libc.math cimport fmin, fmax, fabs

cdef enum:
	# number of sections to try in golden section search
	NSECTIONS = 20

cdef class CModel:
	"""
	Base class which serves as the workhorse of the model.
	"""
	cdef:
		# parameters and grids
		public Params p
		public Income income
		public Grid grids
		public double[:] xgrid_curr, xgrid_next

		# value of the shock next period, used for MPCs out of news
		public double nextMPCShock

		# tuples of dimension lengths
		readonly tuple dims, dims_yT

		# borrowing limit, adjusted for next mpc shock if necessary
		public double borrLimCurr, borrLimNext

		# value functions
		public double[:,:,:,:] valueNoSwitch, valueSwitch, valueFunction

		# EMAX and next period's EMAX
		public double[:,:,:,:] EMAX, EMAXnext

		# Number of valid consumption points at each x
		long[:] validConsumptionPts

		# variables for construction of interpMat
		long [:] I, J
		double [:] V

		# excess value from switching
		public object valueDiff, willSwitch, cChosen

		# policy function for c, conditional on switching
		public object cSwitchingPolicy, inactionRegion

	def __init__(self, params, income, grids):
		"""
		Class constructor

		Parameters
		----------
		params : a Params object containing the model parameters
		"""
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
			long ix, length

		self.preliminaryComputations()

		# (I,J) indicate the (row,column) for the value in V
		entries = self.p.nx*self.p.nc*self.p.nz*self.p.nyP*self.p.nyP*self.p.nyT*2
		self.I = np.zeros(entries, dtype=int)
		self.J = np.zeros(entries, dtype=int)
		self.V = np.zeros(entries)

		length = self.p.nx * self.p.nc * self.p.nz * self.p.nyP
		for ix in prange(self.p.nx, num_threads=4, schedule='static', nogil=True):
			self.findInterpMatOneX(ix)

		self.interpMat = sparse.coo_matrix((self.V, (self.I,self.J)),
			shape=(length,length)).tocsr()

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def preliminaryComputations(self):
		cdef:
			long ix
			double xval
			object validCs

		self.xgrid_curr = np.asarray(self.grids.x_flat) \
			+ (self.borrLimCurr - self.p.borrowLim)

		self.xgrid_next = np.asarray(self.grids.x_flat) \
			+ (self.borrLimNext - self.p.borrowLim)

		self.mustSwitch = np.asarray(self.xgrid_curr)[:,None,None,None] \
			- np.asarray(self.grids.c_wide) \
			< self.borrLimCurr

		self.validConsumptionPts = np.zeros(self.p.nx, dtype=int)
		cgrid_np = np.asarray(self.grids.c_flat);
		for ix in range(self.p.nx):
			xval = self.xgrid_curr[ix];
			validCs = cgrid_np <= xval - self.borrLimCurr
			self.validConsumptionPts[ix] = np.sum(validCs).astype(int)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	cdef void findInterpMatOneX(self, long ix) nogil:
		"""
		Constructs the rows of the interpolant matrix
		corresponding with initial cash holdings of
		xgrid[ix]
		"""
		cdef: 
			double xWeights[2]
			long xIndices[2]
			double xval, assets, cash, Pytrans, yP2, sav
			long nEntries_yP1, nEntries_yP2
			long ic, iz, iyP1, iyP2, iyT, ii, ii2, row

		xval = self.xgrid_curr[ix]

		ii = ix * 2 * self.p.nc * self.p.nz * self.p.nyP * self.p.nyP * self.p.nyT
		ii2 = ii + 1

		for ic in range(self.p.nc):
			sav = xval - self.grids.c_flat[ic]
			if sav < self.borrLimCurr:
				continue

			assets = self.p.R * sav + self.nextMPCShock + self.p.govTransfer

			for iz in range(self.p.nz):
				for iyP1 in range(self.p.nyP):
					nEntries_yP1 = self.p.nx * ic + self.p.nx * self.p.nc * iz \
						+ self.p.nx * self.p.nc * self.p.nz * iyP1
					for iyP2 in range(self.p.nyP):
						Pytrans = self.income.yPtrans[iyP1, iyP2]
						yP2 = self.income.yPgrid[iyP2]

						nEntries_yP2 = self.p.nx * ic + self.p.nx * self.p.nc * iz \
								+ self.p.nx * self.p.nc * self.p.nz * iyP2

						for iyT in range(self.p.nyT):

							cash = assets + yP2 * self.income.yTgrid[iyT] 

							# EMAX associated with next period may be over adjusted grid
							cfunctions.getInterpolationWeights(&self.xgrid_next[0],
								cash, self.p.nx, &xIndices[0], &xWeights[0])

							row = ix + nEntries_yP1
							self.I[ii] = row
							self.I[ii2] = row

							self.J[ii] = xIndices[0] + nEntries_yP2
							self.J[ii2] = xIndices[1] + nEntries_yP2

							self.V[ii] = Pytrans * self.income.yTdist[iyT] * xWeights[0]
							self.V[ii2] = Pytrans * self.income.yTdist[iyT] * xWeights[1]
							ii += 2
							ii2 += 2

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def maximizeValueFromSwitching(self, final=False):
		"""
		Updates valueSwitch via a maximization over u(c) + beta * EMAX(c)
		at each point in the (x,z,yP)-space by computing u(c) and interpolating 
		EMAX(c) at each iteration.
		"""
		cdef:
			long iyP, ix, ii, iz, ic
			double xval, maxAdmissibleC, tmp, cSwitch, dst
			double[:] emaxVec, yderivs
			double[:] risk_aver_grid, discount_factor_grid
			double[:] sections
			double bounds[NSECTIONS][2]
			double gssResults[2]
			double cVals[NSECTIONS+2]
			double funVals[NSECTIONS+2],
			double cOptimal, vOptimal
			long iOptimal
			bint inactionFound
			double inactionPoints[2]
			double vSwitch, bestInactionPoint
			FnArgs fargs
			objectiveFn iteratorFn

		fargs.cgrid = &self.grids.c_flat[0]
		fargs.deathProb = self.p.deathProb
		fargs.nc = self.p.nc

		if self.p.risk_aver_grid.size > 1:
			fargs.hetType = 1
			fargs.timeDiscount = self.p.timeDiscount
		elif self.p.discount_factor_grid.size > 1:
			fargs.hetType = 2
			fargs.riskAver = self.p.riskAver
		else:
			fargs.hetType = 0
			fargs.timeDiscount = self.p.timeDiscount
			fargs.riskAver = self.p.riskAver

		sections = np.linspace(0, 1, num=NSECTIONS+1)

		emaxVec = np.zeros(self.p.nc)
		yderivs = np.zeros(self.p.nc)

		fargs.emaxVec = &emaxVec[0]
		fargs.yderivs = &yderivs[0]

		iteratorFn = <objectiveFn> self.findValueAtState

		if not final:
			self.cSwitchingPolicy = np.zeros((self.p.nx,1,self.p.nz,self.p.nyP),
				order='F')
			self.valueSwitch = np.zeros((self.p.nx,1,self.p.nz,self.p.nyP),
				order='F')
		else:
			self.inactionRegion = np.zeros((self.p.nx,2,self.p.nz,self.p.nyP),
				order='F')

		for iyP in range(self.p.nyP):
			for ix in range(self.p.nx-1, -1, -1):
				fargs.ncValid = self.validConsumptionPts[ix]
				if ix == 0:
					for iz in range(self.p.nz):
						if final:
							self.inactionRegion[ix,0,iz,iyP] = self.cSwitchingPolicy[0,0,iz,iyP]
							self.inactionRegion[ix,1,iz,iyP] = self.cSwitchingPolicy[0,0,iz,iyP]
						else:
							self.setFnArgs(&fargs, fargs.emaxVec, iyP,
								ix, iz, fargs.yderivs)

							cSwitch = self.cSwitchingPolicy[1,0,iz,iyP] \
								-(self.xgrid_curr[1] - self.xgrid_curr[0])
							cSwitch = fmax(cSwitch, self.p.cMin)
							self.cSwitchingPolicy[ix,0,iz,iyP] = cSwitch
							self.valueSwitch[ix,0,iz,iyP] = \
								self.findValueAtState(cSwitch, fargs) - self.p.adjustCost
					break
				
				xval = self.xgrid_curr[ix]
				maxAdmissibleC = fmin(xval - self.borrLimCurr, self.p.cMax)
				dst = maxAdmissibleC - self.p.cMin

				if not final:
					for ii in range(NSECTIONS):
						bounds[ii][0] = self.p.cMin + dst * sections[ii]
						bounds[ii][1] = self.p.cMin + dst * sections[ii+1]

				for iz in range(self.p.nz):
					self.setFnArgs(&fargs, fargs.emaxVec, iyP,
							ix, iz, fargs.yderivs)

					if not final:
						for ii in range(NSECTIONS):
							cfunctions.goldenSectionSearch(iteratorFn, bounds[ii][0],
								bounds[ii][1], 1e-8, &gssResults[0], fargs)
							funVals[ii] = gssResults[0]
							cVals[ii] = gssResults[1]

						# Try consuming cmin
						cVals[NSECTIONS] = self.p.cMin
						funVals[NSECTIONS] = self.findValueAtState(self.p.cMin, fargs)

						# Try consuming max amount
						cVals[NSECTIONS+1] = maxAdmissibleC
						funVals[NSECTIONS+1] = self.findValueAtState(maxAdmissibleC, fargs)

						iOptimal = cfunctions.cargmax(funVals, NSECTIONS+2)

						self.cSwitchingPolicy[ix,0,iz,iyP] = cVals[iOptimal]
						self.valueSwitch[ix,0,iz,iyP] = funVals[iOptimal] \
							- self.p.adjustCost
					else:
						inactionFound = self.lookForInactionPoints(ix, iz, iyP,
							inactionPoints, fargs)
						if inactionFound:
							# Look for lowest inaction point
							self.inactionRegion[ix,0,iz,iyP] = self.findExtremeNoSwitchPoint(
								inactionPoints[0], self.p.cMin, self.valueSwitch[ix,0,iz,iyP],
								fargs)

							# Highest inaction point
							self.inactionRegion[ix,1,iz,iyP] = self.findExtremeNoSwitchPoint(
								inactionPoints[1], maxAdmissibleC, self.valueSwitch[ix,0,iz,iyP],
								fargs)
						else:
							self.inactionRegion[ix,0,iz,iyP] = self.cSwitchingPolicy[ix,0,iz,iyP]
							self.inactionRegion[ix,1,iz,iyP] = self.cSwitchingPolicy[ix,0,iz,iyP]
		if final:
			self.inactionRegion = self.inactionRegion.reshape(
				(self.p.nx,2,self.p.nz,self.p.nyP,1), order='F')

	@cython.boundscheck(False)
	@cython.wraparound(False)
	cdef void setFnArgs(self, FnArgs *fargs, double *emaxvec, long iyP,
		long ix, long iz, double *yderivs):
		cdef:
			long ic

		if fargs.hetType == 1:
			dereference(fargs).riskAver = self.p.risk_aver_grid[iz]
		elif fargs.hetType == 2:
			dereference(fargs).timeDiscount = self.p.discount_factor_grid[iz]

		if dereference(fargs).ncValid >= 4:
			for ic in range(dereference(fargs).ncValid):
				emaxvec[ic] = self.EMAX[ix,ic,iz,iyP]
				spline.spline(&self.grids.c_flat[0], emaxvec, dereference(fargs).ncValid,
					1.0e30, 1.0e30, yderivs)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	cdef double findValueAtState(self, double cSwitch, FnArgs fargs) nogil:
		"""
		Outputs the value u(cSwitch) + beta * EMAX(cSwitch) for a given cSwitch.
		"""
		cdef double u, emax, value
		cdef double weights[2]
		cdef long indices[2]
	
		if fargs.ncValid == 1:
			emax = fargs.emaxVec[0]
		elif fargs.ncValid  < 4:
			cfunctions.getInterpolationWeights(fargs.cgrid, cSwitch,
				fargs.ncValid, &indices[0], &weights[0])
			emax = weights[0] * fargs.emaxVec[indices[0]] + weights[1] * fargs.emaxVec[indices[1]]
		else:
			spline.splint(fargs.cgrid, fargs.emaxVec, fargs.yderivs,
				fargs.ncValid, cSwitch, &emax)

		u = cfunctions.utility(fargs.riskAver, cSwitch)
		value = u + fargs.timeDiscount * (1 - fargs.deathProb) * emax

		return value

	@cython.boundscheck(False)
	@cython.wraparound(False)
	cdef bint lookForInactionPoints(self, long ix, long iz, long iyP,
		double *inactionPoints, FnArgs fargs):

		cdef:
			double[:] vNoSwitchCheck
			double cCheck[2]
			double gssResults[2]
			double vSwitch
			bint inactionFound = False
			objectiveFn iteratorFn

		vSwitch = self.valueSwitch[ix,0,iz,iyP]

		# Look for lowest point of inaction
		for ic in range(fargs.ncValid):
			if not self.willSwitch[ix,ic,iz,iyP]:
				inactionPoints[0] = self.grids.c_flat[ic]
				inactionFound = True
				break

		if inactionFound:
			# Look for high inaction point
			for ic in range(fargs.ncValid-1, -1, -1):
				if not self.willSwitch[ix,ic,iz,iyP]:
					inactionPoints[1] = self.grids.c_flat[ic]
					return True

			inactionPoints[1] = inactionPoints[0]
			return True
		else:
			# Look between most promising two consumption values
			vNoSwitchCheck = np.zeros(fargs.ncValid)
			for ic in range(fargs.ncValid):
				vNoSwitchCheck[ic] = self.valueNoSwitch[ix,ic,iz,iyP]

			i1 = np.argmax(vNoSwitchCheck)
			vNoSwitchCheck[i1] = -1e9
			i2 = np.argmax(vNoSwitchCheck)
			
			cCheck[0] = fmin(self.grids.c_flat[i1], self.grids.c_flat[i2])
			cCheck[1] = fmax(self.grids.c_flat[i1], self.grids.c_flat[i2])

			# Look between these points for a no-switching point
			iteratorFn = <objectiveFn> self.findValueAtState
			cfunctions.goldenSectionSearch(iteratorFn, cCheck[0],
				cCheck[1], 1e-8, &gssResults[0], fargs)

			if gssResults[0] >= vSwitch:
				inactionPoints[0] = gssResults[1]
				inactionPoints[1] = gssResults[1]
				return True
			else:
				return False

	@cython.boundscheck(False)
	@cython.wraparound(False)
	cdef double findExtremeNoSwitchPoint(self, double x0,
		double bound, double vSwitch, FnArgs fargs):

		cdef:
			double vNoSwitch, tol = 1e-9
			double xb, xg, xm
			long maxIters = long(1e5)
			long it = 0

		xb = x0
		xg = bound
		
		while (fabs(xb - xg) > tol) and (it < maxIters):
			xm = (xb + xg) / 2.0
			if self.findValueAtState(xm, fargs) >= vSwitch:
				xb = xm
			else:
				xg = xm

			it += 1

		if self.findValueAtState(xb, fargs) >= vSwitch:
			return xb
		else:
			return -1

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def updateValueNoSwitch(self):
		"""
		Updates valueNoSwitch via valueNoSwitch(c) = u(c) + beta * EMAX(c)
		"""
		cdef long ix, ic, nvalid

		self.valueNoSwitch = functions.utilityMat(self.p.risk_aver_grid, self.grids.c_wide) \
			+ np.asarray(self.p.discount_factor_grid_wide) * (1 - self.p.deathProb) \
			* np.asarray(self.EMAX)

		# Force switching if current consumption level might imply
		# that borrowing constraint is invalidated next period
		for ix in range(self.p.nx):
			nvalid = self.validConsumptionPts[ix]
			for ic in range(nvalid, self.p.nc):
				self.valueNoSwitch[ix,ic,:,:] = np.nan

	def resetParams(self, newParams):
		self.p = newParams