
import numpy as np
cimport numpy as np

cimport cython
from cython.parallel cimport prange
from cython.operator cimport dereference

from misc import functions
from misc cimport cfunctions
from misc.cfunctions cimport objectiveFn

from Grid cimport Grid
from Params cimport Params
from Income cimport Income

import pandas as pd
from scipy import sparse

from libc.math cimport fmin, fmax, fabs, log, pow

cdef double INV_GOLDEN_RATIO = 0.61803398874989479150
cdef double INV_GOLDEN_RATIO_SQ = 0.38196601125

cdef class CModel:
	"""
	Base class which serves as the workhorse of the model.
	"""
	cdef:
		# parameters and grids
		public Params p
		public Income income
		public Grid grids

		# xgrid for this period
		public double[:] xgrid_curr

		# xgrid for next period
		public double[:] xgrid_next

		# value of the shock next period, used for MPCs out of news
		public double nextMPCShock

		# tuples of dimension lengths
		readonly tuple dims, dims_yT

		# borrowing limit, adjusted for next mpc shock if necessary
		public double borrLimCurr, borrLimNext

		# value functions
		public double[:,:,:,:] valueNoSwitch, valueSwitch, valueFunction

		# EMAX
		public double[:,:,:,:] EMAX, EMAX_HTM
		public double[:,:] htmGrid

		# Temp variables
		public double[:] temp_emaxVec, temp_emaxHtmVec, temp_htmGrid
		public double temp_timeDiscount, temp_riskAver
		public long temp_ncValid

		# Number of valid consumption points at each x
		long[:] validConsumptionPts

		# variables for construction of interpMat
		long [:] I, J
		double [:] V

		public object willSwitch, cChosen

		# policy functions
		public object cSwitchingPolicy, inactionRegion

		# sparse matrix for computing EMAX via interpolation
		public object interpMat

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
		EMAX(x,c,z,yP) := E[V(R(x-c)+yP'*yT'+tau,c,z',yP')|z,yP] = A * V
		where V is the flattened value function.
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
			object validCs

		# xgrids adjusted for news of a future shock
		self.xgrid_curr = self.grids.genAdjustedXGrid(self.borrLimCurr)
		self.xgrid_next = self.grids.genAdjustedXGrid(self.borrLimNext)

		self.mustSwitch = np.asarray(self.xgrid_curr)[:,None,None,None] \
			- np.asarray(self.grids.c_wide) \
			< self.borrLimCurr

		self.validConsumptionPts = np.zeros(self.p.nx, dtype=int)
		cgrid_np = np.asarray(self.grids.c_flat);
		for ix in range(self.p.nx):
			validCs = cgrid_np <= self.xgrid_curr[ix] - self.borrLimCurr
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

		for ic in range(self.validConsumptionPts[ix]):
			sav = xval - self.grids.c_flat[ic]

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
	def updateEMAX_HTM(self):
		cdef:
			long ix, iz, iyP1, iyP2, iyT, npts, ic, p0
			long xIndices[2]
			double xWeights[2]
			double[:] cgrid
			double emax, Pytrans, yP2, cash, maxAdmissibleC
			double inctrans, cbar, xval, assets

		npts = 10

		self.EMAX_HTM = np.zeros((self.p.nx,npts,self.p.nz,self.p.nyP))
		self.htmGrid = np.zeros((self.p.nx,npts))

		for ix in range(self.p.nx):
			cbar = self.grids.c_flat[self.validConsumptionPts[ix]-1]
			xval = self.xgrid_curr[ix]

			maxAdmissibleC = fmin(xval - self.borrLimCurr, self.p.cMax)
			cgrid = np.linspace(cbar, maxAdmissibleC, num=npts)

			for ic in range(npts):
				assets = self.p.R * (xval - cgrid[ic]) + self.nextMPCShock + self.p.govTransfer
				self.htmGrid[ix,ic] = cgrid[ic]

				for iz in range(self.p.nz):

					for iyP1 in range(self.p.nyP):
						emax = 0

						for iyP2 in range(self.p.nyP):
							Pytrans = self.income.yPtrans[iyP1, iyP2]
							yP2 = self.income.yPgrid[iyP2]

							for iyT in range(self.p.nyT):
								cash = assets + yP2 * self.income.yTgrid[iyT]

								# EMAX associated with next period may be over adjusted grid
								cfunctions.getInterpolationWeights(&self.xgrid_next[0],
									cash, self.p.nx, &xIndices[0], &xWeights[0])

								inctrans = Pytrans * self.income.yTdist[iyT]

								for p0 in range(2):
									emax += inctrans * xWeights[p0] * \
										self.valueFunction[xIndices[p0],ic,iz,iyP2]

						self.EMAX_HTM[ix,ic,iz,iyP1] = emax


	@cython.boundscheck(False)
	@cython.wraparound(False)
	def evaluateSwitching(self, final=False):
		"""
		Constructs required objects for one of two tasks:
		(1) Computing the value of switching via optimization over c.
		(2) Computing the inaction region.
		"""
		self.temp_timeDiscount = self.p.timeDiscount
		self.temp_riskAver = self.p.riskAver

		self.temp_emaxVec = np.zeros(self.p.nc)
		self.temp_emaxHtmVec = np.zeros(10)
		self.temp_htmGrid = np.zeros(10)

		if not final:
			self.maximizeValueFromSwitching()
		else:
			self.findInactionRegion()

	@cython.boundscheck(False)
	@cython.wraparound(False)
	cdef void maximizeValueFromSwitching(self):
		"""
		Computes the value function for switching consumption.
		"""
		cdef:
			long iyP, ix, ii, iz, ic
			double xval, maxAdmissibleC, cSwitch, con
			double[:] cVals, funVals, gssBounds
			double[:] gssResults
			long iOptimal
			objectiveFn iteratorFn

		gssBounds = np.zeros(self.p.nSectionsGSS+1)
		cVals = np.zeros(self.p.nSectionsGSS+2)
		funVals = np.zeros(self.p.nSectionsGSS+2)

		self.cSwitchingPolicy = np.zeros((self.p.nx,1,self.p.nz,self.p.nyP),
			order='F')
		self.valueSwitch = np.zeros((self.p.nx,1,self.p.nz,self.p.nyP),
			order='F')

		for iyP in range(self.p.nyP):
			for ix in range(self.p.nx):
				self.temp_ncValid = self.validConsumptionPts[ix]

				xval = self.xgrid_curr[ix]
				maxAdmissibleC = fmin(xval - self.borrLimCurr, self.p.cMax)
				maxAdmissibleOnGrid = self.grids.c_flat[self.temp_ncValid-1]

				# Create linearly spaced vector in gssBounds
				cfunctions.linspace(self.p.cMin, maxAdmissibleOnGrid,
					self.p.nSectionsGSS, gssBounds)
				gssBounds[self.p.nSectionsGSS] = maxAdmissibleC

				for iz in range(self.p.nz):
					self.setTempValues(ix, iz, iyP)

					for ii in range(self.p.nSectionsGSS):
						gssResults = self.goldenSectionSearch(gssBounds[ii],
							gssBounds[ii+1], 1e-10)
						funVals[ii] = gssResults[0]
						cVals[ii] = gssResults[1]

					# Try consuming cmin, cmax
					ii = self.p.nSectionsGSS
					for con in [self.p.cMin, maxAdmissibleC]:
						cVals[ii] = con
						funVals[ii] = self.findValueAtState(con)
						ii += 1

					iOptimal = cfunctions.cargmax(funVals)

					self.cSwitchingPolicy[ix,0,iz,iyP] = cVals[iOptimal]
					self.valueSwitch[ix,0,iz,iyP] = funVals[iOptimal] - self.p.adjustCost

	@cython.boundscheck(False)
	@cython.wraparound(False)
	cdef void setTempValues(self, long ix, long iz, long iyP):
		cdef long ic

		if self.p.risk_aver_grid.size > 1:
			self.temp_riskAver = self.p.risk_aver_grid[iz]
		elif self.p.discount_factor_grid.size > 1:
			self.temp_timeDiscount = self.p.discount_factor_grid[iz]

		self.temp_emaxVec[:] = self.EMAX[ix,:,iz,iyP]
		self.temp_emaxHtmVec[:] = self.EMAX_HTM[ix,:,iz,iyP]
		self.temp_htmGrid[:] = self.htmGrid[ix,:]

	@cython.boundscheck(False)
	@cython.wraparound(False)
	cdef void findInactionRegion(self):
		"""
		Computes the value function for not switching consumption.
		"""
		cdef:
			long iyP, ix, iz, ic
			double maxAdmissibleC
			bint inactionFound
			double[:] inactionPoints

		self.inactionRegion = np.zeros((self.p.nx,2,self.p.nz,self.p.nyP),
				order='F')

		for iyP in range(self.p.nyP):
			for ix in range(self.p.nx):
				self.temp_ncValid = self.validConsumptionPts[ix]
				maxAdmissibleC = fmin(self.xgrid_curr[ix] - self.borrLimCurr, self.p.cMax)

				for iz in range(self.p.nz):
					self.setTempValues(ix, iz, iyP)

					inactionPoints = self.lookForInactionPoints(ix, iz, iyP)
					if (inactionPoints[0] != -1):
						# Look for lowest inaction point
						self.inactionRegion[ix,0,iz,iyP] = self.findExtremeNoSwitchPoint(
							inactionPoints[0], self.p.cMin, self.valueSwitch[ix,0,iz,iyP])

						# Highest inaction point
						self.inactionRegion[ix,1,iz,iyP] = self.findExtremeNoSwitchPoint(
							inactionPoints[1], maxAdmissibleC, self.valueSwitch[ix,0,iz,iyP])
					else:
						self.inactionRegion[ix,0,iz,iyP] = self.cSwitchingPolicy[ix,0,iz,iyP]
						self.inactionRegion[ix,1,iz,iyP] = self.cSwitchingPolicy[ix,0,iz,iyP]

		self.inactionRegion = self.inactionRegion.reshape(
			(self.p.nx,2,self.p.nz,self.p.nyP,1), order='F')

	@cython.boundscheck(False)
	@cython.wraparound(False)
	cdef double findValueAtState(self, double cSwitch):
		"""
		Outputs the value u(cSwitch) + beta * EMAX(cSwitch) for a given cSwitch.
		"""
		cdef double u, emax
		cdef double weights[2]
		cdef long indices[2]

		if cSwitch > self.grids.c_flat[self.temp_ncValid-1]:
			emax = cfunctions.interpolate(&self.temp_htmGrid[0], cSwitch, &self.temp_emaxHtmVec[0], 10)
		else:
			emax = cfunctions.interpolate(&self.grids.c_flat[0], cSwitch, &self.temp_emaxVec[0], self.temp_ncValid)

		u = cfunctions.utility(self.temp_riskAver, cSwitch)
		return u + self.temp_timeDiscount * (1 - self.p.deathProb) * emax

	@cython.boundscheck(False)
	@cython.wraparound(False)
	cdef double[:] lookForInactionPoints(self, long ix, long iz, long iyP):
		"""
		Searches for the lowest and highest points of inaction on the consumption grid.
		If none exist, then a golden section search routine is called to look for a point
		of inaction.
		"""
		cdef:
			double[:] vNoSwitchCheck = np.zeros(self.temp_ncValid)
			double inactionPoints[2]
			double cCheck[2]
			double[:] gssResults1, gssResults2
			double vSwitch
			long ic, i1, i2
			bint inactionFound = False

		vSwitch = self.valueSwitch[ix,0,iz,iyP]
		inactionPoints[0] = -1
		inactionPoints[1] = -1

		# Look for lowest and highest points of inaction
		for ic in range(self.temp_ncValid):
			if not self.willSwitch[ix,ic,iz,iyP]:
				# This is a point of inaction
				if not inactionFound:
					# Update lowest inaction found
					inactionPoints[0] = self.grids.c_flat[ic]
					inactionFound = True
				
				# Update highest inaction found
				inactionPoints[1] = self.grids.c_flat[ic]

		if not inactionFound:
			# Look between most promising two consumption values
			vNoSwitchCheck[:] = self.valueNoSwitch[ix,:self.temp_ncValid,iz,iyP]

			i1 = np.argmax(vNoSwitchCheck)
			vNoSwitchCheck[i1] = -1e9
			i2 = np.argmax(vNoSwitchCheck)
			
			cCheck[0] = fmin(self.grids.c_flat[i1], self.grids.c_flat[i2])
			cCheck[1] = fmax(self.grids.c_flat[i1], self.grids.c_flat[i2])

			# Look between these points for a no-switching point
			gssResults1 = self.goldenSectionSearch(cCheck[0],
				cCheck[1], 1e-10)

			if gssResults1[0] >= vSwitch:
				inactionPoints[0] = gssResults1[1]
				inactionPoints[1] = gssResults1[1]
			else:
				# Look closer to x
				gssResults2 = self.goldenSectionSearch(cCheck[1],
					self.xgrid_curr[ix]-self.borrLimCurr, 1e-10)
				if gssResults2[0] >= vSwitch:
					inactionPoints[0] = gssResults2[1]
					inactionPoints[1] = gssResults2[1]

		return inactionPoints

	@cython.boundscheck(False)
	@cython.wraparound(False)
	cdef double findExtremeNoSwitchPoint(self, double x0,
		double bound, double vSwitch):
		"""
		Searches for the largest (or smallest) consumption value that
		implies inaction. The algorithm starts at x0, which is assumed to be
		a point of inaction, and searches toward bound.
		"""
		cdef:
			double vNoSwitch, tol = 1e-10
			double xb, xg, xm
			long maxIters = long(1e6)
			long it = 0

		xb = x0
		xg = bound
		
		while (fabs(xb - xg) > tol) and (it < maxIters):
			xm = (xb + xg) / 2.0
			if self.findValueAtState(xm) >= vSwitch:
				xb = xm
			else:
				xg = xm

			it += 1

		return xb

	def resetParams(self, newParams):
		self.p = newParams

	@cython.boundscheck(False)
	@cython.wraparound(False)
	cdef double[:] goldenSectionSearch(self, double a, double b, 
		double tol):
		"""
		This function iterates over the objective function f using
		the golden section search method in the interval (a,b).

		The maximum function value is supplied to out[0] and the
		maximizer is supplied to out[1]. Arguments of f must be arg1,
		arg2, and fparams.

		Algorithm taken from Wikipedia.
		"""
		cdef:
			double out[2]
			double c, d, diff, fc, fd

		diff = b - a

		c = a + diff * INV_GOLDEN_RATIO_SQ
		d = a + diff * INV_GOLDEN_RATIO 

		fc = -self.findValueAtState(c)
		fd = -self.findValueAtState(d)

		while fabs(c - d) > tol:
			if fc < fd:
				b = d
				d = c
				fd = fc

				diff = diff * INV_GOLDEN_RATIO
				c = a + diff * INV_GOLDEN_RATIO_SQ
				fc = -self.findValueAtState(c)
			else:
				a = c
				c = d
				fc = fd
				diff = diff * INV_GOLDEN_RATIO
				d = a + diff * INV_GOLDEN_RATIO
				fd = -self.findValueAtState(d)

		if fc < fd:
			out[0] = -fc
			out[1] = c
		else:
			out[0] = -fd
			out[1] = d

		return out