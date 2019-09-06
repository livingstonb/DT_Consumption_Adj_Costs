import numpy as np
from scipy.interpolate import RegularGridInterpolator
from aux import aux

class Model:
	def __init__(self, params, income, grids, nextModel=None, nextMPCShock=0):
		self.p = params
		self.grids = grids
		self.income = income

		self.nextMPCShock = nextMPCShock

		self.dims = (params.nx, params.nc, params.nyP)
		self.dims_yT = self.dims + (self.p.nyT,)

		self.xp_s = None
		self.xp_s_T = None

		self.nextModel = nextModel

		# create x, c vectors of length (nx*nc) for interpolator
		self.xgridLong = self.grids.x['matrix'][:,:,1]

	def solveEGP(self):
		conGuess = self.makePolicyGuess()

		self.compute_xprime_s()
		self.compute_Emat()

		self.iterateBackward(conGuess)

	def makePolicyGuess(self):
		# to avoid a degenerate guess, adjust for low r...
		returns = self.p.r + 0.001 * (self.p.r<0.0001)
		conGuess = returns * self.grids.x['matrix']
		return conGuess

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
			con_s = aux.marginalUtility(self.p.riskAver,muc_s)

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
		self.Emat = np.kron(self.income.yPtrans,np.eye(self.p.nx*self.p.nc))

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
		mucnext = aux.marginalUtility(self.p.riskAver,c_xp)
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


