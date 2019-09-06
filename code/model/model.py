import numpy as np
from scipy.interpolate import RegularGridInterpolator

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

			if np.mod(it,50) == 0:
				print(f'    EGP iteration {it}')

			# c(x)
			con = con.reshape(dimTuple)

			# c(x')
			con_xp = self.get_c_xprime(con)

			# MUC in current period, from Euler equation

	def compute_xprime_s(self):
		# find xprime as a function of saving
		income_x = np.kron(self.income.ymat,np.ones((self.p.nx*self.p.nc,1)))
		income_x = income_x.reshape(self.dims_yT)

		self.xp_s = self.p.R * self.grids.s['matrix'][...,None] \
			+ income_x + self.nextMPCShock

		self.xp_s_T = self.xp_s.reshape((-1,self.p.nyT))

	def compute_Emat(self):
		self.Emat = np.kron(self.income.yPtrans,np.eye(self.p.nx))

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