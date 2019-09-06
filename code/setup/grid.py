import numpy as np
import numpy.matlib as matlib

class Grid:
	"""
	This class creates and stores the asset and 
	consumption grids.
	"""

	def __init__(self, params, income):
		self.params = params
		self.nc = params.nc
		self.nx = params.nx

		# saving grid
		self.s = {	'vec': None,
					'matrix': None
					}

		# consumption grid
		self.c = {	'vec': None,
					'nx_nc': None,
					'matrix': None
					}

		# cash-on-hand grid
		self.x = {	'matrix': None
					}

		self.createSavingGrid()

		self.createConsumptionGrid()

		self.createCashGrid(income)

	def createSavingGrid(self):
		sgrid = np.linspace(0,1,num=self.nx)
		sgrid = sgrid.reshape((self.nx,-1))
		sgrid = sgrid ** (1 / self.params.sGridCurv)
		sgrid = self.params.borrowLim \
			+ (self.params.sMax - self.params.borrowLim) * sgrid

		sgrid = self.enforceMinGridSpacing(sgrid)

		self.s['vec'] = sgrid
		self.s['matrix'] = matlib.repmat(sgrid,1,
				self.nc*self.params.nyP).reshape(
				(self.nx,self.params.nc,self.params.nyP))

	def createConsumptionGrid(self):
		cgrid = np.linspace(0,1,num=self.nc)
		cgrid = cgrid.reshape((self.nc,-1))
		cgrid = cgrid ** (1 / self.params.cGridCurv)
		cgrid = self.params.cMin \
			+ (self.params.cMax - self.params.cMin) * cgrid

		cgrid = self.enforceMinGridSpacing(cgrid)

		self.c['vec'] = cgrid
		self.c['nx_nc'] = np.kron(cgrid,np.ones((self.nx,1)))
		self.c['matrix'] = matlib.repmat(self.c['nx_nc'],self.params.nyP,1).reshape(
				(self.nx,self.params.nc,self.params.nyP))

	def createCashGrid(self, income):
		# minimum of income along the yT dimension
		minyT = income.ymat.min(axis=1)
		minyT = np.kron(minyT,np.ones((self.nx,self.nc)))
		minyT = minyT.reshape((self.nx,self.nc,self.params.nyP))

		self.x['matrix'] = self.s['matrix'] + minyT

	def enforceMinGridSpacing(self, grid_in):
		grid_adj = grid_in
		for i in range(grid_in.size-1):
			if grid_adj[i+1] - grid_adj[i] < self.params.minGridSpacing:
				grid_adj[i+1] = grid_adj[i] + self.params.minGridSpacing

		return grid_adj
