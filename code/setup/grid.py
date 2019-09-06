import numpy as np
import numpy.matlib as matlib

class Grid:
	"""
	This class creates and stores the asset and 
	consumption grids.
	"""

	def __init__(self, params, income):
		self.params = params

		self.s = {	'vec': None,
					'matrix': None
					}

		self.x = {	'matrix': None
					}

		self.a = {	'vec': None
					'matrix': None
					}

		self.createSavingGrid()

		self.createCashGrid(income)

	def createSavingGrid(self):
		sgrid = np.linspace(0,1,num=self.params.sGridPts)
		sgrid = sgrid ** (1 / self.params.sGridCurv)
		sgrid = self.params.borrowLim \
			+ (self.params.sMax - self.params.borrowLim) * sgrid

		sgrid = self.enforceMinGridSpacing(sgrid)

		self.s['vec'] = sgrid
		self.s['matrix'] = matlib.repmat(sgrid,1,
				params.cGridPts,params.nyP)

	def createCashGrid(self, income):
		# minimum of income along the yT dimension
		minyT = income['netymat'].min(axis=1)
		minyT =  np.kron(minyT,np.ones((self.params.xGridPts,1)))

		xgrid = self.s['matrix'] + minyT
		xgrid = self.enforceMinGridSpacing(xgrid)

		xgrid = matlib.repmat(xgrid,
			.params.cGridPts*self.params.nyP,1)
		self.x['matrix'] = xgrid.reshape((
				self.params.xGridPts,
				self.params.cGridPts,
				self.params.nyP))

	def createAssetGrid(self):
		agrid = np.linspace(0,1,self.params.aGridPts)
		agrid = agrid ** (1 / self.params.aGridCurv)
		agrid = self.params.borrowLim \
			+ (self.params.xMax - self.params.borrowLim) * agrid
		agrid = self.enforceMinGridSpacing(agrid)

		self.a['vec'] = agrid
		self.a['matrix'] = matlib.repmat(agrid,
			params.cGridPts*self.params.nyP,1)

	def enforceMinGridSpacing(self, grid_in):
		grid_adj = grid_in
		for i in range(grid.size-1):
			if grid_adj(i+1) - grid_adj(i) < self.params.minGridSpacing:
				grid_adj(i+1) = grid_adj(i) + self.params.minGridSpacing

		return grid_adj
