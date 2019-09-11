from scipy.io import loadmat
import numpy as np
import numpy.matlib as matlib

class Params:
	"""
	This class stores the parameters of the model.

	The default values are set in the class constructor
	and to override these values, a dictionary is passed
	into the constructor containing parameter values
	that need to be overriden (e.g. {'tSim':200}).

	Variables such as returns and the probability of
	death must be set on an annual basis. The method
	adjustToQuarterly() adjusts these variables to
	quarterly values.
	"""

	def __init__(self, params_dict=None):

		#-----------------------------------#
		#        SET DEFAULT VALUES         #
		#-----------------------------------#

		# identifiers
		self.name = 'Unnamed'
		self.index = None

		# 1 (annual) or 4 (quarterly)
		self.freq = 4

		# path to income file
		self.locIncomeProcess = None

		# income grid sizes
		self.nyP = None
		self.nyT = None

		# preference/other heterogeneity
		self.nz = 1

		# computation
		self.maxIters = 1e5
		self.tol = 1e-6
		self.nSim = 2e5 # number of draws to sim distribution
		self.tSim = 500 # number of periods to simulate

		# beta iteration
		self.tolWealthTarget = 1e-7
		self.wealthTarget = 3.5
		self.iterateBeta = False

		# mpc options
		self.NsimMPC = 2e5 # number of draws to sim MPCs
		self.MPCshocks = [-1e-5,-0.01,-0.1,1e5,0.01,0.1]

		# fraction of mean annual income
		self.wealthConstraints = [0,0.005,0.01,0.015,0.02,0.05,0.1,0.15]

		# wealth percentiles to compute
		self.wealthPercentiles = [10,25,50,75,90,99,99.9]

		# cash-on-hand / savings grid parameters
		self.sMax = 150 # max of saving grid
		self.nx = 35
		self.sGridCurv = 0.2
		self.aGridCurv = 0.2
		self.borrowLim = 0
		self.minGridSpacing = 0.001

		# consumption grid
		self.nc = 30
		self.cMin = 0.001
		self.cMax = 3
		self.cGridCurv = 0.3

		# options
		self.MPCsOutOfNews = False

		# returns (annual)
		self.r = 0.02
		self.R = 1 + self.r

		# death probability (annual)
		self.deathProb = 1/50
		self.Bequests = True

		# preferences
		self.riskAver = 1
		self.adjustCost = 1
		self.timeDiscount = 0.98

		#-----------------------------------#
		#        OVERRIDE DEFAULTS          #
		#-----------------------------------#

		if params_dict:
			for parameter, value in params_dict.items():
				if hasattr(self,parameter):
					setattr(self,parameter,value)
				else:
					raise Exception(f'"{parameter}" is not a valid parameter')

		#-----------------------------------#
		#     ADJUST TO QUARTERLY FREQ      #
		#-----------------------------------#

		if self.freq == 4:
			self.adjustToQuarterly()

	def adjustToQuarterly(self):
		"""
		Adjusts relevant parameters such as returns to
		the quarterly frequency, if freq = 4 is chosen.

		The number of periods for simulation is multiplied
		by 4 to compensate for longer convergence time.
		"""
		if self.freq == 1:
			return

		self.R = (1 + self.r) ** (1/4)
		self.r = self.R - 1
		self.tSim = self.tSim * 4
		self.deathProb = 1 - (1 - self.deathProb) ** (1/4)
		self.timeDiscount = self.timeDiscount ** (1/4)

	def addIncomeParameters(self, income):
		self.nyP = income.nyP
		self.nyT = income.nyT

class Income:
	"""
	This class stores income variables.

	Mean annual income is normalized to 1 by
	normalizing persistent income to have mean
	1 and transitory income to have mean 1 if
	frequency is annual and 1/4 if frequency
	is quarterly.
	"""

	def __init__(self, params):
		self.p = params

		self.readIncome()

		self.createOtherIncomeVariables()
		
	def readIncome(self):
		matFile = loadmat(self.p.locIncomeProcess)

		# persistent component
		self.logyPgrid = matFile['discmodel1']['logyPgrid'][0][0]
		self.nyP = self.logyPgrid.size
		self.logyPgrid = self.logyPgrid.reshape((self.nyP,-1))

		self.yPdist = matFile['discmodel1']['yPdist'][0][0].reshape((self.nyP,-1))
		self.yPtrans = matFile['discmodel1']['yPtrans'][0][0]

		self.yPgrid = np.exp(self.logyPgrid)
		self.yPgrid = self.yPgrid / np.dot(self.yPdist.T,self.yPgrid)
		self.yPcumdist = np.cumsum(self.yPdist)
		self.yPcumdistT = self.yPcumdist.T
		self.yPcumtrans = np.cumsum(self.yPtrans,axis=1)

		# transitory income
		self.logyTgrid = matFile['discmodel1']['logyTgrid'][0][0]
		self.nyT = self.logyTgrid.size
		self.logyTgrid = self.logyPgrid.reshape((self.nyT,-1))

		self.yTdist = matFile['discmodel1']['yTdist'][0][0].reshape((self.nyT,-1))
		
		self.yTgrid = np.exp(self.logyTgrid)
		self.yTgrid = self.yTgrid / np.dot(self.yTdist.T,self.yTgrid)
		self.yTgrid = self.yTgrid / self.p.freq
		self.yTcumdist = np.cumsum(self.yTdist)
		self.yTcumdistT = self.yTcumdist.T

	def createOtherIncomeVariables(self):
		# matrix of all income values, dimension nyP by nyT
		self.ymat = np.matmul(self.yPgrid,self.yTgrid.T)


class Grid:
	"""
	This class creates and stores the asset and 
	consumption grids.
	"""

	def __init__(self, params, income):
		self.params = params
		self.nc = params.nc
		self.nx = params.nx
		self.nz = params.nz

		self.matrixDim = (	self.nx,
							self.params.nc,
							self.params.nz,
							self.params.nyP,
							)

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

		# z grid
		self.z = {	'matrix': None
					}

		self.createSavingGrid()

		self.createConsumptionGrid()

		self.createCashGrid(income,params.R)

		self.create_zgrid()

	def createSavingGrid(self):
		sgrid = np.linspace(0,1,num=self.nx)
		sgrid = sgrid.reshape((self.nx,-1))
		sgrid = sgrid ** (1 / self.params.sGridCurv)
		sgrid = self.params.borrowLim \
			+ (self.params.sMax - self.params.borrowLim) * sgrid

		sgrid = self.enforceMinGridSpacing(sgrid)

		self.s['vec'] = sgrid
		self.s['matrix'] = matlib.repmat(sgrid,1,
				self.nc*self.nz*self.params.nyP).reshape(self.matrixDim)

	def createConsumptionGrid(self):
		cgrid = np.linspace(0,1,num=self.nc)
		cgrid = cgrid.reshape((self.nc,-1))
		cgrid = cgrid ** (1 / self.params.cGridCurv)
		cgrid = self.params.cMin \
			+ (self.params.cMax - self.params.cMin) * cgrid

		cgrid = self.enforceMinGridSpacing(cgrid)

		self.c['vec'] = cgrid
		self.c['nx_nc'] = np.kron(cgrid,np.ones((self.nx,1)))
		self.c['matrix'] = matlib.repmat(
			self.c['nx_nc'],self.params.nz*self.params.nyP,1
			).reshape(self.matrixDim)

	def createCashGrid(self, income, returns):
		# minimum of income along the yT dimension
		minyT = income.ymat.min(axis=1)
		self.x['matrix'] = returns * self.s['matrix'] + minyT[None,None,None,...]
		self.x['wide'] = self.x['matrix'][:,0,0,:].reshape((self.matrixDim[0],1,1,self.matrixDim[3]))

	def create_zgrid(self):
		self.z['vec'] = np.arange(self.nz).reshape((-1,1))

	def enforceMinGridSpacing(self, grid_in):
		grid_adj = grid_in
		for i in range(grid_in.size-1):
			if grid_adj[i+1] - grid_adj[i] < self.params.minGridSpacing:
				grid_adj[i+1] = grid_adj[i] + self.params.minGridSpacing

		return grid_adj