from scipy.io import loadmat
import numpy as np
import numpy.matlib as matlib

cdef class Params:
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
	cdef:
		public str name, locIncomeProcess
		public int index, freq, nyP, nyT, nz
		public long maxIters, nSim, tSim
		public double tol
		public double tolWealthTarget, wealthTarget
		public bint MPCsOutOfNews, Bequests
		public bint noTransIncome, noPersIncome
		public long NsimMPC
		public list MPCshocks, wealthConstraints, wealthPercentiles
		public int xMax, nx
		public double xGridCurv, borrowLim, minGridSpacing
		public int nc
		public double cMin, cMax, cGridCurv
		public double r, R, deathProb
		public double riskAver, adjustCost, timeDiscount

	def __init__(self, params_dict=None):

		#-----------------------------------#
		#        SET DEFAULT VALUES         #
		#-----------------------------------#

		# identifiers
		self.name = 'Unnamed'
		self.index = 0

		# 1 (annual) or 4 (quarterly)
		self.freq = 4

		# path to income file
		self.locIncomeProcess = ''

		# income grid sizes
		self.nyP = 1
		self.nyT = 1
		self.noTransIncome = False
		self.noPersIncome = False

		# preference/other heterogeneity
		self.nz = 1

		# computation
		self.maxIters = long(1e5)
		self.tol = 1e-7
		self.nSim = long(1e5) # number of draws to sim distribution
		self.tSim = 100 # number of periods to simulate

		# beta iteration
		self.tolWealthTarget = 1e-7
		self.wealthTarget = 3.5

		# mpc options
		self.NsimMPC = long(2e5) # number of draws to sim MPCs
		self.MPCshocks = [-1e-5,-0.01,-0.1,1e-5,0.01,0.1]

		# fraction of mean annual income
		self.wealthConstraints = [0,0.005,0.01,0.015,0.02,0.05,0.1,0.15]

		# wealth percentiles to compute
		self.wealthPercentiles = [10,25,50,75,90,99,99.9]

		# cash-on-hand / savings grid parameters
		self.xMax = 50 # max of saving grid
		self.nx = 40
		self.xGridCurv = 0.2
		self.borrowLim = 0
		self.minGridSpacing = 0.0005

		# consumption grid
		self.nc = 50
		self.cMin = 1e-6
		self.cMax = 3
		self.cGridCurv = 0.1

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
		self.timeDiscount = 0.8

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

		self.R = (1 + self.r) ** 0.25
		self.r = self.R - 1
		self.tSim = self.tSim * 4
		self.deathProb = 1 - (1 - self.deathProb) ** 0.25
		self.timeDiscount = self.timeDiscount ** 0.25

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
		if self.p.noPersIncome:
			self.nyP = 1
			self.yPdist = np.array([1.0]).reshape((1,1))
			self.yPgrid = np.array([1.0]).reshape((1,1))
			self.yPtrans = np.array([1.0]).reshape((1,1))
		else:
			self.logyPgrid = matFile['discmodel1']['logyPgrid'][0][0]
			self.nyP = self.logyPgrid.size
			self.logyPgrid = self.logyPgrid.reshape((self.nyP,1))

			self.yPdist = matFile['discmodel1']['yPdist'][0][0].reshape((self.nyP,1))
			self.yPtrans = matFile['discmodel1']['yPtrans'][0][0]

			self.yPgrid = np.exp(self.logyPgrid)

		self.yPgrid = self.yPgrid / np.dot(self.yPdist.T,self.yPgrid)
		self.yPcumdist = np.cumsum(self.yPdist).reshape((self.nyP,1))
		self.yPcumdistT = self.yPcumdist.T
		self.yPcumtrans = np.cumsum(self.yPtrans,axis=1)

		# transitory income
		if self.p.noTransIncome:
			self.nyT = 1
			self.yTdist = np.array([1.0]).reshape((1,1))
			self.yTgrid = np.array([1.0]).reshape((1,1))
		else:
			self.logyTgrid = matFile['discmodel1']['logyTgrid'][0][0]
			self.nyT = self.logyTgrid.size
			self.logyTgrid = self.logyTgrid.reshape((self.nyT,1))

			self.yTdist = matFile['discmodel1']['yTdist'][0][0].reshape((self.nyT,1))
			
			self.yTgrid = np.exp(self.logyTgrid)

		self.yTgrid = self.yTgrid / np.dot(self.yTdist.T,self.yTgrid)
		self.yTgrid = self.yTgrid / self.p.freq

		self.yTcumdist = np.cumsum(self.yTdist).reshape((self.nyT,1))
		self.yTcumdistT = self.yTcumdist.T

	def createOtherIncomeVariables(self):
		# matrix of all income values, dimension nyP by nyT
		self.ymat = np.matmul(self.yPgrid,self.yTgrid.T)


cdef class GridDouble:
	cdef public double[:] flat
	cdef public double[:,:] vec
	cdef public double[:,:,:,:] wide
	cdef public double[:,:,:,:] matrix

cdef class GridInt:
	cdef public long[:] flat
	cdef public long[:,:] vec
	cdef public long[:,:,:,:] wide
	cdef public long[:,:,:,:] matrix


cdef class GridCreator:
	cdef:
		object p
		public tuple matrixDim
		public GridDouble c, x
		public GridInt z
		public object mustSwitch

	def __init__(self, params, income):
		self.p = params

		self.matrixDim = (	params.nx,
							params.nc,
							params.nz,
							params.nyP,
							)

		self.createCashGrid(income)

		self.createConsumptionGrid()

		self.mustSwitch = np.asarray(self.c.matrix) > np.asarray(self.x.matrix)

		self.create_zgrid()

	def createCashGrid(self, income):
		self.x  = GridDouble()

		xmin = self.p.borrowLim + income.ymat.min().min()

		xgrid = np.linspace(0,1,num=self.p.nx)
		xgrid = xgrid.reshape((self.p.nx,1))
		xgrid = xgrid ** (1 / self.p.xGridCurv)
		xgrid = xmin \
			+ (self.p.xMax - xmin) * xgrid

		xgrid = self.enforceMinGridSpacing(xgrid)

		self.x.flat = xgrid.flatten()
		self.x.vec = xgrid
		self.x.wide = xgrid.reshape((-1,1,1,1))
		self.x.matrix = np.tile(self.x.wide,
			(1,self.p.nc,self.p.nz,self.p.nyP))

	def createConsumptionGrid(self):
		self.c = GridDouble()

		cgrid = np.linspace(0,1,num=self.p.nc)
		cgrid = cgrid.reshape((self.p.nc,1))
		cgrid = cgrid ** (1 / self.p.cGridCurv)
		cgrid = self.p.cMin \
			+ (self.p.cMax - self.p.cMin) * cgrid

		cgrid = self.enforceMinGridSpacing(cgrid)

		self.c.flat = cgrid.flatten()
		self.c.vec = cgrid
		self.c.wide = cgrid.reshape((1,self.p.nc,1,1))
		self.c.matrix = np.tile(self.c.wide,
			(self.p.nx,1,self.p.nz,self.p.nyP))

	def create_zgrid(self):
		self.z = GridInt()

		zgrid = np.arange(self.p.nz).reshape((-1,1))
		self.z.flat = zgrid.flatten()
		self.z.vec = zgrid
		self.z.wide = zgrid.reshape((1,1,self.p.nz,1))
		self.z.matrix = np.tile(self.z.wide,
			(self.p.nx,self.p.nc,1,self.p.nyP))

	def enforceMinGridSpacing(self, grid_in):
		grid_adj = grid_in
		for i in range(grid_in.size-1):
			if grid_adj[i+1] - grid_adj[i] < self.p.minGridSpacing:
				grid_adj[i+1] = grid_adj[i] + self.p.minGridSpacing

		return grid_adj