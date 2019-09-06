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

	def __init__(self, params_dict):

		#-----------------------------------#
		#        SET DEFAULT VALUES         #
		#-----------------------------------#

		# identifiers
		self.name = 'Unnamed'
		self.index = None

		# 1 (annual) or 4 (quarterly)
		self.freq = None

		# path to income file
		self.IncomeProcess = None

		# income grid sizes
		self.nyF = None
		self.nyP = None
		self.nyT = None

		# computation
		self.maxIterEGP = 1e5
		self.tolIterEGP = 1e-6
		self.nSim = 2e5 # number of draws to sim distribution
		self.tSim = 500 # number of periods to simulate

		# beta iteration
		self.maxIterBeta = 50
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
		self.nGridPts = 100
		self.xGridPts = 100
		self.aGridPts = 100
		self.sGridCurv = 0.2
		self.aGridCurv = 0.2
		self.borrowLim = 0
		self.minGridSpacing = 0.001

		# consumption grid
		self.cMin = 0.001
		self.cMax = 3
		self.cGridPts = 100
		self.cGridCurv = 0.3

		# options
		self.MPCsOutOfNews = False

		# returns (annual)
		self.r = 0.02
		self.R = 1 + self.r

		# death probability (annual)
		self.deathProb = 1/50

		# preferences
		self.riskAver = 1

		#-----------------------------------#
		#        OVERRIDE DEFAULTS          #
		#-----------------------------------#

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

		self.R = (1 + self.r) ^ (1/4)
		self.r = self.R - 1
		self.tSim = self.tSim * 4
		self.deathProb = 1 - (1 - self.deathProb) ^ (1/4)

	def addIncomeParameters(self, income):
		self.nyF = income.nyF
		self.nyP = income.nyP
		self.nyT = income.nyT