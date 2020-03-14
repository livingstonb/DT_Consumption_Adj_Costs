import numpy as np
cimport numpy as np
import numpy.matlib as matlib
import pandas as pd

cdef class Params:
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
		self.maxIters = long(2e4)
		self.tol = 1e-7
		self.nSim = long(1e5) # number of draws to sim distribution
		self.tSim = 80 # number of periods to simulate

		# beta iteration
		self.tolWealthTarget = 1e-7
		self.wealthTarget = 3.5
		self.wealthIters = 200;

		# mpc options
		self.NsimMPC = long(2e5) # number of draws to sim MPCs
		self.MPCshocks = [-0.081, -0.0405, -0.0081, 0.0081, 0.0405, 0.081, 0]
		self.nshocks = len(self.MPCshocks)

		# fraction of mean annual income
		self.wealthConstraints = [0,0.005,0.01,0.015,0.02,0.05,0.1,0.15]

		# wealth percentiles to compute
		self.wealthPercentiles = [10,25,50,75,90,99,99.9]

		# cash-on-hand / savings grid parameters
		self.xMax = 50 # max of saving grid
		self.nx = 50
		self.nxLow = 5
		self.xGridTerm1Wt = 0.01
		self.xGridTerm1Curv = 0.9
		self.xGridCurv = 0.15
		self.borrowLim = 0

		# consumption grid
		self.nc = 75
		self.cMin = 1e-6
		self.cMax = 5
		self.cGridTerm1Wt = 0.005
		self.cGridTerm1Curv = 0.9
		self.cGridCurv = 0.15

		# options
		self.MPCsOutOfNews = False

		# returns (annual)
		self.r = 0.02
		self.R = 1.0 + self.r

		# death probability (annual)
		self.deathProb = 1.0/50.0

		self.Bequests = True

		# preferences
		self.riskAver = 1
		self.adjustCost = 1
		self.timeDiscount = 0.8
		self.risk_aver_grid = np.array([0.0],dtype=float) # riskAver is added to this
		self.discount_factor_grid = np.array([0.0],dtype=float) # timeDiscount is added to this

		# gov transfer
		self.govTransfer = 0.0081 * 2.0 * 4.0

		#-----------------------------------#
		#        OVERRIDE DEFAULTS          #
		#-----------------------------------#

		if params_dict:
			for parameter, value in params_dict.items():
				if hasattr(self, parameter):
					setattr(self, parameter, value)
				else:
					raise Exception(f'"{parameter}" is not a valid parameter')

		self.risk_aver_grid = self.risk_aver_grid.astype(float)
		self.discount_factor_grid = self.discount_factor_grid.astype(float)

		if (self.risk_aver_grid.size>1) and (self.discount_factor_grid.size>1):
			raise Exception('Cannot have both IES and discount factor heterogeneity')
		else:
			self.nz = max(self.risk_aver_grid.size, self.discount_factor_grid.size)

		#-----------------------------------#
		#     ADJUST TO QUARTERLY FREQ      #
		#-----------------------------------#

		if self.freq == 4:
			self.adjustToQuarterly()

		self.discount_factor_grid += self.timeDiscount
		self.risk_aver_grid += self.riskAver

		#-----------------------------------#
		#     CREATE USEFUL OBJECTS         #
		#-----------------------------------#
		self.discount_factor_grid_wide = \
			self.discount_factor_grid.reshape(
				(1,1,self.discount_factor_grid.size,1))

		#-----------------------------------#
		#     SERIES FOR OUTPUT TABLE       #
		#-----------------------------------#
		index = [	'Discount factor (annualized)',
					'Discount factor (quarterly)',
					'Adjustment cost',
					'RA Coeff',
					'Returns r',
					'Frequency',
					]
		data = [	self.timeDiscount ** self.freq,
					self.timeDiscount ** (self.freq/4),
					self.adjustCost,
					self.riskAver,
					self.r,
					self.freq,
					]
		self.series = pd.Series(data=data,index=index)


	def adjustToQuarterly(self):
		"""
		Adjusts relevant parameters such as returns to
		the quarterly frequency, if freq = 4 is chosen.

		The number of periods for simulation is multiplied
		by 4 to compensate for longer convergence time.
		"""
		if self.freq == 1:
			return

		self.R = (1.0 + self.r) ** 0.25
		self.r = self.R - 1.0
		self.tSim *= 4
		self.deathProb = 1.0 - (1.0 - self.deathProb) ** 0.25
		self.timeDiscount = self.timeDiscount ** 0.25
		self.adjustCost /= 4.0
		self.govTransfer /= 4.0

	def addIncomeParameters(self, income):
		self.nyP = income.nyP
		self.nyT = income.nyT

	def resetDiscountRate(self, newTimeDiscount):
		self.discount_factor_grid = np.asarray(self.discount_factor_grid) \
			+ newTimeDiscount - self.timeDiscount
		self.discount_factor_grid_wide = np.asarray(self.discount_factor_grid_wide) \
			+ newTimeDiscount - self.timeDiscount
		self.timeDiscount = newTimeDiscount
		self.series['Discount factor (annualized)'] = newTimeDiscount ** self.freq
		self.series['Discount factor (quarterly)'] = newTimeDiscount ** (self.freq/4)

	def resetAdjustCost(self, newAdjustCost):
		self.adjustCost = newAdjustCost
		self.series['Adjustment cost'] = newAdjustCost

	def setParam(self, name, value, printReset=False):
		if name == 'adjustCost':
			self.series['Adjustment cost'] = value
		elif name == 'timeDiscount':
			self.discount_factor_grid = np.asarray(self.discount_factor_grid) \
				+ value - self.timeDiscount
			self.discount_factor_grid_wide = np.asarray(self.discount_factor_grid_wide) \
				+ value - self.timeDiscount
			self.series['Discount factor (annualized)'] = value ** self.freq
			self.series['Discount factor (quarterly)'] = value ** (self.freq/4)

		if hasattr(self, name):
			setattr(self, name, value)
		else:
			raise Exception(f'"{name}" is not a valid parameter')

		if printReset:
			print(f"The parameter '{name}' was reset to {value}.")

	def getParam(self, name):
		if hasattr(self, name):
			return getattr(self, name)
		else:
			raise Exception(f'"{name}" is not a valid parameter')

	def reportFinalParameters(self):
		index = [	'r',
					'government transfer',
					'risk aversion',
					'death probability',
					'number of discount factor pts']
		data = [self.r,
				self.govTransfer,
				self.riskAver,
				self.deathProb,
				self.discount_factor_grid.shape[0]]
		final_params = pd.Series(data=data,index=index)
		final_params = pd.concat([self.series, final_params])
		print(final_params)