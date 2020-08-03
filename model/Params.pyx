import numpy as np
cimport numpy as np
import numpy.matlib as matlib
import pandas as pd
import Defaults

cdef class Params:
	def __init__(self, params_dict=None):
		#-----------------------------------#
		#     SET DEFAULTS FROM DICTIONARY  #
		#-----------------------------------#
		self.updateFromDict(Defaults.parameters())

		#-----------------------------------#
		#     OVERRIDE DEFAULTS             #
		#-----------------------------------#
		if params_dict:
			self.updateFromDict(params_dict)

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

		if self.fastSettings:
			self.useFastSettings()

		self.n_discountFactor = self.discount_factor_grid.size
		self.n_riskAver = self.risk_aver_grid.size
		self.discount_factor_grid += self.timeDiscount
		self.risk_aver_grid += self.riskAver

		#-----------------------------------#
		#     OTHER ADJUSTMENTS             #
		#-----------------------------------#
		self.nshocks = len(self.MPCshocks)

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

	def updateFromDict(self, paramsUpdate):
		for parameter, value in paramsUpdate.items():
			if hasattr(self, parameter):
				setattr(self, parameter, value)
			else:
				raise Exception(f'"{parameter}" is not a valid parameter')


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

	def useFastSettings(self):
		self.nx = 20
		self.nc = 20
		self.noPersIncome = True
		self.nSim = long(1e4)
		self.NsimMPC = long(1e4)

	def addIncomeParameters(self, income):
		self.nyP = income.nyP
		self.nyT = income.nyT

	def setParam(self, name, value, printReset=False):
		if name == 'adjustCost':
			self.series['Adjustment cost'] = value
		elif name == 'timeDiscount':
			self.discount_factor_grid = np.asarray(self.discount_factor_grid) \
				+ value - self.timeDiscount
			mean_discount = self.discount_factor_grid.mean()
			self.series['Discount factor (annualized)'] = mean_discount ** self.freq
			self.series['Discount factor (quarterly)'] = mean_discount ** (self.freq/4)
		elif name == 'riskAver':
			self.risk_aver_grid = np.asarray(self.risk_aver_grid) \
				+ value - self.riskAver

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