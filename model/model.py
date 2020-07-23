import numpy as np

from model import Params, Income, Grid
from model.cmodel import CModel
from misc import functions

class Model(CModel):
	"""
	Inherits attributes and methods from the base extension class
	CModel. This is not to be used when computing MPCs out of news.
	"""
	def initialize(self):
		self.nextMPCShock = 0 # no shock next period

		self.borrLimCurr = self.p.borrowLim
		self.borrLimNext = self.p.borrowLim

		print('\nConstructing interpolant array for EMAX')
		self.constructInterpolantForEMAX()

	def makeValueGuess(self):
		denom = 1 - self.p.timeDiscount * (1-self.p.deathProb)
		denom = np.maximum(denom, 1e-3)
		valueGuess = functions.utilityMat(self.p.risk_aver_grid,
			self.grids.x_matrix) / denom

		self.valueFunction = valueGuess

	def solve(self, oneIter=False):
		if not oneIter:
			print('\nBeginning value function iteration...')
			self.makeValueGuess()

		distance = 1e5
		self.iteration = 0

		while distance > self.p.tol:

			if self.iteration > self.p.maxIters:
				raise Exception(f'No convergence after {self.iteration+1} iterations...')

			Vprevious = self.valueFunction

			# update EMAX = E[V|x,c,z,yP], where c is chosen c
			self.updateEMAX()

			# update value function of not switching
			self.updateValueNoSwitch()

			# update value function of switching
			self.maximizeValueFromSwitching()

			# compute V = max(VSwitch,VNoSwitch)
			self.updateValueFunction()

			distance = np.abs(
				np.asarray(self.valueFunction) - np.asarray(Vprevious)
				).flatten().max()

			if (np.mod(self.iteration,50) == 0) and not oneIter:
				print(f'    Iteration {self.iteration+1}, norm of |V1-V| = {distance}')

			if (self.iteration>2000) and (distance>1e5):
				raise Exception('No convergence')

			self.iteration += 1

			if oneIter:
				break

		self.doComputations()

		# find inaction region
		self.maximizeValueFromSwitching(final=True)

		if not oneIter:
			print(f'Value function converged after {self.iteration} iterations.')

	def updateValueFunction(self):
		"""
		This method updates valueFunction by finding max(valueSwitch-adjustCost,valueNoSwitch),
		where valueSwitch is used wherever c > x in the state space.
		"""
		correctedValNoSwitch = functions.replaceNumpyNan(self.valueNoSwitch, -1e9)
		self.valueFunction = np.where(self.mustSwitch,
			np.asarray(self.valueSwitch),
			np.maximum(self.valueSwitch, correctedValNoSwitch)
			)

	def updateEMAX(self):
		"""
		This method computes EMAX, which is interpMat * valueFunction when
		there is NOT news of a future shock. In the case of news, next
		period's value function should be used.
		"""
		self.EMAX = self.interpMat.dot(np.reshape(self.valueFunction,(-1,1),order='F')
			).reshape(self.grids.matrixDim, order='F')

	def doComputations(self):
		correctedValNoSwitch = functions.replaceNumpyNan(self.valueNoSwitch, -1e9)
		self.willSwitch = np.asarray(self.valueSwitch) >= correctedValNoSwitch
		self.cChosen = self.willSwitch * self.cSwitchingPolicy + \
			(~self.willSwitch) * self.grids.c_wide

		self.cSwitchingPolicy = self.cSwitchingPolicy.reshape((self.p.nx,1,self.p.nz,self.p.nyP,1),
			order='F')

class ModelWithNews(Model):
	"""
	This class solves the model when a future shock is expected. The value function
	used to compute EMAX should come from the solution to a model in which an
	there is an immediate shock or if a future shock is expected.
	"""
	def __init__(self, params, income, grids, valueNext,
		shock, periodsUntilShock):
		super().__init__(params, income, grids)

		self.nextMPCShock = (periodsUntilShock == 1) * shock
		self.valueFunction = valueNext

		ymin = income.ymin + params.govTransfer
		borrowLims = functions.computeAdjBorrLims(shock,
			ymin, params.borrowLim, params.R, periodsUntilShock)

		self.borrLimCurr = borrowLims.pop()
		self.borrLimNext = borrowLims.pop()
		self.constructInterpolantForEMAX()

	def solve(self):
		super().solve(True)