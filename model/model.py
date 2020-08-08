import numpy as np

from model import Params, Income, Grid
from model.cmodel import CModel
from misc import functions

class Model(CModel):
	"""
	Inherits attributes and methods from the base extension class
	CModel. This is not to be used when computing MPCs out of news.
	"""
	def __init__(self, params, income, grids):
		CModel.__init__(self, params, income, grids)

		self.nextMPCShock = 0 # no shock next period
		self.borrLimCurr = self.p.borrowLim
		self.borrLimNext = self.p.borrowLim

		print('\nConstructing interpolant array for EMAX')
		self.constructInterpolantForEMAX()

	def makeValueGuess(self):
		denom = 1 - self.p.timeDiscount * (1-self.p.deathProb)
		self.valueFunction = functions.utilityMat(self.p.risk_aver_grid,
			self.grids.x_matrix) / np.maximum(denom, 1e-3)

	def solve(self):
		print('\nBeginning value function iteration...')
		self.makeValueGuess()

		distance = 1e5
		self.iteration = 0

		while distance > self.p.tol:

			if self.iteration > self.p.maxIters:
				raise Exception(f'No convergence after {self.iteration+1} iterations...')

			Vprevious = self.valueFunction

			self.iterateOnce()

			distance = checkProgress(self.valueFunction, Vprevious, self.iteration)
			self.iteration += 1

		self.doComputations()
		self.findInactionRegion()

		print(f'Value function converged after {self.iteration} iterations.')

	def iterateOnce(self):
		# update EMAX = E[V|x,c,z,yP], where c is chosen c
		self.updateEMAX()

		# update value function of not switching
		self.updateValueNoSwitch()

		# update value function of switching
		self.maximizeValueFromSwitching()

		# compute V = max(VSwitch,VNoSwitch)
		self.updateValueFunction()

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

	def updateValueNoSwitch(self):
		"""
		Updates valueNoSwitch via valueNoSwitch(c) = u(c) + beta * EMAX(c)
		"""
		discountFactor_broadcast = np.reshape(self.p.discount_factor_grid,
			(1, 1, self.p.n_discountFactor, 1))
		riskAver_broadcast = np.reshape(self.p.risk_aver_grid,
			(1, 1, self.p.n_riskAver, 1))
		valueNoSwitch = functions.utilityMat(riskAver_broadcast, self.grids.c_wide) \
			+ discountFactor_broadcast * (1 - self.p.deathProb) * np.asarray(self.EMAX)

		valueNoSwitch[self.mustSwitch[:,:,0,0]] = np.nan
		self.valueNoSwitch = valueNoSwitch

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
	there is an immediate shock or in whichs a future shock is expected.
	"""
	def __init__(self, params, income, grids, valueNext,
		shock, periodsUntilShock):
		CModel.__init__(self, params, income, grids)

		self.nextMPCShock = (periodsUntilShock == 1) * shock
		self.valueFunction = valueNext

		ymin = income.ymin + params.govTransfer
		borrowLims = functions.computeAdjBorrLims(shock,
			ymin, params.borrowLim, params.R, periodsUntilShock)

		self.borrLimCurr = borrowLims.pop()
		self.borrLimNext = borrowLims.pop()
		self.constructInterpolantForEMAX()

	def solve(self):
		self.iterateOnce()
		self.doComputations()
		self.findInactionRegion()

def checkProgress(vCurr, vPrev, iteration):
	distance = np.abs(
		np.asarray(vCurr) - np.asarray(vPrev)
		).flatten().max()

	if np.mod(iteration, 100) == 0:
		print(f'    Iteration {iteration+1}, norm of |V1-V| = {distance}')

	if (iteration>2000) and (distance>1e5):
		raise Exception('No convergence')

	return distance