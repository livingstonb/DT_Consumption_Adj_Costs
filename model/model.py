import numpy as np

from model.cmodel import CModel
from misc import functions

class Model(CModel):
	def __init__(self, params, income, grids):
		super().__init__(params, income, grids)
		
		self.initialize()

	def initialize(self):
		self.nextMPCShock = 0

		print('\nConstructing interpolant array for EMAX')
		self.constructInterpolantForEMAX()

		# make initial guess for value function
		denom = 1 - self.p.timeDiscount * (1-self.p.deathProb)
		if np.abs(denom) < 1e-3:
			denom = 1e-3
		valueGuess = functions.utilityMat(self.p.risk_aver_grid,
			self.grids.x.matrix) / denom

		self.valueFunction = valueGuess

	def solve(self):
		print('\nBeginning value function iteration...')
		distance = 1e5
		self.iteration = 0

		while distance > self.p.tol:

			if self.iteration > self.p.maxIters:
				raise Exception(f'No convergence after {iteration+1} iterations...')

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

			if np.mod(self.iteration,50) == 0:
				print(f'    Iteration {self.iteration}, norm of |V1-V| = {distance}')

			if (self.iteration>2000) and (distance>1e5):
				raise Exception('No convergence')

			self.iteration += 1

		# compute c-policy function conditional on switching
		self.maximizeValueFromSwitching(findPolicy=True)

		print(f'Value function converged after {self.iteration+1} iterations.')

		self.doComputations()

	def updateValueFunction(self):
		"""
		This method updates self.valueFunction by finding max(valueSwitch-adjustCost,valueNoSwitch),
		where valueSwitch is used wherever c > x in the state space.
		"""
		self.valueFunction = np.where(self.grids.mustSwitch,
			np.asarray(self.valueSwitch),
			np.maximum(self.valueNoSwitch,np.asarray(self.valueSwitch))
			)

	def updateEMAX(self):
		"""
		This method computes E[V] from the most recent value function iteration.
		"""
		self.EMAX = self.interpMat.dot(np.reshape(self.valueFunction,(-1,1),order='F')).reshape(self.grids.matrixDim, order='F')

			# np.reshape(self.valueFunction,(-1,1),order='F')
				# ).reshape(self.grids.matrixDim,order='F')

	def updateValueNoSwitch(self):
		"""
		Updates self.valueNoSwitch via valueNoSwitch(c) = u(c) + beta * EMAX(c)
		"""
		self.valueNoSwitch = functions.utilityMat(self.p.risk_aver_grid,self.grids.c.matrix) \
			+ np.asarray(self.p.discount_factor_grid_wide) * (1 - self.p.deathProb) \
			* np.asarray(self.EMAX)

	def doComputations(self):
		super().doComputations()

class ModelWithNews(Model):
	def __init__(self, params, income, grids, valueNext, nextMPCShock):
		self.nextMPCShock = nextMPCShock
		self.valueFunction = valueNext

		super().__init__(params, income, grids)

	def initialize(self):
		print('\nConstructing interpolant array for EMAX')
		self.constructInterpolantForEMAX()
		super().updateEMAX()

	def updateEMAX(self):
		pass