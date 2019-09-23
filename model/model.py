import numpy as np

from model.cmodel import CModel
from misc import functions

class Model(CModel):
	def __init__(self, params, income, grids):
		super().__init__(params, income, grids)
		
		self.initialize()

	def initialize(self):
		self.nextMPCShock = 0

		if not self.p.cubicEMAXInterp:
			print('Constructing interpolant array for EMAX')
			self.constructInterpolantForEMAX()

		# make initial guess for value function
		valueGuess = functions.utilityMat(self.p.riskAver,self.grids.c.matrix
			) / (1 - self.p.timeDiscount * (1 - self.p.deathProb))

		# subtract the adjustment cost for states with c > x
		self.valueFunction = valueGuess

	def solve(self):
		print('Beginning value function iteration...')
		distance = 1e5
		iteration = 0
		while distance > self.p.tol:

			if iteration > self.p.maxIters:
				raise Exception(f'No convergence after {iteration+1} iterations...')

			Vprevious = self.valueFunction

			# update EMAX = E[V|x,c,z,yP], where c is chosen c
			if self.p.cubicEMAXInterp:
				self.updateEMAXslow()
			else:
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

			if np.mod(iteration,50) == 0:
				print(f'    Iteration {iteration}, norm of |V1-V| = {distance}')

			iteration += 1

		# compute c-policy function conditional on switching
		self.maximizeValueFromSwitching(findPolicy=True)

		print('Value function converged')

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
		self.EMAX = self.interpMat.dot(np.reshape(self.valueFunction,(-1,1),order='F')
				).reshape(self.grids.matrixDim,order='F')

	def updateValueNoSwitch(self):
		"""
		Updates self.valueNoSwitch via valueNoSwitch(c) = u(c) + beta * EMAX(c)
		"""
		self.valueNoSwitch = functions.utilityMat(self.p.riskAver,self.grids.c.matrix) \
			+ self.p.timeDiscount * (1 - self.p.deathProb) \
			* np.asarray(self.EMAX)

	def doComputations(self):
		super().doComputations()

class ModelWithNews(Model):
	def __init__(self, params, income, grids, EMAX, valueGuess, nextMPCShock):
		super().__init__(self, params, income, grids)

		self.nextMPCShock = nextMPCShock
		self.EMAX = EMAX
		self.valueFunction = valueGuess

	def initialize(self):
		pass

	def updateEMAX(self):
		# EMAX comes from next period's model
		pass