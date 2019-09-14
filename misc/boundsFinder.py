class BoundsFinder:
	# This class implements an algorithm to increase or reduce
	# beta in order to find a valid upper or lower bound for beta
	# prior to iteration via fzero.
	#
	# The algorithm first uses the step sizes provided in the
	# arguments, and then switches to a midpoint formula when
	# it is needed to move in the opposite direction.
	#
	# The 'bound' variable takes the value 'upper' or 'lower'
	#
	# fnHandle is the function whose first output is mean wealth
	# less the mean wealth target. It must be a function of beta
	# only.

	def __init__(self, neg_stepsize, pos_stepsize, beta0):
		self.neg_stepsize = neg_stepsize
		self.pos_stepsize = pos_stepsize
		self.betaBoundHistory = []
		self.beta0 = beta0
		self.currentBetaBound = None
		self.lastDirection = None
		self.lag = None

		self.maxIter = 50

	def increase(self):
		print(f'Raising candidate for {self.bound} bound on time discount factor.')
		if self.lastDirection in [None,'up']:
			# increase by step
			self.currentBetaBound += self.pos_stepsize
			self.lag = 1
		elif self.lastDirection == 'down':
			# updating by midpoint formula
			self.currentBetaBound = (self.betaBoundHistory[-lag-1]
										+ self.currentBetaBound) / 2
			self.lag += 1

		self.betaBoundHistory.append(self.currentBetaBound)

	def decrease(self):
		if self.lastDirection in [None,'down']:
			# decrease by step
			self.currentBetaBound += self.neg_stepsize
			self.lag = 1
		elif self.lastDirection == 'up':
			# update by midpoint formula
			self.currentBetaBound = (self.betaBoundHistory[-lag-1]
										+ self.currentBetaBound) / 2
			self.lag += 1

		self.betaBoundHistory.append(self.currentBetaBound)