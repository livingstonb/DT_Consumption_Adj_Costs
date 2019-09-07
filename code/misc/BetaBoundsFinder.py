class BetaBoundsFinder:
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

	def __init__(self, neg_stepsize, pos_stepsize, beta0, fnHandle):
		self.neg_stepsize = neg_stepsize
		self.pos_stepsize = pos_stepsize
		self.betaBoundHistory = []
		self.beta0 = beta0
		self.currentBetaBound = None
		self.lastDirection = None
		self.lag = None
		self.bound = bound

		self.fnHandle = fnHandle

		self.maxIter = 50

	def findBounds(self):
		lowerBound0 = self.beta0
		lowerBound = self.findLowerBound(lowerBound0)

		upperBound0 = lowerBound + 0.01
		upperBound = self.findUpperBound(upperBound0)

		return [lowerBound,upperBound]

	def findLowerBound(self, betaInitial):
		self.currentBetaBound = betaInitial

		ii = 1
		betaLowerBoundFound = False
		while (ii <= self.maxIter) and not betaLowerBoundFound:
			ii += 1
			try:
				# Attempt to solve model
		        AYdiff = self.fnHandle(self.currentBetaBound)

		        # If wealth/income - target < 0, beta lower 
		        # bound has been found
		        if AYdiff < 0
		            betaLowerBoundFound = true
		        else
		        	# mean wealth too low, increase discount factor
		        	print('Increasing candidate lower bound on discount factor.')
		            self.increase()
		    except as e:
		    	print(e)
		    	# discount factor probably too high
		    	print('Reducing candidate lower bound on discount factor.')
		    	self.decrease()

		if betaLowerBoundFound:
			return self.currentBetaBound
		else:
			raise Exception('Failed to find lower bound for discount factor.')

	def findUpperBound(self, betaInitial):
		self.currentBetaBound = betaInitial

		ii = 1
		betaUpperBoundFound = False
		while (ii <= self.maxIter) and not betaUpperBoundFound:
			ii += 1
			try:
				# Attempt to solve model
		        AYdiff = self.fnHandle(self.currentBetaBound)

		        # If wealth/income - target > 0, beta upper 
		        # bound has been found
		        if AYdiff > 0
		            betaUpperBoundFound = true
		        else
		        	# mean wealth too low, reduce discount factor
		        	print('Decreasing candidate upper bound on discount factor.')
		            self.decrease()
		    except as e:
		    	print(e)
		    	# discount factor probably too low
		    	print('Increasing candidate upper bound on discount factor.')
		    	self.increase()

		if betaUpperBoundFound:
			return self.currentBetaBound
		else:
			raise Exception('Failed to find upper bound for discount factor.')


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

	def reset(self):
		self.betaBoundHistory = []
		self.currentBetaBound = self.beta0
		self.lastDirection = None
		self.lag = None
