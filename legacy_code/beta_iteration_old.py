if IterateBeta:
	Simulate = True
	SimulateMPCs = True

	#-----------------------------------------------------------#
	#      FIND VALID LOWER BOUND FOR DISCOUNT RATE             #
	#-----------------------------------------------------------#
	print('\nLooking for valid lower bound on discount rate\n')

	neg_stepsize = -0.02
	pos_stepsize = 0.02
	lowerBoundInitial = 0.85
	lowerBoundFinder = BoundsFinder(
		neg_stepsize,pos_stepsize,lowerBoundInitial)
	lowerBoundFound = False
	
	ii = 0
	while (ii < lowerBoundFinder.maxIter) and not lowerBoundFound:
		try:
			print(f'--Trying lower bound = {lowerBoundFinder.currentBetaBound:.6f}')
			# Attempt to solve model
			params.resetDiscountRate(lowerBoundFinder.currentBetaBound)
			model.solve()
			eqSimulator = simulator.EquilibriumSimulator(params, income, grids, model.cSwitchingPolicy,
				model.valueDiff)
			eqSimulator.simulate()

			if eqSimulator.results['Mean wealth'] < params.wealthTarget:
				lowerBoundFound = True
			else:
				# mean wealth too high, reduce discount factor
				print('Increasing candidate lower bound on discount factor.')
				lowerBoundFinder.decrease()

		except Exception as e:
				print(e)
				# discount factor probably too low
				lowerBoundFinder.increase()

		ii += 1

	if lowerBoundFound:
		betaLowerBound = lowerBoundFinder.currentBetaBound
	else:
		raise Exception('Lower bound for discount factor could not be found.')

	#-----------------------------------------------------------#
	#      FIND VALID UPPER BOUND FOR DISCOUNT RATE             #
	#-----------------------------------------------------------#
	print('\nLooking for valid upper bound on discount rate\n')
	neg_stepsize = -0.02
	pos_stepsize = 0.004
	upperBoundInitial = 0.995
	upperBoundFinder = BoundsFinder(
		neg_stepsize,pos_stepsize,upperBoundInitial)
	upperBoundFound = False
	
	ii = 0
	while (ii < upperBoundFinder.maxIter) and not upperBoundFound:
		try:
			# Attempt to solve model
			print(f'--Trying upper bound = {upperBoundFinder.currentBetaBound:.6f}')
			params.resetDiscountRate(upperBoundFinder.currentBetaBound)
			model.solve()
			eqSimulator = simulator.EquilibriumSimulator(params, income, grids, model.cSwitchingPolicy,
				model.valueDiff)
			eqSimulator.simulate()

			if eqSimulator.results['Mean wealth'] > params.wealthTarget:
				upperBoundFound = True
			else:
				# mean wealth too high, reduce discount factor
				print('Increasing candidate upper bound on discount factor.')
				upperBoundFinder.increase()
		except Exception as e:
				print(e)
				# discount factor probably too high
				upperBoundFinder.decrease()

		ii += 1

	if upperBoundFound:
		betaUpperBound = upperBoundFinder.currentBetaBound
	else:
		raise Exception('Upper bound for discount factor could not be found.')

	#-----------------------------------------------------------#
	#      ITERATE OVER DISCOUNT RATE                           #
	#-----------------------------------------------------------#
	print('\nBeginning iteration over the discount factor\n')
	def iterateOverBeta(x):
		print(f'-- Trying discount rate {x:.6f}')
		params.resetDiscountRate(x)
		model.solve()

		eqSimulator = simulator.EquilibriumSimulator(params, income, grids, 
			model.cSwitchingPolicy, model.valueDiff)
		eqSimulator.simulate()

		assets = eqSimulator.results['Mean wealth']
		return assets - params.wealthTarget

	betaOpt = optimize.root_scalar(iterateOverBeta, bracket=(betaLowerBound,betaUpperBound),
		method='brentq', xtol=1e-7, rtol=1e-9, maxiter=params.wealthIters).root

	params.resetDiscountRate(betaOpt)