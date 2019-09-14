import sys
import os
from matplotlib import pyplot as plt
import numpy as np

from model import modelObjects
from misc.load_specifications import load_specifications
from misc.boundsFinder import BoundsFinder
from model.model import Model
from model import simulator

#---------------------------------------------------------------#
#      LOCATION OF INCOME PROCESS                               #
#---------------------------------------------------------------#
basedir = os.getcwd()
locIncomeProcess = os.path.join(
	basedir,'input','IncomeGrids','quarterly_b.mat')

#---------------------------------------------------------------#
#      SET PARAMETERIZATION NUMBER                              #
#---------------------------------------------------------------#
if len(sys.argv) > 1:
	paramIndex = sys.argv[1]
else:
	paramIndex = 0

#---------------------------------------------------------------#
#      LOAD PARAMETERS                                          #
#---------------------------------------------------------------#
params = load_specifications(locIncomeProcess,index=paramIndex)

#---------------------------------------------------------------#
#      LOAD INCOME PROCESS                                      #
#---------------------------------------------------------------#
income = modelObjects.Income(params)
params.addIncomeParameters(income)

#---------------------------------------------------------------#
#      CREATE GRIDS                                             #
#---------------------------------------------------------------#
grids = modelObjects.GridCreator(params,income)

#---------------------------------------------------------------#
#      CREATE MODEL                                             #
#---------------------------------------------------------------#
model = Model(params,income,grids)

if params.iterateBeta:
	#-----------------------------------------------------------#
	#      FIND VALID LOWER BOUND FOR DISCOUNT RATE             #
	#-----------------------------------------------------------#
	neg_stepsize = -0.02
	pos_stepsize = 0.02
	lowerBoundInitial = 0.85
	lowerBoundFinder = BoundsFinder(
		neg_stepsize,pos_stepsize,lowerBoundInitial)
	lowerBoundFound = False
	
	ii = 0
	while (ii < lowerBoundFinder.maxIter) and not lowerBoundFound:
		try:
			# Attempt to solve model
			model.solve()
			eqSimulator = EquilibriumSimulator(params,income,grids,model)
			eqSimulator.simulate()

			if eqSimulator.results['mean assets'] < params.wealthTarget:
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
	neg_stepsize = -0.02
	pos_stepsize = 0.02
	upperBoundInitial = 0.98
	upperBoundFinder = BoundsFinder(
		neg_stepsize,pos_stepsize,upperBoundInitial)
	upperBoundFound = False
	
	ii = 0
	while (ii < upperBoundFinder.maxIter) and not upperBoundFound:
		try:
			# Attempt to solve model
			model.solve()
			eqSimulator = EquilibriumSimulator(params,income,grids,model)
			eqSimulator.simulate()

			if eqSimulator.results['mean assets'] > params.wealthTarget:
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
	# not yet coded
	raise Exception('iteration over discount rate not yet implemented')
else:
	#-----------------------------------------------------------#
	#      SOLVE MODEL ONCE                                     #
	#-----------------------------------------------------------#
	model.solve()

	eqSimulator = simulator.EquilibriumSimulator(params,income,grids,model)
	eqSimulator.simulate()

#-----------------------------------------------------------#
#      SIMULATE MPCs OUT OF AN IMMEDIATE SHOCK              #
#-----------------------------------------------------------#
shockIndices = [1,4] # only do 0.01 shock for now

finalSimStates = eqSimulator.returnFinalStates()
mpcSimulator = simulator.MPCSimulator(
	params,income,grids,model,shockIndices,finalSimStates)
mpcSimulator.simulate()

print('\nResults from simulation:\n')
print(eqSimulator.results.to_string())

print('\nMPCS:\n')
print(mpcSimulator.mpcs.to_string())