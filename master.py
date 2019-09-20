import sys
import os
from matplotlib import pyplot as plt
import numpy as np

from scipy import optimize

from model import modelObjects
from misc.load_specifications import load_specifications
from misc.boundsFinder import BoundsFinder
from model.model import Model
from model import simulator

#---------------------------------------------------------------#
#      OPTIONS                                                  #
#---------------------------------------------------------------#
IterateBeta = True
Simulate = False # relevant if IterateBeta is False
MakePlots = False

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
	paramIndex = int(sys.argv[1])
else:
	paramIndex = 1

#---------------------------------------------------------------#
#      LOAD PARAMETERS                                          #
#---------------------------------------------------------------#
params = load_specifications(locIncomeProcess,name='fast')

#---------------------------------------------------------------#
#      LOAD INCOME PROCESS                                      #
#---------------------------------------------------------------#
income = modelObjects.Income(params)
params.addIncomeParameters(income)

#---------------------------------------------------------------#
#      CREATE GRIDS                                             #
#---------------------------------------------------------------#
grids = modelObjects.GridCreator(params,income)
# import pdb; pdb.set_trace()

#---------------------------------------------------------------#
#      CREATE MODEL                                             #
#---------------------------------------------------------------#
model = Model(params,income,grids)

if IterateBeta:
	Simulate = True

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
			print(f'--Trying lower bound = {lowerBoundFinder.currentBetaBound}')
			# Attempt to solve model
			params.resetDiscountRate(lowerBoundFinder.currentBetaBound)
			model.solve()
			eqSimulator = simulator.EquilibriumSimulator(params,income,grids,model)
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
	pos_stepsize = 0.002
	upperBoundInitial = 0.99
	upperBoundFinder = BoundsFinder(
		neg_stepsize,pos_stepsize,upperBoundInitial)
	upperBoundFound = False
	
	ii = 0
	while (ii < upperBoundFinder.maxIter) and not upperBoundFound:
		try:
			# Attempt to solve model
			print(f'--Trying upper bound = {upperBoundFinder.currentBetaBound}')
			params.resetDiscountRate(upperBoundFinder.currentBetaBound)
			model.solve()
			eqSimulator = simulator.EquilibriumSimulator(params,income,grids,model)
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
		print(f'-- Trying discount rate {x}')
		params.resetDiscountRate(x)
		model.solve()

		eqSimulator = simulator.EquilibriumSimulator(params, income, grids, model)
		eqSimulator.simulate()

		assets = eqSimulator.results['Mean wealth']
		return assets - params.wealthTarget

	betaOpt = optimize.brentq(iterateOverBeta, betaLowerBound, betaUpperBound,
		xtol=1e-6,rtol=1e-8)

else:
	#-----------------------------------------------------------#
	#      SOLVE MODEL ONCE                                     #
	#-----------------------------------------------------------#
	model.solve()

	if Simulate:
		eqSimulator = simulator.EquilibriumSimulator(params,income,grids,model)
		eqSimulator.simulate()

#-----------------------------------------------------------#
#      SIMULATE MPCs OUT OF AN IMMEDIATE SHOCK              #
#-----------------------------------------------------------#
if Simulate:
	shockIndices = [1,4] # only do 0.01 shock for now

	finalSimStates = eqSimulator.returnFinalStates()
	mpcSimulator = simulator.MPCSimulator(
		params,income,grids,model,shockIndices,finalSimStates)
	mpcSimulator.simulate()

#-----------------------------------------------------------#
#      SOLVE FOR POLICY GIVEN SHOCK NEXT PERIOD             #
#-----------------------------------------------------------#
# shockIndices = [4]

#  # i-th element is the model for a shock in i+1 periods
# futureShockModels = [None] * 4
# for ishock in shockIndices:
# 	# shock next period
# 	futureShockModels[0] = Model(
# 		params,income,grids,
# 		nextMPCShock=params.MPCshocks[ishock])
# 	futureShockModels[0].solve()

# 	for period in range(1,4):
# 		# shock in two or more periods
# 		futureShockModels[period] = Model(
# 			params,income,grids,
# 			EMAX=futureShockModels[period-1].EMAX)
# 		futureShockModels[period].solve()

#-----------------------------------------------------------#
#      PRINT RESULTS                                        #
#-----------------------------------------------------------#

if Simulate:
	print('\nResults from simulation:\n')
	print(eqSimulator.results.to_string())

	print('\nMPCS:\n')
	print(mpcSimulator.mpcs.to_string())

#-----------------------------------------------------------#
#      PLOT POLICY FUNCTION                                 #
#-----------------------------------------------------------#

if MakePlots:
	cSwitch = np.asarray(model.valueFunction) == (np.asarray(model.valueSwitch) - params.adjustCost)
	cPolicy = cSwitch * np.asarray(model.cSwitchingPolicy) + (~cSwitch) * np.asarray(grids.c.matrix)

	ixvals = [0,10,20,40,50,70]
	xvals = np.array([grids.x.flat[i] for i in ixvals])
	print(xvals)

	fig, ax = plt.subplots(nrows=2,ncols=3)
	fig.suptitle('Consumption function vs. state c')
	i = 0
	for row in range(2):
		for col in range(3):
			ax[row,col].plot(grids.c.flat,cPolicy[ixvals[i],:,0,5])
			ax[row,col].set_title(f'x = {xvals[i]}')
			ax[row,col].set_xlabel('c, state')
			ax[row,col].set_ylabel('actual consumption')
			i += 1

	fig, ax = plt.subplots(nrows=2,ncols=3)
	fig.suptitle('Consumption function vs. state c, zoomed')
	i = 0
	for row in range(2):
		for col in range(3):
			ax[row,col].plot(grids.c.flat,cPolicy[ixvals[i],:,0,5])
			ax[row,col].set_title(f'x = {xvals[i]}')
			ax[row,col].set_xlabel('c, state')
			ax[row,col].set_ylabel('actual consumption')
			ax[row,col].set_xbound(0, 0.5)
			i += 1

	fig, ax = plt.subplots(nrows=2,ncols=3)
	fig.suptitle('Value function vs. state c')
	i = 0
	for row in range(2):
		for col in range(3):
			ax[row,col].plot(grids.c.flat,model.valueFunction[ixvals[i],:,0,5])
			ax[row,col].set_title(f'x = {ixvals[i]}')
			ax[row,col].set_xlabel('c, state')
			ax[row,col].set_ylabel('value function')
			i += 1

	icvals = [10,25,50,100,500,750]
	cvals = np.array([grids.c.flat[i] for i in icvals])
	print(cvals)

	fig, ax = plt.subplots(nrows=2,ncols=3)
	fig.suptitle('Consumption function vs. assets')
	i = 0
	for row in range(2):
		for col in range(3):
			ax[row,col].plot(grids.x.flat,cPolicy[:,icvals[i],0,5])
			ax[row,col].set_title(f'c = {cvals[i]}')
			ax[row,col].set_xlabel('x, cash-on-hand')
			ax[row,col].set_ylabel('actual consumption')
			i += 1

	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	ax.plot(grids.x.flat,model.inactionRegionLower[:,0,5])
	ax.plot(grids.x.flat,model.cSwitchingPolicy[:,0,0,5])
	ax.plot(grids.x.flat,model.inactionRegionUpper[:,0,5])

	ax.set_title('Inaction region for consumption')
	ax.set_xlabel('cash-on-hand, x')
	ax.set_ylabel('consumption')
	ax.legend(['Lower bound of inaction region', 'Desired c without adj cost', 
		'Upper bound of inaction region'])

	plt.show()
