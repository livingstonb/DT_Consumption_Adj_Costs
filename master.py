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
	paramIndex = int(sys.argv[1])
else:
	paramIndex = 1

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
# import pdb; pdb.set_trace()

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
#      PLOT POLICY FUNCTION                                 #
#-----------------------------------------------------------#

cSwitch = np.asarray(model.valueFunction) == (np.asarray(model.valueSwitch) - params.adjustCost)
cPolicy = cSwitch * np.asarray(model.cSwitchingPolicy) + (~cSwitch) * np.asarray(grids.c.matrix)

ixvals = [10,20,30,40,50,60]
xvals = np.array([grids.x.flat[i] for i in ixvals])
print(xvals)

fig, ax = plt.subplots(nrows=2,ncols=3)
fig.suptitle('Consumption function vs. state c')
i = 0
for row in range(2):
	for col in range(3):
		ax[row,col].plot(grids.c.flat,cPolicy[ixvals[i],:,0,5])
		ax[row,col].set_title(f'x = {xvals[i]}')
		i += 1

icvals = [10,20,30,40,50,100]
cvals = np.array([grids.c.flat[i] for i in icvals])
print(cvals)

fig, ax = plt.subplots(nrows=2,ncols=3)
fig.suptitle('Consumption function vs. assets')
i = 0
for row in range(2):
	for col in range(3):
		ax[row,col].plot(grids.x.flat,cPolicy[:,icvals[i],0,5])
		ax[row,col].set_title(f'c = {cvals[i]}')
		i += 1

fig, ax = plt.subplots(nrows=2,ncols=3)
fig.suptitle('Value function vs. assets')
i = 0
for row in range(2):
	for col in range(3):
		ax[row,col].plot(grids.x.flat,model.valueFunction[:,icvals[i],0,5])
		ax[row,col].set_title(f'c = {cvals[i]}')
		i += 1

plt.show()