import sys
import os
from matplotlib import pyplot as plt
import numpy as np
from itertools import combinations

from scipy import optimize

import pandas as pd

from model import Params, Income, Grid
from misc.load_specifications import load_specifications
from misc.boundsFinder import BoundsFinder
from misc import mpcsTable
from model.model import Model, ModelWithNews
from model import simulator

#---------------------------------------------------------------#
#      SET PARAMETERIZATION NUMBER                              #
#---------------------------------------------------------------#
indexSet = False
for arg in sys.argv:
	try:
		paramIndex = int(arg)
		indexSet = True
		break
	except:
		pass

if not indexSet:
	paramIndex = 1

#---------------------------------------------------------------#
#      OR SET PARAMETERIZATION NAME                             #
#---------------------------------------------------------------#
# THIS OVERRIDES paramIndex: TO IGNORE SET TO EMPTY STRING
name = ''

#---------------------------------------------------------------#
#      OPTIONS                                                  #
#---------------------------------------------------------------#
IterateBeta = False
Simulate = True # relevant if IterateBeta is False
SimulateMPCs = True

basedir = os.getcwd()
outdir = os.path.join(basedir,'output')
if not os.path.exists(outdir):
	os.mkdir(outdir)

#---------------------------------------------------------------#
#      LOCATION OF INCOME PROCESS                               #
#---------------------------------------------------------------#

locIncomeProcess = os.path.join(
	basedir,'input','IncomeGrids','quarterly_b.mat')

#---------------------------------------------------------------#
#      LOAD PARAMETERS                                          #
#---------------------------------------------------------------#
if name:
	params = load_specifications(locIncomeProcess, name=name)
else:
	params = load_specifications(locIncomeProcess, index=paramIndex)

#---------------------------------------------------------------#
#      LOAD INCOME PROCESS                                      #
#---------------------------------------------------------------#
income = Income.Income(params)
params.addIncomeParameters(income)

#---------------------------------------------------------------#
#      CREATE GRIDS                                             #
#---------------------------------------------------------------#
grids = Grid.GridCreator(params, income)

#---------------------------------------------------------------#
#      CREATE MODEL                                             #
#---------------------------------------------------------------#
model = Model(params, income, grids)
model.initialize()

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

# #-----------------------------------------------------------#
# #      CALIBRATING TO A VARIABLE OTHER THAN MEAN WEALTH     #
# #-----------------------------------------------------------#
# def calibrator(variables):
# 	params.resetDiscountRate(np.abs(variables[0])/(1+np.abs(variables[0]))-0.02)
# 	params.resetAdjustCost(np.abs(variables[1])+0.0001)

# 	print(params.timeDiscount)
# 	print(params.adjustCost)

# 	model.solve()

# 	eqSimulator = simulator.EquilibriumSimulator(params, income, grids,
# 		model.cSwitchingPolicy, model.valueDiff)
# 	eqSimulator.simulate()

# 	shockIndices = [3]
# 	mpcSimulator = simulator.MPCSimulator(
# 		params, income, grids,
# 		model.cSwitchingPolicy,
# 		model.valueDiff,
# 		shockIndices,
# 		eqSimulator.finalStates)
# 	mpcSimulator.simulate()

# 	rowname = f'P(Q1 MPC > 0) for shock of {params.MPCshocks[3]}'

# 	targets = np.array([
# 		eqSimulator.results['Wealth <= $1000'] - 0.23,
# 		mpcSimulator.results[rowname] - 0.2
# 		])

# 	print(f"\n\n --- P(a < $1000) = {eqSimulator.results['Wealth <= $1000']} ---\n")
# 	print(f" --- P(MPC > 0) = {mpcSimulator.results[rowname]} ---\n\n")

# 	return targets

# # for P(a<$1000) = 0.23, P(MPC>0) = 0.18, use 25.9, 0.002479
# x0 = np.array([69, 0.005559])
# opt_results = optimize.root(calibrator, x0, method='krylov').x

#-----------------------------------------------------------#
#      SOLVE MODEL ONCE                                     #
#-----------------------------------------------------------#
model.solve()

eqSimulator = simulator.EquilibriumSimulator(params, income, grids, 
	model.cSwitchingPolicy, model.valueDiff)
if Simulate:
	eqSimulator.simulate(final=True)
	finalSimStates = eqSimulator.finalStates
else:
	finalSimStates = []

#-----------------------------------------------------------#
#      SIMULATE MPCs OUT OF AN IMMEDIATE SHOCK              #
#-----------------------------------------------------------#
shockIndices = [0,1,2,3,4,5]

mpcSimulator = simulator.MPCSimulator(
	params, income, grids, 
	model.cSwitchingPolicy,
	model.valueDiff, 
	shockIndices, finalSimStates)

if Simulate and SimulateMPCs:
	mpcSimulator.simulate()
#-----------------------------------------------------------#
#      SOLVE FOR POLICY GIVEN SHOCK NEXT PERIOD             #
#-----------------------------------------------------------#
shockIndices_shockNextPeriod = [2,3,4,5]
currentShockIndices = [6] * len(shockIndices_shockNextPeriod) # 6 is shock of 0

cSwitch_shockNextPeriod = np.zeros((params.nx,1,params.nz,params.nyP,len(shockIndices_shockNextPeriod)+1))
valueDiffs_shockNextPeriod = np.zeros((params.nx,params.nc,params.nz,params.nyP,len(shockIndices_shockNextPeriod)+1))

cSwitch_shockNextPeriod[:,:,:,:,-1] = model.cSwitchingPolicy[:,:,:,:,0]
valueDiffs_shockNextPeriod[:,:,:,:,-1] = model.valueDiff[:,:,:,:,0]
valueBaseline = model.valueFunction
emaxBaseline = model.EMAX
model.interpMat = []

i = 0
for ishock in shockIndices_shockNextPeriod:
	model_shockNextPeriod = ModelWithNews(
		params, income, grids,
		valueBaseline,
		emaxBaseline,
		params.MPCshocks[ishock])

	if SimulateMPCs:
		print(f'Solving for shock of {params.MPCshocks[ishock]} next period')
		model_shockNextPeriod.solve()
		cSwitch_shockNextPeriod[:,:,:,:,i] = model_shockNextPeriod.cSwitchingPolicy[:,:,:,:,0]
		valueDiffs_shockNextPeriod[:,:,:,:,i] = model_shockNextPeriod.valueDiff[:,:,:,:,0]

	del model_shockNextPeriod
	i += 1
		# for period in range(1,4):
		# 	# shock in two or more periods
		# 	futureShockModels[period] = Model(
		# 		params,income,grids,
		# 		EMAX=futureShockModels[period-1].EMAX)
		# 	futureShockModels[period].solve()

#-----------------------------------------------------------#
#      SOLVE FOR 1-YEAR LOAN                                #
#-----------------------------------------------------------#
cSwitch_loan = np.zeros((params.nx,1,params.nz,params.nyP,2))
valueDiffs_loan = np.zeros((params.nx,params.nc,params.nz,params.nyP,2))

ishock = 0
# start with last quarter
model_loan = ModelWithNews(
	params, income, grids,
	valueBaseline,
	emaxBaseline,
	params.MPCshocks[ishock])

if SimulateMPCs:
	print(f'Solving for one year loan')
	model_loan.solve()

	for period in range(3):
		shock = 0.0
		model_loan = ModelWithNews(
			params, income, grids,
			model_loan.valueFunction,
			model_loan.EMAX,
			shock)
		model_loan.solve()

	cSwitch_loan[:,:,:,:,0] = model_loan.cSwitchingPolicy[:,:,:,:,0]
	cSwitch_loan[:,:,:,:,1] = model.cSwitchingPolicy[:,:,:,:,0]
	valueDiffs_loan[:,:,:,:,0] = model_loan.valueDiff[:,:,:,:,0]
	valueDiffs_loan[:,:,:,:,1] = model.valueDiff[:,:,:,:,0]
	del model_loan

#-----------------------------------------------------------#
#      SHOCK OF -$500 IN 2 YEARS                            #
#-----------------------------------------------------------#
cSwitch_shock2Years = np.zeros((params.nx,1,params.nz,params.nyP,2))
valueDiffs_shock2Years = np.zeros((params.nx,params.nc,params.nz,params.nyP,2))

ishock = 2
model_shock2Years = ModelWithNews(
	params, income, grids,
	valueBaseline,
	emaxBaseline,
	params.MPCshocks[ishock])

if SimulateMPCs:
	model_shock2Years.solve()

	for period in range(7):
		shock = 0.0
		model_shock2Years = ModelWithNews(
			params, income, grids,
			model_shock2Years.valueFunction,
			model_shock2Years.EMAX,
			shock)
		model_shock2Years.solve()

	cSwitch_shock2Years[:,:,:,:,0] = model_shock2Years.cSwitchingPolicy[:,:,:,:,0]
	cSwitch_shock2Years[:,:,:,:,1] = model.cSwitchingPolicy[:,:,:,:,0]
	valueDiffs_shock2Years[:,:,:,:,0] = model_shock2Years.valueDiff[:,:,:,:,0]
	valueDiffs_shock2Years[:,:,:,:,1] = model.valueDiff[:,:,:,:,0]
	del model_shock2Years

#-----------------------------------------------------------#
#      SIMULATE MPCs OUT OF NEWS                            #
#-----------------------------------------------------------#
currentShockIndices = [6] * len(shockIndices_shockNextPeriod)
mpcNewsSimulator_shockNextPeriod = simulator.MPCSimulatorNews(
	params, income, grids, 
	cSwitch_shockNextPeriod, valueDiffs_shockNextPeriod,
	shockIndices_shockNextPeriod, currentShockIndices,
	finalSimStates, periodsUntilShock=1)

shockIndices_shock2Years = [2]
currentShockIndices = [6]
mpcNewsSimulator_shock2Years = simulator.MPCSimulatorNews(
	params, income, grids,
	cSwitch_shock2Years, valueDiffs_shock2Years,
	shockIndices_shock2Years, currentShockIndices,
	finalSimStates, periodsUntilShock=8)

shockIndices_loan = [0]
currentShockIndices = [5]
mpcNewsSimulator_loan = simulator.MPCSimulatorNews_Loan(
	params, income, grids,
	cSwitch_loan, valueDiffs_loan,
	shockIndices_loan, currentShockIndices,
	finalSimStates, periodsUntilShock=4)

if SimulateMPCs:
	mpcNewsSimulator_shockNextPeriod.simulate()
	mpcNewsSimulator_shock2Years.simulate()
	mpcNewsSimulator_loan.simulate()

#-----------------------------------------------------------#
#      RESULTS                                              #
#-----------------------------------------------------------#
# find fractions of households that respond to one, both, or neither of
# two treatments
if Simulate:
	mpcs_over_states = dict()
	mpcs_over_states['$500 GAIN'] = mpcSimulator.mpcs[3]
	mpcs_over_states['$2500 GAIN'] = mpcSimulator.mpcs[4]
	mpcs_over_states['$5000 GAIN'] = mpcSimulator.mpcs[5]
	mpcs_over_states['$500 LOSS'] = mpcSimulator.mpcs[2]
	mpcs_over_states['$500 NEWS-GAIN'] = mpcNewsSimulator_shockNextPeriod.mpcs[3]
	mpcs_over_states['$5000 NEWS-GAIN'] = mpcNewsSimulator_shockNextPeriod.mpcs[5]
	mpcs_over_states['$500 NEWS-LOSS'] = mpcNewsSimulator_shockNextPeriod.mpcs[2]
	mpcs_over_states['$500 NEWS-LOSS IN 2 YEARS'] = mpcNewsSimulator_shock2Years.mpcs[2]
	mpcs_over_states['$5000 LOAN'] = mpcNewsSimulator_loan.mpcs[0]

	index = []
	treatmentResponses = pd.DataFrame()
	for pair in combinations(mpcs_over_states.keys(), 2):
		key = pair[0] + ', ' + pair[1]
		index.append(key)
		thisTreatmentPair = {
			'Response to 1 only' : ((mpcs_over_states[pair[0]] > 0)  & (mpcs_over_states[pair[1]] == 0) ).mean(),
			'Response to 2 only' : ((mpcs_over_states[pair[0]] == 0)  & (mpcs_over_states[pair[1]] > 0) ).mean(),
			'Response to both' : ((mpcs_over_states[pair[0]] > 0)  & (mpcs_over_states[pair[1]] > 0) ).mean(),
			'Response to neither' : ((mpcs_over_states[pair[0]] == 0)  & (mpcs_over_states[pair[1]] == 0) ).mean(),
		}
		treatmentResponses = treatmentResponses.append(thisTreatmentPair, ignore_index=True)

	
	treatmentResponses.index = index
	savepath = os.path.join(outdir, f'run{paramIndex}_treatment_responses.csv')
	# treatmentResponses.to_excel(savepath, freeze_panes=(0,0), index_label=params.name)
	treatmentResponses.to_csv(savepath, index_label=params.name)

	# find fractions responding in certain wealth groups
	group1 = np.asarray(finalSimStates['asim']) <= 0.081
	group2 = (np.asarray(finalSimStates['asim']) > 0.081) & (np.asarray(finalSimStates['asim'])<= 0.486)
	group3 = (np.asarray(finalSimStates['asim']) > 0.486) & (np.asarray(finalSimStates['asim']) <= 4.05)
	group4 = (np.asarray(finalSimStates['asim']) > 4.05)

	groups = [group1, group2, group3, group4]

	groupLabels = [
		'$0-$5000',
		'$5000-$30,000',
		'$30,000-$250,000',
		'$250,000+',
	]

	treatments = [
		('$500 GAIN', '$500 NEWS-GAIN'),
		('$5000 GAIN', '$5000 NEWS-GAIN'),
		('$500 GAIN', '$500 LOSS'),
		('$500 LOSS', '$500 NEWS-LOSS'),	
	]

	# loop over income groups
	for i in range(4):	
		index = []
		treatmentResults = []

		for pair in treatments:
			thisTreatmentPair = dict()
			
			index.append(pair[0] + ', ' + pair[1])

			mpcs_treatment1 = mpcs_over_states[pair[0]][groups[i].flatten()]
			mpcs_treatment2 = mpcs_over_states[pair[1]][groups[i].flatten()]
			thisTreatmentPair['Response to 1 only'] =  ((mpcs_treatment1 > 0) & (mpcs_treatment2 == 0)).mean()
			thisTreatmentPair['Response to 2 only'] =  ((mpcs_treatment1 == 0) & (mpcs_treatment2 > 0)).mean()
			thisTreatmentPair['Response to both'] = ((mpcs_treatment1 > 0) & (mpcs_treatment2 > 0)).mean()
			thisTreatmentPair['Response to neither'] = ((mpcs_treatment1 == 0) & (mpcs_treatment2 == 0)).mean()

			treatmentResults.append(thisTreatmentPair)
		
		thisGroup = pd.DataFrame(data=treatmentResults, index=index)
		savepath = os.path.join(outdir, f'run{paramIndex}_wealthgroup{i+1}_responses.csv')
		# thisGroup.to_excel(savepath, freeze_panes=(0,0), index_label=params.name)
		thisGroup.to_csv(savepath, index_label=params.name)

	# put main results into a Series
	print('\nResults from simulation:\n')
	print(eqSimulator.results.dropna().to_string())

	print('\nMPCS:\n')
	print(mpcSimulator.results.dropna().to_string())

	print('\nMPCS out of news:\n')
	print(mpcNewsSimulator_shockNextPeriod.results.dropna().to_string())

	name_series = pd.Series({'Experiment':params.name})
	index_series = pd.Series({'Index':params.index})
	results = pd.concat([	name_series,
							index_series,
							params.series, 
							eqSimulator.results.dropna(),
							mpcSimulator.results.dropna(),
							mpcNewsSimulator_shockNextPeriod.results.dropna(),
							mpcNewsSimulator_shock2Years.results.dropna(),
							mpcNewsSimulator_loan.results.dropna(),
							])

	savepath = os.path.join(outdir,f'run{paramIndex}.pkl')
	results.to_pickle(savepath)

	mpcs_table = mpcsTable.create(params, mpcSimulator, 
		mpcNewsSimulator_shockNextPeriod,
		mpcNewsSimulator_shock2Years,
		mpcNewsSimulator_loan,
		)
	savepath = os.path.join(outdir,f'run{paramIndex}_mpcs_table.csv')
	# mpcs_table.to_excel(savepath, freeze_panes=(0,0), index_label=params.name)
	mpcs_table.to_csv(savepath, index_label=params.name)

#-----------------------------------------------------------#
#      PLOTS                                                #
#-----------------------------------------------------------#
ixvals = [0,params.nx//8,params.nx//6,params.nx//4,params.nx//3,params.nx//2,params.nx-1]
xvals = np.array([grids.x.flat[i] for i in ixvals])

icvals = [0,params.nc//8,params.nc//6,params.nc//4,params.nc//3,params.nc//2,params.nc-1]
cvals = np.array([grids.c.flat[i] for i in icvals])

def plot_policies():
	cSwitch = np.asarray(model.valueFunction) == np.asarray(model.valueSwitch)
	cPolicy = cSwitch * np.asarray(model.cSwitchingPolicy[:,:,:,:,0]) + (~cSwitch) * np.asarray(grids.c.matrix)

	print(xvals)

	if params.nyP == 1:
		iyP = 0
	else:
		iyP = 5

	fig, ax = plt.subplots(nrows=2,ncols=3)
	fig.suptitle('Consumption function vs. state c')
	i = 0
	for row in range(2):
		for col in range(3):
			ax[row,col].scatter(grids.c.flat,cPolicy[ixvals[i],:,0,iyP])
			ax[row,col].set_title(f'x = {xvals[i]}')
			ax[row,col].set_xlabel('c, state')
			ax[row,col].set_ylabel('actual consumption')
			i += 1

	fig, ax = plt.subplots(nrows=2,ncols=3)
	fig.suptitle('Consumption function vs. state c, zoomed')
	i = 0
	for row in range(2):
		for col in range(3):
			ax[row,col].scatter(grids.c.flat,cPolicy[ixvals[i],:,0,iyP])
			ax[row,col].set_title(f'x = {xvals[i]}')
			ax[row,col].set_xlabel('c, state')
			ax[row,col].set_ylabel('actual consumption')
			ax[row,col].set_xbound(0, 0.5)
			i += 1

	fig, ax = plt.subplots(nrows=2,ncols=3)
	fig.suptitle('EMAX vs. state c')
	i = 0
	for row in range(2):
		for col in range(3):
			ax[row,col].scatter(grids.c.flat,model.EMAX[ixvals[i],:,0,iyP])
			ax[row,col].set_title(f'x = {xvals[i]}')
			ax[row,col].set_xlabel('c, state')
			ax[row,col].set_ylabel('EMAX')
			i += 1

	fig, ax = plt.subplots(nrows=2,ncols=3)
	fig.suptitle('Consumption function vs. assets')
	i = 0
	for row in range(2):
		for col in range(3):
			ax[row,col].scatter(grids.x.flat,cPolicy[:,icvals[i],0,iyP])
			ax[row,col].set_title(f'c = {cvals[i]}')
			ax[row,col].set_xlabel('x, cash-on-hand')
			ax[row,col].set_ylabel('actual consumption')
			i += 1

	fig, ax = plt.subplots(nrows=2,ncols=3)
	fig.suptitle('EMAX vs. cash-on-hand')
	i = 0
	for row in range(2):
		for col in range(3):
			ax[row,col].scatter(grids.c.flat,model.EMAX[ixvals[i],:,0,iyP])
			ax[row,col].set_title(f'c = {cvals[i]}')
			ax[row,col].set_xlabel('cash-on-hand, x')
			ax[row,col].set_ylabel('EMAX')
			i += 1

	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	ax.scatter(grids.x.flat,model.inactionRegionLower[:,0,iyP])
	ax.scatter(grids.x.flat,model.cSwitchingPolicy[:,0,0,iyP])
	ax.scatter(grids.x.flat,model.inactionRegionUpper[:,0,iyP])

	ax.set_title('Inaction region for consumption')
	ax.set_xlabel('cash-on-hand, x')
	ax.set_ylabel('consumption')
	ax.legend(['Lower bound of inaction region', 'Desired c without adj cost', 
		'Upper bound of inaction region'])

	plt.show()

def plot_mpcs():
	if not Simulate:
		return

	ishock = 4
	idx_yP = np.asarray(mpcSimulator.finalStates['yPind']) == 5
	idx_yP = idx_yP.reshape((-1,1))
	mpcs = np.asarray(mpcSimulator.mpcs[ishock]).reshape((-1,1))
	cash = np.asarray(mpcSimulator.finalStates['xsim'])
	c = np.asarray(mpcSimulator.finalStates['csim'])

	fig, ax = plt.subplots(nrows=2, ncols=3)
	fig.suptitle('Quarterly MPC vs. initial cash-on-hand')
	i = 0
	for row in range(2):
		for col in range(3):
			idx_c = np.logical_and(c >= cvals[i], c < cvals[i+1])
			x = cash[np.logical_and(idx_yP,idx_c)]
			y = mpcs[np.logical_and(idx_yP,idx_c)]
			ax[row,col].scatter(x,y)
			ax[row,col].set_title(f'{cvals[i]:.3g} <= state c < {cvals[i+1]:.3g}')
			ax[row,col].set_xlabel('cash-on-hand, x')
			ax[row,col].set_ylabel('MPC out of 0.01')
			i += 1

	fig, ax = plt.subplots(nrows=2, ncols=3)
	fig.suptitle('Quarterly MPC vs. consumption state')
	i = 0
	for row in range(2):
		for col in range(3):
			idx_x = np.logical_and(cash >= xvals[i], cash < xvals[i+1])
			x = c[np.logical_and(idx_yP,idx_x)]
			y = mpcs[np.logical_and(idx_yP,idx_x)]
			ax[row,col].scatter(x,y)
			ax[row,col].set_title(f'{xvals[i]:.3g} <= x < {xvals[i+1]:.3g}')
			ax[row,col].set_xlabel('state c')
			ax[row,col].set_ylabel('MPC out of 0.01')
			i += 1

	plt.show()