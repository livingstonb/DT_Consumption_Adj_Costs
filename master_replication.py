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
from misc.optim_constraints import constraint_transform
from misc.optim_constraints import constraint_transform_inv
from misc import mpcsTable
from model.model import Model, ModelWithNews
from model import simulator

#---------------------------------------------------------------#
#      LOOK FOR/CREATE OUTPUT DIRECTORY                         #
#---------------------------------------------------------------#
basedir = os.getcwd()
outdir = os.path.join(basedir, 'output')
if not os.path.exists(outdir):
	os.mkdir(outdir)

#---------------------------------------------------------------#
#      SET DIRECTORY CONTAINING INCOME PROCESS                  #
#---------------------------------------------------------------#
locIncomeProcess = os.path.join(
	basedir, 'input', 'quarterly_b.mat')

#---------------------------------------------------------------#
#      CHOOSE CALIBRATION                                       #
#---------------------------------------------------------------#
# 0 -- Baseline w/discount heterogeneity
# 1 -- Mean wealth target, no discount heterogeneity
# 2 -- Wealth constrained target, no discount heterogeneity
num_specification = 2
paramsDicts = []

#---------------------------------------------------------------#
#      BASELINE SPECIFICATION                                   #
#---------------------------------------------------------------#
width = 0.031928196837238
paramsDict = {}
paramsDict['name'] = f'Baseline'
paramsDict['adjustCost'] = 0.01
paramsDict['riskAver'] = 1
paramsDict['discount_factor_grid'] = np.array([-width, 0, width])
paramsDict['wealthTarget'] = 3.2
paramsDict['nx'] = 120
paramsDict['nc'] = 120
paramsDict['nSim'] = 1e5
paramsDict['locIncomeProcess'] = locIncomeProcess
paramsDict['timeDiscount'] = 0.875800159738212
paramsDicts.append(paramsDict)

#---------------------------------------------------------------#
#      MEAN WEALTH TARGET, NO BETA HETEROGENEITY                #
#---------------------------------------------------------------#
paramsDict = {}
paramsDict['name'] = f'Mean wealth target'
paramsDict['adjustCost'] = 0.004765897372858
paramsDict['riskAver'] = 1
paramsDict['wealthTarget'] = 3.2
paramsDict['nx'] = 120
paramsDict['nc'] = 120
paramsDict['nSim'] = 1e5
paramsDict['locIncomeProcess'] = locIncomeProcess
paramsDict['timeDiscount'] = 0.984599014194288
paramsDicts.append(paramsDict)

#---------------------------------------------------------------#
#      WEALTH < $1000 TARGET, NO BETA HETEROGENEITY             #
#---------------------------------------------------------------#
paramsDict = {}
paramsDict['name'] = f'Wealth constrained target'
paramsDict['riskAver'] = 1
paramsDict['nx'] = 80
paramsDict['nc'] = 80
paramsDict['nSim'] = 1e5
paramsDict['locIncomeProcess'] = locIncomeProcess
paramsDict['adjustCost'] = 0.005663097501924793 * 4
paramsDict['timeDiscount'] = 0.9657141937933638 ** 4
paramsDicts.append(paramsDict)

#---------------------------------------------------------------#
#      CREATE PARAMS OBJECT                                     #
#---------------------------------------------------------------#
params = Params.Params(paramsDicts[num_specification])

#---------------------------------------------------------------#
#      LOAD INCOME PROCESS                                      #
#---------------------------------------------------------------#
income = Income.Income(params)
params.addIncomeParameters(income)

#---------------------------------------------------------------#
#      CREATE GRIDS                                             #
#---------------------------------------------------------------#
grids = Grid.Grid(params, income)

#---------------------------------------------------------------#
#      CREATE MODEL                                             #
#---------------------------------------------------------------#
model = Model(params, income, grids)
model.initialize()

#-----------------------------------------------------------#
#      CALIBRATING TO FRACTION OF HHs WITH WEALTH <= $1000  #
#-----------------------------------------------------------#
# def calibrator(variables):
# 	newDiscount = constraint_transform(variables[0], 0, 0.98)
# 	params.resetDiscountRate(newDiscount)

# 	newAdjustCost = constraint_transform(variables[1], 0.0001, 0.05)
# 	params.resetAdjustCost(newAdjustCost)
# 	model.solve()

# 	eqSimulator = simulator.EquilibriumSimulator(params, income, grids, 
# 		model.cSwitchingPolicy, model.valueDiff)
# 	eqSimulator.simulate()
# 	finalSimStates = eqSimulator.finalStates

# 	shockIndices = [3]
# 	mpcSimulator = simulator.MPCSimulator(
# 		params, income, grids,
# 		model.cSwitchingPolicy,
# 		model.valueDiff,
# 		shockIndices,
# 		finalSimStates)
# 	mpcSimulator.simulate()

# 	rowname = f'P(Q1 MPC > 0) for shock of {params.MPCshocks[3]}'
# 	targets = np.array([
# 		eqSimulator.results['Wealth <= $1000'] - 0.23,
# 		mpcSimulator.results[rowname] - 0.2
# 		])

# 	print(f"\n\n --- P(a < $1000) = {eqSimulator.results['Wealth <= $1000']} ---\n")
# 	print(f" --- P(MPC > 0) = {mpcSimulator.results[rowname]} ---\n\n")

# 	return targets

# discount0_transf = constraint_transform_inv(0.961, 0, 0.98)
# adjustCost0_transf = constraint_transform_inv(0.0024, 0.0001, 0.05)
# x0 = np.array([discount0_transf, adjustCost0_transf])
# opt_results = optimize.root(calibrator, x0).x

#-----------------------------------------------------------#
#      SOLVE AND SIMULATE A FINAL TIME                      #
#-----------------------------------------------------------#
model.solve()

eqSimulator = simulator.EquilibriumSimulator(params, income, grids, 
	model.cSwitchingPolicy, model.valueDiff)
eqSimulator.simulate(final=True)

finalSimStates = eqSimulator.finalStates

#-----------------------------------------------------------#
#      SIMULATE MPCs OUT OF AN IMMEDIATE SHOCK              #
#-----------------------------------------------------------#
shockIndices = [0,1,2,3,4,5]

mpcSimulator = simulator.MPCSimulator(
	params, income, grids, 
	model.cSwitchingPolicy,
	model.valueDiff, 
	shockIndices, finalSimStates)

mpcSimulator.simulate()

#-----------------------------------------------------------#
#      SOLVE FOR POLICY GIVEN SHOCK NEXT PERIOD             #
#-----------------------------------------------------------#
shockIndices_shockNextPeriod = [2,3,4,5]
currentShockIndices = [6] * len(shockIndices_shockNextPeriod) # 6 is shock of 0

cSwitch_shockNextPeriod = np.zeros(
	(params.nx,1,params.nz,params.nyP,len(shockIndices_shockNextPeriod)+1))
valueDiffs_shockNextPeriod = np.zeros(
	(params.nx,params.nc,params.nz,params.nyP,len(shockIndices_shockNextPeriod)+1))

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

	print(f'Solving for shock of {params.MPCshocks[ishock]} next period')
	model_shockNextPeriod.solve()
	cSwitch_shockNextPeriod[:,:,:,:,i] = model_shockNextPeriod.cSwitchingPolicy[:,:,:,:,0]
	valueDiffs_shockNextPeriod[:,:,:,:,i] = model_shockNextPeriod.valueDiff[:,:,:,:,0]

	del model_shockNextPeriod
	i += 1

#-----------------------------------------------------------#
#      SOLVE FOR 1-YEAR LOAN                                #
#-----------------------------------------------------------#
cSwitch_loan = np.zeros((params.nx,1,params.nz,params.nyP,2))
valueDiffs_loan = np.zeros((params.nx,params.nc,params.nz,params.nyP,2))

ishock = 0
# Start with last quarter
model_loan = ModelWithNews(
	params, income, grids,
	valueBaseline,
	emaxBaseline,
	params.MPCshocks[ishock])

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

mpcNewsSimulator_shockNextPeriod.simulate()
mpcNewsSimulator_shock2Years.simulate()
mpcNewsSimulator_loan.simulate()

#-----------------------------------------------------------#
#      RESULTS                                              #
#-----------------------------------------------------------#
# Put main results into a Series
print('\nResults from simulation:\n')
print(eqSimulator.results.dropna().to_string())

print('\nMPCS:\n')
print(mpcSimulator.results.dropna().to_string())

print('\nMPCS out of news:\n')
print(mpcNewsSimulator_shockNextPeriod.results.dropna().to_string())

name_series = pd.Series({'Experiment':params.name})
index_series = pd.Series({'Index':params.index})
statistics = pd.concat([	name_series,
							index_series,
							params.series, 
							eqSimulator.results.dropna(),
							mpcSimulator.results.dropna(),
							mpcNewsSimulator_shockNextPeriod.results.dropna(),
							mpcNewsSimulator_shock2Years.results.dropna(),
							mpcNewsSimulator_loan.results.dropna(),
						])

savepath = os.path.join(outdir,f'statistics.csv')
statistics.to_csv(savepath, index_label=params.name, header=False)

mpcs_table = mpcsTable.create(params, mpcSimulator, 
	mpcNewsSimulator_shockNextPeriod,
	mpcNewsSimulator_shock2Years,
	mpcNewsSimulator_loan,
	)
savepath = os.path.join(outdir,f'mpcs_table.csv')
# mpcs_table.to_excel(savepath, freeze_panes=(0,0), index_label=params.name)
mpcs_table.to_csv(savepath, index_label=params.name)