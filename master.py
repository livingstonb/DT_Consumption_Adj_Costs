import sys
import os
import numpy as np
import pandas as pd

from model import Params, Income, Grid
from misc.calibrations import load_calibrations
from misc import mpcsTable, functions, otherStatistics
from misc.Calibrator import Calibrator
from model.model import Model, ModelWithNews
from model import simulator
from misc import plots

#---------------------------------------------------------------#
#      FUNCTIONS                                                #
#---------------------------------------------------------------#
def set_from_cmd_arg(cmd_line_args):
	"""
	Returns the first integer-valued command-line
	argument passed, if available.
	"""
	for arg in cmd_line_args:
		try:
			return int(arg)
		except:
			pass

	return None

def create_objects(params, PrintGrids):
	income = Income.Income(params, False)
	params.addIncomeParameters(income)

	grids = Grid.Grid(params, income)

	if PrintGrids:
		print('Cash grid:')
		functions.printVector(grids.x_flat)
		print('Consumption grid:')
		functions.printVector(grids.c_flat)
		quit()

	return (grids, income)

def solve_back_from_shock(params, income, grids,
	valueNext, shock, periodsUntilShock):

	model = ModelWithNews(params, income, grids,
		valueNext, shock, 1)
	model.solve()

	for ip in range(2, periodsUntilShock+1):
		model = ModelWithNews(
			params, income, grids,
			model.valueFunction, shock, ip)
		model.solve()

	return model

#---------------------------------------------------------------#
#      CHOOSE PARAMETERIZATION, SET OPTIONS                     #
#---------------------------------------------------------------#
# One of these will be used if no command line arg is passed.
# If both are set to 'None' and no arg is passed, paramIndex
# will be set to 0.
# paramName is favored over paramIndex
paramIndex = None
paramName = None

# Run options
Calibrate = True # use solver to match targets
Simulate = True
SimulateMPCs = True
MPCsNews = True
Fast = False # run w/small grids for debugging
PrintGrids = False
MakePlots = False

#---------------------------------------------------------------#
#      HOUSEKEEPING                                             #
#---------------------------------------------------------------#
indexPassed = set_from_cmd_arg(sys.argv)
if indexPassed is not None:
	paramIndex = indexPassed
	param_kwargs = {'index': paramIndex}
elif paramIndex is not None:
	param_kwargs = {'index': paramIndex}
elif paramName is not None:
	param_kwargs = {'name': paramName}
	paramIndex = 0
else:
	paramIndex = 0
	param_kwargs = {'index': paramIndex}

basedir = os.getcwd()
outdir = os.path.join(basedir, 'output')
if not os.path.exists(outdir):
	os.mkdir(outdir)

locIncomeProcess = os.path.join(
	basedir,'input', 'income_quarterly_b.mat')

#---------------------------------------------------------------#
#      CREATE PARAMS, GRIDS, AND INCOME OBJECTS                 #
#---------------------------------------------------------------#
params_dict = load_calibrations(locIncomeProcess, **param_kwargs)
params_dict['fastSettings'] = Fast
params = Params.Params(params_dict)
grids, income = create_objects(params, PrintGrids)

#---------------------------------------------------------------#
#      INITIALIZE AND SOLVE MODEL                               #
#---------------------------------------------------------------#
model = Model(params, income, grids)
model.initialize()

if Calibrate:
	calibrator = Calibrator(*params.cal_options)
	opt_results = calibrator.calibrate(params, model, income, grids)
	calibrator = None
else:
	model.solve()
model.interpMat = None

eqSimulator = simulator.EquilibriumSimulator(params, income, grids)

if Simulate:
	eqSimulator.initialize(model.cSwitchingPolicy, model.inactionRegion)
	eqSimulator.simulate()
	finalSimStates = eqSimulator.finalStates
else:
	finalSimStates = []

#-----------------------------------------------------------#
#      SIMULATE MPCs OUT OF AN IMMEDIATE SHOCK              #
#-----------------------------------------------------------#
shockIndices = [0,1,2,3,4,5]

mpcSimulator = simulator.MPCSimulator(
	params, income, grids,
	shockIndices)

if Simulate and SimulateMPCs:
	mpcSimulator.initialize(
		model.cSwitchingPolicy, model.inactionRegion,
		finalSimStates)
	mpcSimulator.simulate()

#-----------------------------------------------------------#
#      SOLVE FOR POLICY GIVEN SHOCK NEXT PERIOD             #
#-----------------------------------------------------------#
shockIndices_shockNextPeriod = [2,3,4,5]
currentShockIndices = [6] * len(shockIndices_shockNextPeriod) # 6 is shock of 0

nmodels = len(shockIndices_shockNextPeriod) + 1
cSwitch_shockNextPeriod = np.zeros(
	(params.nx,1,params.nz,params.nyP,nmodels))
inactionRegions_shockNextPeriod = np.zeros(
	(params.nx,2,params.nz,params.nyP,nmodels))

cSwitch_shockNextPeriod[:,:,:,:,-1] = model.cSwitchingPolicy[:,:,:,:,0]
inactionRegions_shockNextPeriod[:,:,:,:,-1] = model.inactionRegion[:,:,:,:,0]
valueBaseline = model.valueFunction

i = 0
periodsUntilShock = 1
for ishock in shockIndices_shockNextPeriod:
	model_shockNextPeriod = ModelWithNews(
		params, income, grids,
		valueBaseline,
		params.MPCshocks[ishock],
		periodsUntilShock)

	if SimulateMPCs and MPCsNews:
		print(f'Solving for shock of {params.MPCshocks[ishock]} next period')
		model_shockNextPeriod.solve()
		cSwitch_shockNextPeriod[:,:,:,:,i] = model_shockNextPeriod.cSwitchingPolicy[:,:,:,:,0]
		inactionRegions_shockNextPeriod[:,:,:,:,i] = model_shockNextPeriod.inactionRegion[:,:,:,:,0]

	del model_shockNextPeriod
	i += 1

#-----------------------------------------------------------#
#      SOLVE FOR 1-YEAR LOAN                                #
#-----------------------------------------------------------#
cSwitch_loan = np.zeros((params.nx,1,params.nz,params.nyP,2))
inactionRegions_loan = np.zeros((params.nx,2,params.nz,params.nyP,2))


if SimulateMPCs and MPCsNews:
	shock = params.MPCshocks[0]
	model_loan = solve_back_from_shock(params,
		income, grids, valueBaseline, shock, 4)

	cSwitch_loan[:,:,:,:,0] = model_loan.cSwitchingPolicy[:,:,:,:,0]
	cSwitch_loan[:,:,:,:,1] = model.cSwitchingPolicy[:,:,:,:,0]
	inactionRegions_loan[:,:,:,:,0] = model_loan.inactionRegion[:,:,:,:,0]
	inactionRegions_loan[:,:,:,:,1] = model.inactionRegion[:,:,:,:,0]
	del model_loan

#-----------------------------------------------------------#
#      SHOCK OF -$500 IN 2 YEARS                            #
#-----------------------------------------------------------#
cSwitch_shock2Years = np.zeros((params.nx,1,params.nz,params.nyP,2))
inactionRegions_shock2Years = np.zeros((params.nx,2,params.nz,params.nyP,2))

if SimulateMPCs and MPCsNews:
	shock = params.MPCshocks[2]
	model_shock2Years = solve_back_from_shock(params,
		income, grids, valueBaseline, shock, 8)

	cSwitch_shock2Years[:,:,:,:,0] = model_shock2Years.cSwitchingPolicy[:,:,:,:,0]
	cSwitch_shock2Years[:,:,:,:,1] = model.cSwitchingPolicy[:,:,:,:,0]
	inactionRegions_shock2Years[:,:,:,:,0] = model_shock2Years.inactionRegion[:,:,:,:,0]
	inactionRegions_shock2Years[:,:,:,:,1] = model.inactionRegion[:,:,:,:,0]
	del model_shock2Years

#-----------------------------------------------------------#
#      SIMULATE MPCs OUT OF NEWS                            #
#-----------------------------------------------------------#
currentShockIndices = [6] * len(shockIndices_shockNextPeriod)
mpcNewsSimulator_shockNextPeriod = simulator.MPCSimulatorNews(
	params, income, grids,
	shockIndices_shockNextPeriod,
	currentShockIndices,
	periodsUntilShock=1)

shockIndices_shock2Years = [2]
currentShockIndices = [6]
mpcNewsSimulator_shock2Years = simulator.MPCSimulatorNews(
	params, income, grids,
	shockIndices_shock2Years,
	currentShockIndices,
	periodsUntilShock=8)

shockIndices_loan = [0]
currentShockIndices = [5]
mpcNewsSimulator_loan = simulator.MPCSimulatorNews_Loan(
	params, income, grids,
	shockIndices_loan,
	currentShockIndices,
	periodsUntilShock=4)

if SimulateMPCs and MPCsNews:
	mpcNewsSimulator_shockNextPeriod.initialize(
		cSwitch_shockNextPeriod,
		inactionRegions_shockNextPeriod,
		finalSimStates)
	mpcNewsSimulator_shockNextPeriod.simulate()

	mpcNewsSimulator_shock2Years.initialize(
		cSwitch_shock2Years,
		inactionRegions_shock2Years,
		finalSimStates)
	mpcNewsSimulator_shock2Years.simulate()

	mpcNewsSimulator_loan.initialize(
		cSwitch_loan,
		inactionRegions_loan,
		finalSimStates)
	mpcNewsSimulator_loan.simulate()

#-----------------------------------------------------------#
#      RESULTS                                              #
#-----------------------------------------------------------#
# parameters
print('\nSelected parameters:\n')
print(params.series.to_string())

if Simulate:
	if MPCsNews:
		otherStatistics.saveWealthGroupStats(
			mpcSimulator, mpcNewsSimulator_shockNextPeriod,
			mpcNewsSimulator_shock2Years, mpcNewsSimulator_loan,
			finalSimStates, outdir, paramIndex, params)

	# put main results into a Series
	print('\nResults from simulation:\n')
	print(eqSimulator.results.dropna().to_string())

if SimulateMPCs:
	print('\nMPCS:\n')
	print(mpcSimulator.results.dropna().to_string())

if MPCsNews:
	print('\nMPCS out of news:\n')
	print(mpcNewsSimulator_shockNextPeriod.results.dropna().to_string())

	print('\nMPCS out of future loss:\n')
	print(mpcNewsSimulator_shock2Years.results.dropna().to_string())
	print(mpcNewsSimulator_loan.results.dropna().to_string())

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

savepath = os.path.join(outdir,f'run{paramIndex}_statistics.csv')
results.to_csv(savepath, index_label=params.name, header=True)

mpcs_table = mpcsTable.create(params, mpcSimulator, 
	mpcNewsSimulator_shockNextPeriod,
	mpcNewsSimulator_shock2Years,
	mpcNewsSimulator_loan,
	)
savepath = os.path.join(outdir,f'run{paramIndex}_mpcs_table.csv')
# mpcs_table.to_excel(savepath, freeze_panes=(0,0), index_label=params.name)
mpcs_table.to_csv(savepath, index_label=params.name, header=True)

#-----------------------------------------------------------#
#      PLOTS                                                #
#-----------------------------------------------------------#
if MakePlots:
	plots.plot_policies(model, grids, params, paramIndex, outdir)