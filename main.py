import os
import numpy as np
import pandas as pd

from model import Params, Income, Grid
from misc.calibrations import load_calibration
from misc.calibrations import load_replication
from misc import mpcsTable, functions, otherStatistics
from misc.Calibrator import Calibrator
from model.model import Model, ModelWithNews
from model import simulator
from misc import plots

#---------------------------------------------------------------#
#      FUNCTIONS                                                #
#---------------------------------------------------------------#
def create_objects(params, locIncome, PrintGrids):
	income = Income.Income(params, locIncome, False)
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
	valueNext, shockIndices, periodsUntilShock):
	"""
	Solves for the consumption function given a future
	shock, backward starting from the period immediately
	prior to the shock.
	"""
	multipleShocks = len(shockIndices) > 1
	nLast = len(shockIndices) + 1
	switching = np.zeros(
		(params.nx,1,params.nz,params.nyP,nLast))
	inaction = np.zeros(
		(params.nx,2,params.nz,params.nyP,nLast))

	ii = 0
	for ishock in shockIndices:
		# Period before the shock
		shock = params.MPCshocks[ishock]
		model = ModelWithNews(params, income, grids,
			valueNext, shock, 1)
		model.solve()

		# Now iterate backward, starting at two periods
		# before the shock
		for ip in range(2, periodsUntilShock+1):
			model = ModelWithNews(
				params, income, grids,
				model.valueFunction, shock, ip)
			model.solve()

		# Set output arrays to the policy functions associated
		# with this shock
		switching[:,:,:,:,ii] = model.cSwitchingPolicy[:,:,:,:,0]
		inaction[:,:,:,:,ii] = model.inactionRegion[:,:,:,:,0]

		ii += 1
		del model

	return (switching, inaction)

def main(paramIndex=None, runopts=None, replication=None):
	if runopts is None:
		# Default run options
		Calibrate = False # use solver to match targets
		Simulate = True
		SimulateMPCs = True
		MPCsNews = True
		Fast = False # run w/small grids for debugging
		PrintGrids = False
		MakePlots = False
	else:
		Calibrate = runopts['Calibrate']
		Simulate = runopts['Simulate']
		SimulateMPCs = runopts['SimulateMPCs']
		MPCsNews = runopts['MPCsNews']
		Fast = runopts['Fast']
		PrintGrids = runopts['PrintGrids']
		MakePlots = runopts['MakePlots']

	#---------------------------------------------------------------#
	#      HOUSEKEEPING                                             #
	#---------------------------------------------------------------#
	basedir = os.getcwd()
	locIncomeProcess = os.path.join(
		basedir, 'input', 'income_quarterly_b.mat')

	#---------------------------------------------------------------#
	#      CREATE PARAMS, GRIDS, AND INCOME OBJECTS                 #
	#---------------------------------------------------------------#
	functions.printLine()
	if replication is not None:
		params_dict = load_replication(replication)
	else:
		params_dict = load_calibration(index=paramIndex)
	functions.printLine()
	params_dict['fastSettings'] = Fast
	params = Params.Params(params_dict)
	grids, income = create_objects(params, locIncomeProcess, PrintGrids)

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

	eqSimulator = simulator.EquilibriumSimulator(params, income, grids)

	if Simulate:
		eqSimulator.initialize(model.cSwitchingPolicy, model.inactionRegion)
		eqSimulator.simulate()
		finalSimStates = eqSimulator.finalStates
	else:
		finalSimStates = []

	model.interpMat = None
	valueBaseline = model.valueFunction

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

	cSwitch_shockNextPeriod, inactionRegions_shockNextPeriod = solve_back_from_shock(
		params, income, grids, valueBaseline, shockIndices_shockNextPeriod, 1)

	cSwitch_shockNextPeriod[:,:,:,:,-1] = model.cSwitchingPolicy[:,:,:,:,0]
	inactionRegions_shockNextPeriod[:,:,:,:,-1] = model.inactionRegion[:,:,:,:,0]

	#-----------------------------------------------------------#
	#      SOLVE FOR 1-YEAR LOAN                                #
	#-----------------------------------------------------------#
	if SimulateMPCs and MPCsNews:
		cSwitch_loan, inactionRegions_loan = solve_back_from_shock(params,
			income, grids, valueBaseline, [0], 4)

		cSwitch_loan[:,:,:,:,-1] = model.cSwitchingPolicy[:,:,:,:,0]
		inactionRegions_loan[:,:,:,:,-1] = model.inactionRegion[:,:,:,:,0]

	#-----------------------------------------------------------#
	#      SHOCK OF -$500 IN 2 YEARS                            #
	#-----------------------------------------------------------#
	if SimulateMPCs and MPCsNews:
		cSwitch_shock2Years, inactionRegions_shock2Years = solve_back_from_shock(
			params, income, grids, valueBaseline, [2], 8)

		cSwitch_shock2Years[:,:,:,:,-1] = model.cSwitchingPolicy[:,:,:,:,0]
		inactionRegions_shock2Years[:,:,:,:,-1] = model.inactionRegion[:,:,:,:,0]

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