import os
import numpy as np
import pandas as pd

from model import Params, Income, Grid
from misc import mpcsTable, functions, otherStatistics
from misc.Calibrator import Calibrator
from model.model import Model, ModelWithNews
from misc.calibrations import load_replication, load_calibration
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
	valueNext, shockIndices, periodsUntilShock,
	baselineSwitch, baselineInaction):
	"""
	Solves for the consumption function given a future
	shock, backward starting from the period immediately
	prior to the shock. Sets the policy functions a the
	end of the last dimension to the baseline policy
	functions.
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

	switching[:,:,:,:,nLast-1] = baselineSwitch[:,:,:,:,0]
	inaction[:,:,:,:,nLast-1] = baselineInaction[:,:,:,:,0]

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
	outdir = os.path.join(basedir, 'output')

	if not os.path.exists(outdir):
		os.mkdir(outdir)

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
	news = dict()
	news['shockIndices'] = [2,3,4,5]
	news['currentShockIndices'] = [6] * 4
	news['periodsUntilShock'] = 1

	news['cSwitch'], news['inactionRegions'] \
		= solve_back_from_shock(params, income, grids,
			valueBaseline, news['shockIndices'],
			news['periodsUntilShock'],
			model.cSwitchingPolicy, model.inactionRegion)

	# news['cSwitch'][:,:,:,:,-1] = model.cSwitchingPolicy[:,:,:,:,0]
	# news['inactionRegions'][:,:,:,:,-1] = model.inactionRegion[:,:,:,:,0]

	#-----------------------------------------------------------#
	#      SOLVE FOR 1-YEAR LOAN                                #
	#-----------------------------------------------------------#
	loan = dict()
	loan['shockIndices'] = [0]
	loan['currentShockIndices'] = [5]
	loan['periodsUntilShock'] = 4

	if SimulateMPCs and MPCsNews:
		loan['cSwitch'], loan['inactionRegions'] = \
			solve_back_from_shock(params, income, grids,
				valueBaseline, loan['shockIndices'],
				loan['periodsUntilShock'],
				model.cSwitchingPolicy, model.inactionRegion)

	#-----------------------------------------------------------#
	#      SHOCK OF -$500 IN 2 YEARS                            #
	#-----------------------------------------------------------#
	loss2years = dict()
	loss2years['shockIndices'] = [2]
	loss2years['currentShockIndices'] = [6]
	loss2years['periodsUntilShock'] = 8

	if SimulateMPCs and MPCsNews:
		print('Solving for policy functions given future shock')
		loss2years['cSwitch'], loss2years['inactionRegions'] = \
			solve_back_from_shock(params, income, grids,
				valueBaseline, loss2years['shockIndices'],
				loss2years['periodsUntilShock'],
				model.cSwitchingPolicy, model.inactionRegion)

	#-----------------------------------------------------------#
	#      SIMULATE MPCs OUT OF NEWS                            #
	#-----------------------------------------------------------#
	ii = 0
	for newsModel in [news, loan, loss2years]:
		sim_args = [
			params, income, grids,
			newsModel['shockIndices'],
			newsModel['currentShockIndices'],
		]

		if ii == 1:
			newsModel['simulator'] = simulator.MPCSimulatorNews_Loan(
				*sim_args, periodsUntilShock=newsModel['periodsUntilShock'])
		else:
			newsModel['simulator'] = simulator.MPCSimulatorNews(
				*sim_args, periodsUntilShock=newsModel['periodsUntilShock'])

		if SimulateMPCs and MPCsNews:
			print('Simulating MPCs out of news')
			newsModel['simulator'].initialize(
				newsModel['cSwitch'], newsModel['inactionRegions'],
				finalSimStates)
			newsModel['simulator'].simulate()

		ii += 1

	#-----------------------------------------------------------#
	#      RESULTS                                              #
	#-----------------------------------------------------------#
	# parameters
	print('\nSelected parameters:\n')
	print(params.series.to_string())

	if Simulate:
		if MPCsNews:
			otherStatistics.saveWealthGroupStats(
				mpcSimulator, news['simulator'],
				loss2years['simulator'], loan['simulator'],
				finalSimStates, outdir, paramIndex, params)

		# put main results into a Series
		print('\nResults from simulation:\n')
		print(eqSimulator.results.dropna().to_string())

	if SimulateMPCs:
		print('\nMPCS:\n')
		print(mpcSimulator.results.dropna().to_string())

	if MPCsNews:
		print('\nMPCS out of news:\n')
		print(news['simulator'].results.dropna().to_string())

		print('\nMPCS out of future loss:\n')
		print(loss2years['simulator'].results.dropna().to_string())
		print(loan['simulator'].results.dropna().to_string())

	name_series = pd.Series({'Experiment':params.name})
	index_series = pd.Series({'Index':params.index})
	results = pd.concat([	name_series,
							index_series,
							params.series, 
							eqSimulator.results.dropna(),
							mpcSimulator.results.dropna(),
							news['simulator'].results.dropna(),
							loss2years['simulator'].results.dropna(),
							loan['simulator'].results.dropna(),
							])

	savepath = os.path.join(outdir,f'run{paramIndex}.pkl')
	results.to_pickle(savepath)

	savepath = os.path.join(outdir,f'run{paramIndex}_statistics.csv')
	results.to_csv(savepath, index_label=params.name, header=True)

	mpcs_table = mpcsTable.create(params, mpcSimulator, 
		news['simulator'], loss2years['simulator'],
		loan['simulator'],
		)
	savepath = os.path.join(outdir,f'run{paramIndex}_mpcs_table.csv')
	# mpcs_table.to_excel(savepath, freeze_panes=(0,0), index_label=params.name)
	mpcs_table.to_csv(savepath, index_label=params.name, header=True)

	#-----------------------------------------------------------#
	#      PLOTS                                                #
	#-----------------------------------------------------------#
	if MakePlots:
		plots.plot_policies(model, grids, params, paramIndex, outdir)