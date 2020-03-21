from model.Params import Params
import numpy as np

from misc import Calibrator

def load_mean_wealth_replication(adjustCostOn, betaHeterogeneity):
	meanWealthGridParams = {
		'xGridTerm1Wt': 0.05,
		'xGridTerm1Curv': 0.8,
		'xGridCurv': 0.2,
		'xMax': 50,
		'borrowLim': 0,
		'cMin': 1e-6,
		'cMax': 5,
		'cGridTerm1Wt': 0.05,
		'cGridTerm1Curv': 0.9,
		'cGridCurv': 0.15,
	}

	###########################################################
	##### NO ADJUSTMENT COSTS, NO BETA HETEROGENEITY ##########
	###########################################################

	# Target mean wealth


def load_calibrations(locIncomeProcess, index=None, name=None):
	"""
	This function sets the parameterizations to be passed
	to a new Params object.
	"""

	if (index is None) and (name is None):
		raise Exception ('At least one specification must be chosen')

	default_values = Params()
	paramsDicts = []
	ii = 0

	beta_spacings = [0, 0.005, 0.01, 0.02, 0.032]

	if index is None:
		index_beta_het = 0
		index_adj = None
	else:
		nb = len(beta_spacings)
		index_beta_het = index // 4
		index_adj = index % 4

	beta_w = beta_spacings[index_beta_het]
	if beta_w == 0:
		discount_factor_grid = np.array([0.0])
	else:
		discount_factor_grid = np.array([-beta_w, 0.0, beta_w])

	print(f'Selected discount factor spacing = {beta_w}')

	###########################################################
	##### TARGET WEALTH < $1000 ###############################
	###########################################################
	targeted_shock = default_values.MPCshocks[3]
	targeted_stat = f'P(Q1 MPC > 0) for shock of {targeted_shock}'

	# Set shared parameters
	wealthConstrainedTarget = dict()
	wealthConstrainedTarget['riskAver'] = 1
	wealthConstrainedTarget['locIncomeProcess'] = locIncomeProcess
	wealthConstrainedTarget['timeDiscount'] = (0.96926309097 - beta_w) ** 4.0
	wealthConstrainedTarget['discount_factor_grid'] = discount_factor_grid

	timeDiscount_variable = Calibrator.OptimVariable(
		'timeDiscount', [0.94 - beta_w, 0.99 - beta_w],
		wealthConstrainedTarget['timeDiscount'] ** 0.25,
		scale=0.2)

	wealthConstrained_target = Calibrator.OptimTarget(
		'Wealth <= $1000', 0.23, 'Equilibrium')

	opts = {
		'norm_deg': 3,
		'norm_raise_to': 1,
	}
	solver_opts = Calibrator.SolverOptions(
		'minimize', other_opts=opts)

	# Without adjustment costs
	paramsDicts.append({})
	paramsDicts[ii] = wealthConstrainedTarget.copy()
	paramsDicts[ii]['name'] = 'Wealth constrained target w/o adj costs'
	paramsDicts[ii]['adjustCost'] = 0
	paramsDicts[ii]['cal_options'] = [
		[timeDiscount_variable],
		[wealthConstrained_target],
		solver_opts,
	]
	ii += 1

	# With adjustment costs
	paramsDicts.append({})
	paramsDicts[ii] = wealthConstrainedTarget.copy()
	paramsDicts[ii]['name'] = 'Wealth constrained target w/adj costs'
	paramsDicts[ii]['adjustCost'] = 4 * 0.000274119156

	adjustCost_variable = Calibrator.OptimVariable(
		'adjustCost', [0.000002, 0.001],
		paramsDicts[ii]['adjustCost'] / 4.0,
		scale=3)
	mpc_target = Calibrator.OptimTarget(
		targeted_stat, 0.2, 'MPC')

	paramsDicts[ii]['cal_options'] = [
		[adjustCost_variable, timeDiscount_variable],
		[mpc_target, wealthConstrained_target],
		solver_opts,
	]
	ii += 1

	###########################################################
	##### TARGET 3.2 MEAN WEALTH ##############################
	###########################################################
	targeted_shock = default_values.MPCshocks[3]
	targeted_stat = f'P(Q1 MPC > 0) for shock of {targeted_shock}'

	beta0 = 0.996263091 - beta_w
	chi0 = 1.19049306771e-05

	if beta_w > 0.02:
		beta0 = 0.966602
		chi0 = 0.0008502

	# Set shared parameters
	meanWealthTarget = dict()
	meanWealthTarget['riskAver'] = 1
	meanWealthTarget['locIncomeProcess'] = locIncomeProcess
	meanWealthTarget['timeDiscount'] = beta0 ** 4.0
	meanWealthTarget['xMax'] = 50
	meanWealthTarget['discount_factor_grid'] = discount_factor_grid

	meanWealthTarget['xGridTerm1Wt'] = 0.05
	meanWealthTarget['xGridTerm1Curv'] = 0.8
	meanWealthTarget['xGridCurv'] = 0.2
	meanWealthTarget['borrowLim'] = 0

	meanWealthTarget['cMin'] = 1e-6
	meanWealthTarget['cMax'] = 5
	meanWealthTarget['cGridTerm1Wt'] = 0.05
	meanWealthTarget['cGridTerm1Curv'] = 0.9
	meanWealthTarget['cGridCurv'] = 0.15

	timeDiscount_variable = Calibrator.OptimVariable(
		'timeDiscount', [0.97 - beta_w, 0.9995 - beta_w],
		beta0, scale=0.2)

	meanWealth_target = Calibrator.OptimTarget(
		'Mean wealth', 3.2, 'Equilibrium')

	opts = {
		'norm_deg': 3,
		'norm_raise_to': 1,
	}
	solver_opts = Calibrator.SolverOptions(
		'minimize', other_opts=opts)
	
	# Without adjustment costs
	paramsDicts.append({})
	paramsDicts[ii] = meanWealthTarget.copy()
	paramsDicts[ii]['name'] = 'Mean wealth target w/o adj costs'
	paramsDicts[ii]['adjustCost'] = 0
	paramsDicts[ii]['cal_options'] = [
		[timeDiscount_variable],
		[meanWealth_target],
		solver_opts,
	]
	ii += 1

	# With adjustment costs
	paramsDicts.append({})
	paramsDicts[ii] = meanWealthTarget.copy()
	paramsDicts[ii]['name'] = 'Mean wealth target w/adj costs'
	paramsDicts[ii]['adjustCost'] = 4.0 * chi0

	adjustCost_variable = Calibrator.OptimVariable(
		'adjustCost', [0.000002, 0.001],
		chi0, scale=3.0)
	mpc_target = Calibrator.OptimTarget(
		targeted_stat, 0.2, 'MPC')

	paramsDicts[ii]['cal_options'] = [
		[adjustCost_variable, timeDiscount_variable],
		[mpc_target, meanWealth_target],
		solver_opts,
	]
	ii += 1

	#-----------------------------------------------------#
	#        CREATE PARAMS OBJECT, DO NOT CHANGE          #
	#-----------------------------------------------------#
	if index_adj is not None:
		chosenParameters = paramsDicts[index_adj]
		chosenParameters['index'] = index
		print(f'Selected parameterization #{index}:')
		print(f"\t{chosenParameters['name']}")
	else:
		indexFound = False
		for ii in range(len(paramsDicts)):
			if paramsDicts[ii]['name'] == name:
				chosenParameters = paramsDicts[ii]
				chosenParameters['index'] = ii
				print(f'Selected parameterization {ii}')
				indexFound = True

		if not indexFound:
			raise Exception('Parameter name not found')

	return chosenParameters