from model.Params import Params
import numpy as np

from misc import Calibrator

from IPython.core.debugger import set_trace

def load_specifications(locIncomeProcess, index=None, name=None):
	"""
	This function defines the parameterizations and passes the
	one selected by 'index' or 'name' to a new Params object.
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

	# ###########################################################
	# ##### TARGET P(assets<$1000) AND P(MPC>0) = 0.2, w/HET ####
	# ###########################################################
	# w = 0.031928196837238
	# paramsDicts.append({})
	# paramsDicts[ii]['name'] = f'3-pt discount factor w/width{w}'
	# paramsDicts[ii]['index'] = ii
	# paramsDicts[ii]['adjustCost'] = 0.01
	# paramsDicts[ii]['noPersIncome'] = False
	# paramsDicts[ii]['riskAver'] = 1
	# paramsDicts[ii]['discount_factor_grid'] = np.array([-w, 0, w])
	# paramsDicts[ii]['nx'] = 120
	# paramsDicts[ii]['nc'] = 120
	# paramsDicts[ii]['nSim'] = 1e5
	# paramsDicts[ii]['locIncomeProcess'] = locIncomeProcess
	# paramsDicts[ii]['timeDiscount'] = 0.875800159738212

	# ii += 1

	###########################################################
	##### ASSETS 3.5 NO ADJ COSTS BASELINE ####################
	###########################################################
	# paramsDicts.append({})
	# paramsDicts[ii]['name'] = f'baseline_Q'
	# paramsDicts[ii]['index'] = ii
	# paramsDicts[ii]['riskAver'] = 1
	# paramsDicts[ii]['nx'] = 50
	# paramsDicts[ii]['nc'] = 80
	# paramsDicts[ii]['cMax'] = 15
	# paramsDicts[ii]['xMax'] = 40
	# paramsDicts[ii]['nSim'] = 5e5
	# paramsDicts[ii]['locIncomeProcess'] = locIncomeProcess
	# paramsDicts[ii]['adjustCost'] = 0.00002
	# paramsDicts[ii]['timeDiscount'] = 0.995939 ** 4
	# paramsDicts[ii]['cGridCurv'] = 0.2
	# paramsDicts[ii]['govTransfer'] = 0
	# paramsDicts[ii]['MPCshocks'] = [-0.1, -0.01, -1e-5, 1e-5, 0.01, 0.1, 0]

	# # paramsDicts[ii]['nx'] = 45
	# # paramsDicts[ii]['nc'] = 45
	# # paramsDicts[ii]['nSim'] = 1e5
	# paramsDicts[ii]['minGridSpacing'] = 0
	# ii += 1

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

	# Set shared parameters
	meanWealthTarget = dict()
	meanWealthTarget['riskAver'] = 1
	meanWealthTarget['locIncomeProcess'] = locIncomeProcess
	meanWealthTarget['timeDiscount'] = (0.996263091 - beta_w) ** 4.0
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
		meanWealthTarget['timeDiscount'] ** 0.25,
		scale=0.2)

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
	paramsDicts[ii]['adjustCost'] = 4.0 * 1.19049306771e-05

	adjustCost_variable = Calibrator.OptimVariable(
		'adjustCost', [0.000002, 0.001],
		paramsDicts[ii]['adjustCost'] / 4.0,
		scale=3)
	mpc_target = Calibrator.OptimTarget(
		targeted_stat, 0.2, 'MPC')

	paramsDicts[ii]['cal_options'] = [
		[adjustCost_variable, timeDiscount_variable],
		[mpc_target, meanWealth_target],
		solver_opts,
	]
	ii += 1

	# ###########################################################
	# ##### 3-PT DISCOUNT FACTOR HETEROGENEITY ##################
	# ###########################################################

	# adjustCost = 0.005
	# discount_widths = [	0.031262804098703,
	# 					0.019123368101102,
	# 					0.013129768413022,
	# 					]
	# wealthTarget = 3.2

	# for w in discount_widths:
	# 	paramsDicts.append({})
	# 	paramsDicts[ii]['name'] = f'3-pt discount factor w/width{w}'
	# 	paramsDicts[ii]['index'] = ii
	# 	paramsDicts[ii]['adjustCost'] = adjustCost
	# 	paramsDicts[ii]['noPersIncome'] = False
	# 	paramsDicts[ii]['riskAver'] = 1
	# 	paramsDicts[ii]['discount_factor_grid'] = np.array([-w, 0, w])
	# 	paramsDicts[ii]['wealthTarget'] = wealthTarget
	# 	paramsDicts[ii]['nx'] = 120
	# 	paramsDicts[ii]['nc'] = 120
	# 	paramsDicts[ii]['nSim'] = 1e5
	# 	paramsDicts[ii]['locIncomeProcess'] = locIncomeProcess
	# 	ii += 1

	# adjustCost = 0.01
	# discount_widths = [	0.031928196837238,
	# 					0.018806744625507,
	# 					0.013045865727511,
	# 					]
	# wealthTarget = 3.2

	# for w in discount_widths:
	# 	paramsDicts.append({})
	# 	paramsDicts[ii]['name'] = f'3-pt discount factor w/width{w}'
	# 	paramsDicts[ii]['index'] = ii
	# 	paramsDicts[ii]['adjustCost'] = adjustCost
	# 	paramsDicts[ii]['noPersIncome'] = False
	# 	paramsDicts[ii]['riskAver'] = 1
	# 	paramsDicts[ii]['discount_factor_grid'] = np.array([-w, 0, w])
	# 	paramsDicts[ii]['wealthTarget'] = wealthTarget
	# 	paramsDicts[ii]['nx'] = 120
	# 	paramsDicts[ii]['nc'] = 120
	# 	paramsDicts[ii]['nSim'] = 1e5
	# 	paramsDicts[ii]['locIncomeProcess'] = locIncomeProcess
	# 	ii += 1

	# adjustCost = 0.025
	# discount_widths = [	0.035365099279654,
	# 					0.019339420312185,
	# 					0.013597038299208,
	# 					]
	# wealthTarget = 3.2

	# for w in discount_widths:
	# 	paramsDicts.append({})
	# 	paramsDicts[ii]['name'] = f'3-pt discount factor w/width{w}'
	# 	paramsDicts[ii]['index'] = ii
	# 	paramsDicts[ii]['adjustCost'] = adjustCost
	# 	paramsDicts[ii]['noPersIncome'] = False
	# 	paramsDicts[ii]['riskAver'] = 1
	# 	paramsDicts[ii]['discount_factor_grid'] = np.array([-w, 0, w])
	# 	paramsDicts[ii]['wealthTarget'] = wealthTarget
	# 	paramsDicts[ii]['nx'] = 120
	# 	paramsDicts[ii]['nc'] = 120
	# 	paramsDicts[ii]['nSim'] = 1e5
	# 	paramsDicts[ii]['locIncomeProcess'] = locIncomeProcess
	# 	ii += 1

	# adjustCost = 0.05
	# discount_widths = [	0.04152681089598,
	# 					0.02,
	# 					0.014246309]
	# wealthTarget = 3.2

	# for w in discount_widths:
	# 	paramsDicts.append({})
	# 	paramsDicts[ii]['name'] = f'3-pt discount factor w/width{w}'
	# 	paramsDicts[ii]['index'] = ii
	# 	paramsDicts[ii]['adjustCost'] = adjustCost
	# 	paramsDicts[ii]['noPersIncome'] = False
	# 	paramsDicts[ii]['riskAver'] = 1
	# 	paramsDicts[ii]['discount_factor_grid'] = np.array([-w, 0, w])
	# 	paramsDicts[ii]['wealthTarget'] = wealthTarget
	# 	paramsDicts[ii]['nx'] = 120
	# 	paramsDicts[ii]['nc'] = 120
	# 	paramsDicts[ii]['nSim'] = 1e5
	# 	paramsDicts[ii]['locIncomeProcess'] = locIncomeProcess
	# 	ii += 1

	# ###########################################################
	# ##### 3-PT DISCOUNT FACTOR HETEROGENEITY: RA = 2 ##########
	# ###########################################################

	# adjustCost = 0.005
	# discount_widths = [	0.035660061154208,
	# 					0.053602353291592]
	# wealthTarget = 3.2

	# for w in discount_widths:
	# 	paramsDicts.append({})
	# 	paramsDicts[ii]['name'] = f'3-pt discount factor w/width{w}'
	# 	paramsDicts[ii]['index'] = ii
	# 	paramsDicts[ii]['adjustCost'] = adjustCost
	# 	paramsDicts[ii]['noPersIncome'] = False
	# 	paramsDicts[ii]['riskAver'] = 2
	# 	paramsDicts[ii]['discount_factor_grid'] = np.array([-w, 0, w])
	# 	paramsDicts[ii]['wealthTarget'] = wealthTarget
	# 	paramsDicts[ii]['nx'] = 120
	# 	paramsDicts[ii]['nc'] = 120
	# 	paramsDicts[ii]['nSim'] = 1e5
	# 	paramsDicts[ii]['locIncomeProcess'] = locIncomeProcess
	# 	ii += 1

	# adjustCost = 0.01
	# discount_widths = [	0.05376130747098,
	# 					0.035595816161569,
	# 					]
	# wealthTarget = 3.2

	# for w in discount_widths:
	# 	paramsDicts.append({})
	# 	paramsDicts[ii]['name'] = f'3-pt discount factor w/width{w}'
	# 	paramsDicts[ii]['index'] = ii
	# 	paramsDicts[ii]['adjustCost'] = adjustCost
	# 	paramsDicts[ii]['noPersIncome'] = False
	# 	paramsDicts[ii]['riskAver'] = 2
	# 	paramsDicts[ii]['discount_factor_grid'] = np.array([-w, 0, w])
	# 	paramsDicts[ii]['wealthTarget'] = wealthTarget
	# 	paramsDicts[ii]['nx'] = 120
	# 	paramsDicts[ii]['nc'] = 120
	# 	paramsDicts[ii]['nSim'] = 1e5
	# 	paramsDicts[ii]['locIncomeProcess'] = locIncomeProcess
	# 	ii += 1

	# adjustCost = 0.025
	# discount_widths = [	0.03510498642002,
	# 					0.05344428679477]
	# wealthTarget = 3.2

	# for w in discount_widths:
	# 	paramsDicts.append({})
	# 	paramsDicts[ii]['name'] = f'3-pt discount factor w/width{w}'
	# 	paramsDicts[ii]['index'] = ii
	# 	paramsDicts[ii]['adjustCost'] = adjustCost
	# 	paramsDicts[ii]['noPersIncome'] = False
	# 	paramsDicts[ii]['riskAver'] = 2
	# 	paramsDicts[ii]['discount_factor_grid'] = np.array([-w, 0, w])
	# 	paramsDicts[ii]['wealthTarget'] = wealthTarget
	# 	paramsDicts[ii]['nx'] = 120
	# 	paramsDicts[ii]['nc'] = 120
	# 	paramsDicts[ii]['nSim'] = 1e5
	# 	paramsDicts[ii]['locIncomeProcess'] = locIncomeProcess
	# 	ii += 1

	# adjustCost = 0.05
	# discount_widths = [	0.035172176637578,
	# 					0.053044165114212]
	# wealthTarget = 3.2

	# for w in discount_widths:
	# 	paramsDicts.append({})
	# 	paramsDicts[ii]['name'] = f'3-pt discount factor w/width{w}'
	# 	paramsDicts[ii]['index'] = ii
	# 	paramsDicts[ii]['adjustCost'] = adjustCost
	# 	paramsDicts[ii]['noPersIncome'] = False
	# 	paramsDicts[ii]['riskAver'] = 2
	# 	paramsDicts[ii]['discount_factor_grid'] = np.array([-w, 0, w])
	# 	paramsDicts[ii]['wealthTarget'] = wealthTarget
	# 	paramsDicts[ii]['nx'] = 120
	# 	paramsDicts[ii]['nc'] = 120
	# 	paramsDicts[ii]['nSim'] = 1e5
	# 	paramsDicts[ii]['locIncomeProcess'] = locIncomeProcess
	# 	ii += 1

	# ###########################################################
	# ##### 5-PT DISCOUNT FACTOR HETEROGENEITY ##################
	# ###########################################################

	# adjustCost = 0.005
	# discount_widths = [	0.01651039812107,
	# 					0.009734770875813,
	# 					0.006790600131364,
	# 					]
	# wealthTarget = 3.2

	# for w in discount_widths:
	# 	paramsDicts.append({})
	# 	paramsDicts[ii]['name'] = f'5-pt discount factor w/width{w}'
	# 	paramsDicts[ii]['index'] = ii
	# 	paramsDicts[ii]['adjustCost'] = adjustCost
	# 	paramsDicts[ii]['noPersIncome'] = False
	# 	paramsDicts[ii]['riskAver'] = 1
	# 	paramsDicts[ii]['discount_factor_grid'] = np.array([-2*w, -w, 0, w, 2*w])
	# 	paramsDicts[ii]['wealthTarget'] = wealthTarget
	# 	paramsDicts[ii]['nx'] = 120
	# 	paramsDicts[ii]['nc'] = 120
	# 	paramsDicts[ii]['nSim'] = 1e5
	# 	paramsDicts[ii]['locIncomeProcess'] = locIncomeProcess
	# 	ii += 1

	# adjustCost = 0.01
	# discount_widths = [	0.00671559019522,
	# 					0.016959141437817,
	# 					0.00959594985356,
	# 					]
	# wealthTarget = 3.2

	# for w in discount_widths:
	# 	paramsDicts.append({})
	# 	paramsDicts[ii]['name'] = f'5-pt discount factor w/width{w}'
	# 	paramsDicts[ii]['index'] = ii
	# 	paramsDicts[ii]['adjustCost'] = adjustCost
	# 	paramsDicts[ii]['noPersIncome'] = False
	# 	paramsDicts[ii]['riskAver'] = 1
	# 	paramsDicts[ii]['discount_factor_grid'] = np.array([-2*w, -w, 0, w, 2*w])
	# 	paramsDicts[ii]['wealthTarget'] = wealthTarget
	# 	paramsDicts[ii]['nx'] = 120
	# 	paramsDicts[ii]['nc'] = 120
	# 	paramsDicts[ii]['nSim'] = 1e5
	# 	paramsDicts[ii]['locIncomeProcess'] = locIncomeProcess
	# 	ii += 1

	# adjustCost = 0.025
	# discount_widths = [	0.018330388490419,
	# 					0.006931535236858,
	# 					]
	# wealthTarget = 3.2

	# for w in discount_widths:
	# 	paramsDicts.append({})
	# 	paramsDicts[ii]['name'] = f'5-pt discount factor w/width{w}'
	# 	paramsDicts[ii]['index'] = ii
	# 	paramsDicts[ii]['adjustCost'] = adjustCost
	# 	paramsDicts[ii]['noPersIncome'] = False
	# 	paramsDicts[ii]['riskAver'] = 1
	# 	paramsDicts[ii]['discount_factor_grid'] = np.array([-2*w, -w, 0, w, 2*w])
	# 	paramsDicts[ii]['wealthTarget'] = wealthTarget
	# 	paramsDicts[ii]['nx'] = 120
	# 	paramsDicts[ii]['nc'] = 120
	# 	paramsDicts[ii]['nSim'] = 1e5
	# 	paramsDicts[ii]['locIncomeProcess'] = locIncomeProcess
	# 	ii += 1

	# adjustCost = 0.05
	# discount_widths = [	0.020700768461677,
	# 					0.020800768461677,
	# 					0.020900768461677,
	# 					]
	# wealthTarget = 3.2

	# for w in discount_widths:
	# 	paramsDicts.append({})
	# 	paramsDicts[ii]['name'] = f'5-pt discount factor w/width{w}'
	# 	paramsDicts[ii]['index'] = ii
	# 	paramsDicts[ii]['adjustCost'] = adjustCost
	# 	paramsDicts[ii]['noPersIncome'] = False
	# 	paramsDicts[ii]['riskAver'] = 1
	# 	paramsDicts[ii]['discount_factor_grid'] = np.array([-2*w, -w, 0, w, 2*w])
	# 	paramsDicts[ii]['wealthTarget'] = wealthTarget
	# 	paramsDicts[ii]['nx'] = 120
	# 	paramsDicts[ii]['nc'] = 120
	# 	paramsDicts[ii]['nSim'] = 1e5
	# 	paramsDicts[ii]['locIncomeProcess'] = locIncomeProcess
	# 	ii += 1

	###########################################################
	##### IES HETEROGENEITY ###################################
	###########################################################

	adjustCost = 0.005
	RA_x = []
	wealthTarget = 0.3

	# for x in RA_x:
	# 	paramsDicts.append({})
	# 	paramsDicts[ii]['name'] = f'2-pt RA het exp(-x), exp(x), x = {x}'
	# 	paramsDicts[ii]['index'] = ii
	# 	paramsDicts[ii]['adjustCost'] = adjustCost
	# 	paramsDicts[ii]['noPersIncome'] = False
	# 	paramsDicts[ii]['riskAver'] = 0
	# 	paramsDicts[ii]['risk_aver_grid'] = np.array([np.exp(-x), np.exp(x)])
	# 	paramsDicts[ii]['wealthTarget'] = wealthTarget
	# 	paramsDicts[ii]['nx'] = 120
	# 	paramsDicts[ii]['nc'] = 120
	# 	paramsDicts[ii]['nSim'] = 1e5
	# 	paramsDicts[ii]['locIncomeProcess'] = locIncomeProcess
	# 	ii += 1

	# paramsDicts.append({})
	# paramsDicts[ii]['name'] = 'fast'
	# paramsDicts[ii]['adjustCost'] = 1
	# paramsDicts[ii]['noPersIncome'] = True
	# paramsDicts[ii]['riskAver'] = 1
	# paramsDicts[ii]['discount_factor_grid'] = np.array([0.0])
	# paramsDicts[ii]['nx'] = 40
	# paramsDicts[ii]['nc'] = 40
	# paramsDicts[ii]['nSim'] = 1e4
	# paramsDicts[ii]['locIncomeProcess'] = locIncomeProcess
	# paramsDicts[ii]['timeDiscount'] = 0.8
	# paramsDicts[ii]['r'] = 0.02
	# paramsDicts[ii]['wealthTarget'] = 0.5
	# ii += 1

	# paramsDicts.append({})
	# paramsDicts[ii]['name'] = 'custom'
	# paramsDicts[ii]['adjustCost'] = 0.001 # 0.005
	# paramsDicts[ii]['riskAver'] = 1
	# paramsDicts[ii]['discount_factor_grid'] = np.array([0.0])
	# paramsDicts[ii]['nx'] = 75
	# paramsDicts[ii]['nc'] = 400
	# paramsDicts[ii]['nSim'] = 1e5
	# paramsDicts[ii]['locIncomeProcess'] = locIncomeProcess
	# paramsDicts[ii]['timeDiscount'] = 0.95
	# paramsDicts[ii]['r'] = 0.02
	# paramsDicts[ii]['wealthTarget'] = 3.5
	# paramsDicts[ii]['minGridSpacing'] = 0
	# paramsDicts[ii]['tSim'] = 100
	# paramsDicts[ii]['deathProb'] = 0
	# paramsDicts[ii]['cMax'] = 25
	# paramsDicts[ii]['xMax'] = 25
	# ii += 1

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