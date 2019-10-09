from model.modelObjects import Params
import numpy as np

def load_specifications(locIncomeProcess, index=None, name=None):
	"""
	This function defines the parameterizations and passes the
	one selected by 'index' or 'name' to a new Params object.
	"""

	if (index is None) and (name is None):
		raise Exception ('At least one specification must be chosen')

	# adjustCosts = [0.005,0.01,0.025,0.05]
	# riskAvers = [1]
	# wealthTargets = [0.3,3.2]

	paramsDicts = []
	ii = 0

	# ii = 0
	# for adjustCost in adjustCosts:
	# 	for riskAver in riskAvers:
	# 		for wealthTarget in wealthTargets:
	# 			paramsDicts.append({})
	# 			paramsDicts[ii]['name'] = f'adjustCost{adjustCost},' \
	# 				+ f' riskAver{riskAver}, wealth{wealthTarget}'
	# 			paramsDicts[ii]['index'] = ii
	# 			paramsDicts[ii]['cubicValueInterp'] = True
	# 			paramsDicts[ii]['adjustCost'] = adjustCost
	# 			paramsDicts[ii]['noPersIncome'] = False
	# 			paramsDicts[ii]['riskAver'] = riskAver
	# 			paramsDicts[ii]['wealthTarget'] = wealthTarget
	# 			paramsDicts[ii]['nx'] = 120
	# 			paramsDicts[ii]['nc'] = 120
	# 			paramsDicts[ii]['nSim'] = 1e5
	# 			paramsDicts[ii]['locIncomeProcess'] = locIncomeProcess

	# 			ii += 1

	###########################################################
	##### DISCOUNT FACTOR HETEROGENEITY #######################
	###########################################################

	adjustCost = 0.005
	discount_widths = []
	wealthTarget = 0.3

	for w in discount_widths:
		paramsDicts.append({})
		paramsDicts[ii]['name'] = f'2-pt discount factor w/width{w}'
		paramsDicts[ii]['index'] = ii
		paramsDicts[ii]['cubicValueInterp'] = True
		paramsDicts[ii]['adjustCost'] = adjustCost
		paramsDicts[ii]['noPersIncome'] = False
		paramsDicts[ii]['riskAver'] = 1
		paramsDicts[ii]['discount_factor_grid'] = np.array([-w/2, w/2])
		paramsDicts[ii]['wealthTarget'] = wealthTarget
		paramsDicts[ii]['nx'] = 120
		paramsDicts[ii]['nc'] = 120
		paramsDicts[ii]['nSim'] = 1e5
		paramsDicts[ii]['locIncomeProcess'] = locIncomeProcess
		ii += 1

	adjustCost = 0.005
	discount_widths = [	0.057399420832,
						0.032600525522637]
	wealthTarget = 3.2

	for w in discount_widths:
		paramsDicts.append({})
		paramsDicts[ii]['name'] = f'2-pt discount factor w/width{w}'
		paramsDicts[ii]['index'] = ii
		paramsDicts[ii]['cubicValueInterp'] = True
		paramsDicts[ii]['adjustCost'] = adjustCost
		paramsDicts[ii]['noPersIncome'] = False
		paramsDicts[ii]['riskAver'] = 1
		paramsDicts[ii]['discount_factor_grid'] = np.array([-w/2, w/2])
		paramsDicts[ii]['wealthTarget'] = wealthTarget
		paramsDicts[ii]['nx'] = 120
		paramsDicts[ii]['nc'] = 120
		paramsDicts[ii]['nSim'] = 1e5
		paramsDicts[ii]['locIncomeProcess'] = locIncomeProcess
		ii += 1

	adjustCost = 0.01
	discount_widths = [0.042131885360252]
	wealthTarget = 0.3

	for w in discount_widths:
		paramsDicts.append({})
		paramsDicts[ii]['name'] = f'2-pt discount factor w/width{w}'
		paramsDicts[ii]['index'] = ii
		paramsDicts[ii]['cubicValueInterp'] = True
		paramsDicts[ii]['adjustCost'] = adjustCost
		paramsDicts[ii]['noPersIncome'] = False
		paramsDicts[ii]['riskAver'] = 1
		paramsDicts[ii]['discount_factor_grid'] = np.array([-w/2, w/2])
		paramsDicts[ii]['wealthTarget'] = wealthTarget
		paramsDicts[ii]['nx'] = 120
		paramsDicts[ii]['nc'] = 120
		paramsDicts[ii]['nSim'] = 1e5
		paramsDicts[ii]['locIncomeProcess'] = locIncomeProcess
		ii += 1

	adjustCost = 0.01
	discount_widths = [	0.059884813105041,
						0.031081732131458]
	wealthTarget = 3.2

	for w in discount_widths:
		paramsDicts.append({})
		paramsDicts[ii]['name'] = f'2-pt discount factor w/width{w}'
		paramsDicts[ii]['index'] = ii
		paramsDicts[ii]['cubicValueInterp'] = True
		paramsDicts[ii]['adjustCost'] = adjustCost
		paramsDicts[ii]['noPersIncome'] = False
		paramsDicts[ii]['riskAver'] = 1
		paramsDicts[ii]['discount_factor_grid'] = np.array([-w/2, w/2])
		paramsDicts[ii]['wealthTarget'] = wealthTarget
		paramsDicts[ii]['nx'] = 120
		paramsDicts[ii]['nc'] = 120
		paramsDicts[ii]['nSim'] = 1e5
		paramsDicts[ii]['locIncomeProcess'] = locIncomeProcess
		ii += 1

	adjustCost = 0.025
	discount_widths = []
	wealthTarget = 0.3

	for w in discount_widths:
		paramsDicts.append({})
		paramsDicts[ii]['name'] = f'2-pt discount factor w/width{w}'
		paramsDicts[ii]['index'] = ii
		paramsDicts[ii]['cubicValueInterp'] = True
		paramsDicts[ii]['adjustCost'] = adjustCost
		paramsDicts[ii]['noPersIncome'] = False
		paramsDicts[ii]['riskAver'] = 1
		paramsDicts[ii]['discount_factor_grid'] = np.array([-w/2, w/2])
		paramsDicts[ii]['wealthTarget'] = wealthTarget
		paramsDicts[ii]['nx'] = 120
		paramsDicts[ii]['nc'] = 120
		paramsDicts[ii]['nSim'] = 1e5
		paramsDicts[ii]['locIncomeProcess'] = locIncomeProcess
		ii += 1

	adjustCost = 0.025
	discount_widths = [	0.067372178620871]
	wealthTarget = 3.2

	for w in discount_widths:
		paramsDicts.append({})
		paramsDicts[ii]['name'] = f'2-pt discount factor w/width{w}'
		paramsDicts[ii]['index'] = ii
		paramsDicts[ii]['cubicValueInterp'] = True
		paramsDicts[ii]['adjustCost'] = adjustCost
		paramsDicts[ii]['noPersIncome'] = False
		paramsDicts[ii]['riskAver'] = 1
		paramsDicts[ii]['discount_factor_grid'] = np.array([-w/2, w/2])
		paramsDicts[ii]['wealthTarget'] = wealthTarget
		paramsDicts[ii]['nx'] = 120
		paramsDicts[ii]['nc'] = 120
		paramsDicts[ii]['nSim'] = 1e5
		paramsDicts[ii]['locIncomeProcess'] = locIncomeProcess
		ii += 1

	adjustCost = 0.05
	discount_widths = [	0.056990375732197]
	wealthTarget = 0.3

	for w in discount_widths:
		paramsDicts.append({})
		paramsDicts[ii]['name'] = f'2-pt discount factor w/width{w}'
		paramsDicts[ii]['index'] = ii
		paramsDicts[ii]['cubicValueInterp'] = True
		paramsDicts[ii]['adjustCost'] = adjustCost
		paramsDicts[ii]['noPersIncome'] = False
		paramsDicts[ii]['riskAver'] = 1
		paramsDicts[ii]['discount_factor_grid'] = np.array([-w/2, w/2])
		paramsDicts[ii]['wealthTarget'] = wealthTarget
		paramsDicts[ii]['nx'] = 120
		paramsDicts[ii]['nc'] = 120
		paramsDicts[ii]['nSim'] = 1e5
		paramsDicts[ii]['locIncomeProcess'] = locIncomeProcess
		ii += 1

	adjustCost = 0.05
	discount_widths = [	0.083319959669837]
	wealthTarget = 3.2

	for w in discount_widths:
		paramsDicts.append({})
		paramsDicts[ii]['name'] = f'2-pt discount factor w/width{w}'
		paramsDicts[ii]['index'] = ii
		paramsDicts[ii]['cubicValueInterp'] = True
		paramsDicts[ii]['adjustCost'] = adjustCost
		paramsDicts[ii]['noPersIncome'] = False
		paramsDicts[ii]['riskAver'] = 1
		paramsDicts[ii]['discount_factor_grid'] = np.array([-w/2, w/2])
		paramsDicts[ii]['wealthTarget'] = wealthTarget
		paramsDicts[ii]['nx'] = 120
		paramsDicts[ii]['nc'] = 120
		paramsDicts[ii]['nSim'] = 1e5
		paramsDicts[ii]['locIncomeProcess'] = locIncomeProcess
		ii += 1

	###########################################################
	##### IES HETEROGENEITY ###################################
	###########################################################

	adjustCost = 0.005
	RA_x = [0.592073575726983]
	wealthTarget = 0.3

	for x in RA_x:
		paramsDicts.append({})
		paramsDicts[ii]['name'] = f'2-pt RA het exp(-x), exp(x), x = {x}'
		paramsDicts[ii]['index'] = ii
		paramsDicts[ii]['cubicValueInterp'] = True
		paramsDicts[ii]['adjustCost'] = adjustCost
		paramsDicts[ii]['noPersIncome'] = False
		paramsDicts[ii]['riskAver'] = 0
		paramsDicts[ii]['risk_aver_grid'] = np.array([np.exp(-x), np.exp(x)])
		paramsDicts[ii]['wealthTarget'] = wealthTarget
		paramsDicts[ii]['nx'] = 120
		paramsDicts[ii]['nc'] = 120
		paramsDicts[ii]['nSim'] = 1e5
		paramsDicts[ii]['locIncomeProcess'] = locIncomeProcess
		ii += 1

	adjustCost = 0.005
	RA_x = []
	wealthTarget = 3.2

	for x in RA_x:
		paramsDicts.append({})
		paramsDicts[ii]['name'] = f'2-pt RA het exp(-x), exp(x), x = {x}'
		paramsDicts[ii]['index'] = ii
		paramsDicts[ii]['cubicValueInterp'] = True
		paramsDicts[ii]['adjustCost'] = adjustCost
		paramsDicts[ii]['noPersIncome'] = False
		paramsDicts[ii]['riskAver'] = 0
		paramsDicts[ii]['risk_aver_grid'] = np.array([np.exp(-x), np.exp(x)])
		paramsDicts[ii]['wealthTarget'] = wealthTarget
		paramsDicts[ii]['nx'] = 120
		paramsDicts[ii]['nc'] = 120
		paramsDicts[ii]['nSim'] = 1e5
		paramsDicts[ii]['locIncomeProcess'] = locIncomeProcess
		ii += 1

	adjustCost = 0.01
	RA_x = [0.684608230252572]
	wealthTarget = 0.3

	for x in RA_x:
		paramsDicts.append({})
		paramsDicts[ii]['name'] = f'2-pt RA het exp(-x), exp(x), x = {x}'
		paramsDicts[ii]['index'] = ii
		paramsDicts[ii]['cubicValueInterp'] = True
		paramsDicts[ii]['adjustCost'] = adjustCost
		paramsDicts[ii]['noPersIncome'] = False
		paramsDicts[ii]['riskAver'] = 0
		paramsDicts[ii]['risk_aver_grid'] = np.array([np.exp(-x), np.exp(x)])
		paramsDicts[ii]['wealthTarget'] = wealthTarget
		paramsDicts[ii]['nx'] = 120
		paramsDicts[ii]['nc'] = 120
		paramsDicts[ii]['nSim'] = 1e5
		paramsDicts[ii]['locIncomeProcess'] = locIncomeProcess
		ii += 1

	adjustCost = 0.01
	RA_x = []
	wealthTarget = 3.2

	for x in RA_x:
		paramsDicts.append({})
		paramsDicts[ii]['name'] = f'2-pt RA het exp(-x), exp(x), x = {x}'
		paramsDicts[ii]['index'] = ii
		paramsDicts[ii]['cubicValueInterp'] = True
		paramsDicts[ii]['adjustCost'] = adjustCost
		paramsDicts[ii]['noPersIncome'] = False
		paramsDicts[ii]['riskAver'] = 0
		paramsDicts[ii]['risk_aver_grid'] = np.array([np.exp(-x), np.exp(x)])
		paramsDicts[ii]['wealthTarget'] = wealthTarget
		paramsDicts[ii]['nx'] = 120
		paramsDicts[ii]['nc'] = 120
		paramsDicts[ii]['nSim'] = 1e5
		paramsDicts[ii]['locIncomeProcess'] = locIncomeProcess
		ii += 1

	adjustCost = 0.025
	RA_x = [0.994811124586202]
	wealthTarget = 0.3

	for x in RA_x:
		paramsDicts.append({})
		paramsDicts[ii]['name'] = f'2-pt RA het exp(-x), exp(x), x = {x}'
		paramsDicts[ii]['index'] = ii
		paramsDicts[ii]['cubicValueInterp'] = True
		paramsDicts[ii]['adjustCost'] = adjustCost
		paramsDicts[ii]['noPersIncome'] = False
		paramsDicts[ii]['riskAver'] = 0
		paramsDicts[ii]['risk_aver_grid'] = np.array([np.exp(-x), np.exp(x)])
		paramsDicts[ii]['wealthTarget'] = wealthTarget
		paramsDicts[ii]['nx'] = 120
		paramsDicts[ii]['nc'] = 120
		paramsDicts[ii]['nSim'] = 1e5
		paramsDicts[ii]['locIncomeProcess'] = locIncomeProcess
		ii += 1

	adjustCost = 0.025
	RA_x = []
	wealthTarget = 3.2

	for x in RA_x:
		paramsDicts.append({})
		paramsDicts[ii]['name'] = f'2-pt RA het exp(-x), exp(x), x = {x}'
		paramsDicts[ii]['index'] = ii
		paramsDicts[ii]['cubicValueInterp'] = True
		paramsDicts[ii]['adjustCost'] = adjustCost
		paramsDicts[ii]['noPersIncome'] = False
		paramsDicts[ii]['riskAver'] = 0
		paramsDicts[ii]['risk_aver_grid'] = np.array([np.exp(-x), np.exp(x)])
		paramsDicts[ii]['wealthTarget'] = wealthTarget
		paramsDicts[ii]['nx'] = 120
		paramsDicts[ii]['nc'] = 120
		paramsDicts[ii]['nSim'] = 1e5
		paramsDicts[ii]['locIncomeProcess'] = locIncomeProcess
		ii += 1

	adjustCost = 0.05
	RA_x = [1.12552051634514]
	wealthTarget = 0.3

	for x in RA_x:
		paramsDicts.append({})
		paramsDicts[ii]['name'] = f'2-pt RA het exp(-x), exp(x), x = {x}'
		paramsDicts[ii]['index'] = ii
		paramsDicts[ii]['cubicValueInterp'] = True
		paramsDicts[ii]['adjustCost'] = adjustCost
		paramsDicts[ii]['noPersIncome'] = False
		paramsDicts[ii]['riskAver'] = 0
		paramsDicts[ii]['risk_aver_grid'] = np.array([np.exp(-x), np.exp(x)])
		paramsDicts[ii]['wealthTarget'] = wealthTarget
		paramsDicts[ii]['nx'] = 120
		paramsDicts[ii]['nc'] = 120
		paramsDicts[ii]['nSim'] = 1e5
		paramsDicts[ii]['locIncomeProcess'] = locIncomeProcess
		ii += 1

	adjustCost = 0.05
	RA_x = []
	wealthTarget = 3.2

	for x in RA_x:
		paramsDicts.append({})
		paramsDicts[ii]['name'] = f'2-pt RA het exp(-x), exp(x), x = {x}'
		paramsDicts[ii]['index'] = ii
		paramsDicts[ii]['cubicValueInterp'] = True
		paramsDicts[ii]['adjustCost'] = adjustCost
		paramsDicts[ii]['noPersIncome'] = False
		paramsDicts[ii]['riskAver'] = 0
		paramsDicts[ii]['risk_aver_grid'] = np.array([np.exp(-x), np.exp(x)])
		paramsDicts[ii]['wealthTarget'] = wealthTarget
		paramsDicts[ii]['nx'] = 120
		paramsDicts[ii]['nc'] = 120
		paramsDicts[ii]['nSim'] = 1e5
		paramsDicts[ii]['locIncomeProcess'] = locIncomeProcess
		ii += 1

	import pdb; pdb.set_trace()

	paramsDicts.append({})
	paramsDicts[ii]['name'] = 'fast'
	paramsDicts[ii]['cubicValueInterp'] = True
	paramsDicts[ii]['adjustCost'] = 1
	paramsDicts[ii]['noPersIncome'] = True
	paramsDicts[ii]['riskAver'] = 1
	paramsDicts[ii]['discount_factor_grid'] = np.array([0.0])
	paramsDicts[ii]['nx'] = 120
	paramsDicts[ii]['nc'] = 120
	paramsDicts[ii]['nSim'] = 4e4
	paramsDicts[ii]['locIncomeProcess'] = locIncomeProcess
	paramsDicts[ii]['timeDiscount'] = 0.8
	paramsDicts[ii]['r'] = 0.02
	paramsDicts[ii]['wealthTarget'] = 0.5
	ii += 1

	paramsDicts.append({})
	paramsDicts[ii]['name'] = 'custom'
	paramsDicts[ii]['cubicValueInterp'] = True
	paramsDicts[ii]['adjustCost'] = 0.01
	paramsDicts[ii]['noPersIncome'] = False
	paramsDicts[ii]['riskAver'] = 1
	paramsDicts[ii]['discount_factor_grid'] = np.array([0.0])
	paramsDicts[ii]['nx'] = 120
	paramsDicts[ii]['nc'] = 120
	paramsDicts[ii]['nSim'] = 1e5
	paramsDicts[ii]['locIncomeProcess'] = locIncomeProcess
	paramsDicts[ii]['timeDiscount'] = 0.9
	paramsDicts[ii]['r'] = 0.02
	paramsDicts[ii]['wealthTarget'] = 0.5
	ii += 1

	#-----------------------------------------------------#
	#        CREATE PARAMS OBJECT, DO NOT CHANGE          #
	#-----------------------------------------------------#
	if index is not None:
		chosenParameters = paramsDicts[index]
		print(f'Selected parameterization #{index} out of {ii}')
	else:
		for ii in range(len(paramsDicts)):
			if paramsDicts[ii]['name'] == name:
				chosenParameters = paramsDicts[ii]
				print(f'Selected parameterization {ii}')

	return Params(chosenParameters)