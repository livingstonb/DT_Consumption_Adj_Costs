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
	discount_widths = [	0.03027467539695,
						0.03127467539695,
						0.03227467539695,
						0.018836652371724,
						0.018936652371724,
						0.019036652371724,
						0.012914663137349,
						0.013014663137349,
						0.013114663137349,
						]
	wealthTarget = 3.2

	for w in discount_widths:
		paramsDicts.append({})
		paramsDicts[ii]['name'] = f'2-pt discount factor w/width{w}'
		paramsDicts[ii]['index'] = ii
		paramsDicts[ii]['cubicValueInterp'] = True
		paramsDicts[ii]['adjustCost'] = adjustCost
		paramsDicts[ii]['noPersIncome'] = False
		paramsDicts[ii]['riskAver'] = 1
		paramsDicts[ii]['discount_factor_grid'] = np.array([-w, 0, w])
		paramsDicts[ii]['wealthTarget'] = wealthTarget
		paramsDicts[ii]['nx'] = 120
		paramsDicts[ii]['nc'] = 120
		paramsDicts[ii]['nSim'] = 1e5
		paramsDicts[ii]['locIncomeProcess'] = locIncomeProcess
		ii += 1

	adjustCost = 0.01
	discount_widths = [	0.031820232841294,
						0.031920232841294,
						0.03220232841294,
						0.007129708739052,
						0.008129708739052,
						0.009129708739052,
						0.011643524203965,
						0.012643524203965,
						0.013643524203965,
						]
	wealthTarget = 3.2

	for w in discount_widths:
		paramsDicts.append({})
		paramsDicts[ii]['name'] = f'2-pt discount factor w/width{w}'
		paramsDicts[ii]['index'] = ii
		paramsDicts[ii]['cubicValueInterp'] = True
		paramsDicts[ii]['adjustCost'] = adjustCost
		paramsDicts[ii]['noPersIncome'] = False
		paramsDicts[ii]['riskAver'] = 1
		paramsDicts[ii]['discount_factor_grid'] = np.array([-w, 0, w])
		paramsDicts[ii]['wealthTarget'] = wealthTarget
		paramsDicts[ii]['nx'] = 120
		paramsDicts[ii]['nc'] = 120
		paramsDicts[ii]['nSim'] = 1e5
		paramsDicts[ii]['locIncomeProcess'] = locIncomeProcess
		ii += 1

	adjustCost = 0.025
	discount_widths = [	0.032186514740509,
						0.033186514740509,
						0.034186514740509,
						0.018339420312185,
						0.019339420312185,
						0.019839420312185,
						0.012934354798077,
						0.013034354798077,
						0.013134354798077,
						]
	wealthTarget = 3.2

	for w in discount_widths:
		paramsDicts.append({})
		paramsDicts[ii]['name'] = f'2-pt discount factor w/width{w}'
		paramsDicts[ii]['index'] = ii
		paramsDicts[ii]['cubicValueInterp'] = True
		paramsDicts[ii]['adjustCost'] = adjustCost
		paramsDicts[ii]['noPersIncome'] = False
		paramsDicts[ii]['riskAver'] = 1
		paramsDicts[ii]['discount_factor_grid'] = np.array([-w, 0, w])
		paramsDicts[ii]['wealthTarget'] = wealthTarget
		paramsDicts[ii]['nx'] = 120
		paramsDicts[ii]['nc'] = 120
		paramsDicts[ii]['nSim'] = 1e5
		paramsDicts[ii]['locIncomeProcess'] = locIncomeProcess
		ii += 1

	adjustCost = 0.05
	discount_widths = [	0.008, 0.01, 0.018, 0.02, 0.03, 0.033, 0.04]
	wealthTarget = 3.2

	for w in discount_widths:
		paramsDicts.append({})
		paramsDicts[ii]['name'] = f'2-pt discount factor w/width{w}'
		paramsDicts[ii]['index'] = ii
		paramsDicts[ii]['cubicValueInterp'] = True
		paramsDicts[ii]['adjustCost'] = adjustCost
		paramsDicts[ii]['noPersIncome'] = False
		paramsDicts[ii]['riskAver'] = 1
		paramsDicts[ii]['discount_factor_grid'] = np.array([-w, 0, w])
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
	RA_x = []
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