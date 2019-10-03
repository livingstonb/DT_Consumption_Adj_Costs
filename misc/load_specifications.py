from model.modelObjects import Params
import numpy as np

def load_specifications(locIncomeProcess, index=None, name=None):
	"""
	This function defines the parameterizations and passes the
	one selected by 'index' or 'name' to a new Params object.
	"""

	if (index is None) and (name is None):
		raise Exception ('At least one specification must be chosen')

	adjustCosts = [0.1,0.5,1,2,5]
	riskAvers = [0.5,1,2,4]
	wealthTargets = [0.3,3.2,5.4]

	paramsDicts = []

	ii = 0
	for adjustCost in adjustCosts:
		for riskAver in riskAvers:
			for wealthTarget in wealthTargets:
				paramsDicts.append({})
				paramsDicts[ii]['name'] = f'adjustCost{adjustCost},' \
					+ f' riskAver{riskAver}, wealth{wealthTarget}'
				paramsDicts[ii]['index'] = ii
				paramsDicts[ii]['cubicValueInterp'] = True
				paramsDicts[ii]['adjustCost'] = adjustCost
				paramsDicts[ii]['noPersIncome'] = False
				paramsDicts[ii]['riskAver'] = riskAver
				paramsDicts[ii]['wealthTarget'] = wealthTarget
				paramsDicts[ii]['nx'] = 150
				paramsDicts[ii]['nc'] = 150
				paramsDicts[ii]['nSim'] = 1e5
				paramsDicts[ii]['locIncomeProcess'] = locIncomeProcess

				ii += 1

	adjustCosts = [0.1,0.5,1,2,5]
	RAgrids = [[-0.25,0,0.25],[-0.5,0,0.5]]
	wealthTarget = 3.2

	ilabel = 0
	for adjustCost in adjustCosts:
		for RAgrid in RAgrids:
			paramsDicts.append({})
			paramsDicts[ii]['name'] = f'IES Heterogeneity {ilabel}'
			paramsDicts[ii]['index'] = ii
			paramsDicts[ii]['cubicValueInterp'] = True
			paramsDicts[ii]['adjustCost'] = adjustCost
			paramsDicts[ii]['noPersIncome'] = False
			paramsDicts[ii]['riskAver'] = 1
			paramsDicts[ii]['risk_aver_grid'] = np.array(RAgrid)
			paramsDicts[ii]['wealthTarget'] = wealthTarget
			paramsDicts[ii]['nx'] = 150
			paramsDicts[ii]['nc'] = 150
			paramsDicts[ii]['nSim'] = 1e5
			paramsDicts[ii]['locIncomeProcess'] = locIncomeProcess

			ii += 1
			ilabel += 1

	adjustCosts = [0.1,0.5,1,2,5]
	discountGrids = [[-0.01,0.01],[-0.025,0.025],[-0.05,0.05]]
	discount_width = [0.02,0.05,0.1]
	wealthTarget = 3.2

	for adjustCost in adjustCosts:
		i_discount_grid = 0
		for discountGrid in discountGrids:
			paramsDicts.append({})
			paramsDicts[ii]['name'] = f'Discount Factor Heterogeneity, beta width{discount_width[i_discount_grid]}'
			paramsDicts[ii]['index'] = ii
			paramsDicts[ii]['cubicValueInterp'] = True
			paramsDicts[ii]['adjustCost'] = adjustCost
			paramsDicts[ii]['noPersIncome'] = False
			paramsDicts[ii]['riskAver'] = 1
			paramsDicts[ii]['discount_factor_grid'] = np.array(discountGrid)
			paramsDicts[ii]['wealthTarget'] = wealthTarget
			paramsDicts[ii]['nx'] = 150
			paramsDicts[ii]['nc'] = 150
			paramsDicts[ii]['nSim'] = 1e5
			paramsDicts[ii]['locIncomeProcess'] = locIncomeProcess
			i_discount_grid += 1

			ii += 1
			ilabel += 1

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
	paramsDicts[ii]['nx'] = 150
	paramsDicts[ii]['nc'] = 150
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