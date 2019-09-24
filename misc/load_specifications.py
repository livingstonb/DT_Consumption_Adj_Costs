from model.modelObjects import Params

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
					+ ' riskAver{riskAver}, wealth{wealthTarget}'
				paramsDicts[ii]['index'] = ii
				paramsDicts[ii]['cubicEMAXInterp'] = False
				paramsDicts[ii]['cubicValueInterp'] = True
				paramsDicts[ii]['adjustCost'] = adjustCost
				paramsDicts[ii]['noPersIncome'] = False
				paramsDicts[ii]['riskAver'] = riskAver
				paramsDicts[ii]['wealthTargets'] = wealthTarget
				paramsDicts[ii]['nx'] = 200
				paramsDicts[ii]['nc'] = 200
				paramsDicts[ii]['nSim'] = 1e5
				paramsDicts[ii]['locIncomeProcess'] = locIncomeProcess
				paramsDicts[ii]['timeDiscount'] = timeDiscount

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
			paramsDicts[ii]['cubicEMAXInterp'] = False
			paramsDicts[ii]['cubicValueInterp'] = True
			paramsDicts[ii]['adjustCost'] = adjustCost
			paramsDicts[ii]['noPersIncome'] = False
			paramsDicts[ii]['riskAver'] = 1
			paramsDicts[ii]['risk_aver_grids'] = np.array(RAgrid)
			paramsDicts[ii]['wealthTargets'] = wealthTarget
			paramsDicts[ii]['nx'] = 200
			paramsDicts[ii]['nc'] = 200
			paramsDicts[ii]['nSim'] = 1e5
			paramsDicts[ii]['locIncomeProcess'] = locIncomeProcess
			paramsDicts[ii]['timeDiscount'] = timeDiscount

			ilabel += 1

	adjustCosts = [0.1,0.5,1,2,5]
	discountGrids = [[-0.02,0.02],[-0.05,0.05],[-0.1,0.1]]
	wealthTarget = 3.2

	ilabel = 0
	for adjustCost in adjustCosts:
		for discountGrid in discountGrids:
			paramsDicts.append({})
			paramsDicts[ii]['name'] = f'Discount Factor Heterogeneity {ilabel}'
			paramsDicts[ii]['index'] = ii
			paramsDicts[ii]['cubicEMAXInterp'] = False
			paramsDicts[ii]['cubicValueInterp'] = True
			paramsDicts[ii]['adjustCost'] = adjustCost
			paramsDicts[ii]['noPersIncome'] = False
			paramsDicts[ii]['riskAver'] = 1
			paramsDicts[ii]['discount_factor_grid'] = np.array(discountGrid)
			paramsDicts[ii]['wealthTargets'] = wealthTarget
			paramsDicts[ii]['nx'] = 200
			paramsDicts[ii]['nc'] = 200
			paramsDicts[ii]['nSim'] = 1e5
			paramsDicts[ii]['locIncomeProcess'] = locIncomeProcess
			paramsDicts[ii]['timeDiscount'] = timeDiscount

			ilabel += 1

	paramsDicts.append({})
	paramsDicts[ii]['name'] = 'fast'
	paramsDicts[ii]['adjustCost'] = 5
	paramsDicts[ii]['noPersIncome'] = True
	paramsDicts[ii]['nx'] = 40
	paramsDicts[ii]['nc'] = 50
	paramsDicts[ii]['nSim'] = 1e4
	paramsDicts[ii]['locIncomeProcess'] = locIncomeProcess
	paramsDicts[ii]['timeDiscount'] = 0.8

	ii += 1
	paramsDicts.append({})
	paramsDicts[ii]['name'] = 'custom'
	paramsDicts[ii]['cubicEMAXInterp'] = False
	paramsDicts[ii]['cubicValueInterp'] = True
	paramsDicts[ii]['adjustCost'] = 0.05
	paramsDicts[ii]['noPersIncome'] = False
	paramsDicts[ii]['riskAver'] = 2
	paramsDicts[ii]['nx'] = 200
	paramsDicts[ii]['nc'] = 200
	paramsDicts[ii]['nSim'] = 1e5
	paramsDicts[ii]['locIncomeProcess'] = locIncomeProcess
	paramsDicts[ii]['timeDiscount'] = 0.99

	#-----------------------------------------------------#
	#        CREATE PARAMS OBJECT, DO NOT CHANGE          #
	#-----------------------------------------------------#
	if index is not None:
		chosenParameters = paramsDicts[index]
		print(f'Selected parameterization {index}')
	else:
		for ii in range(len(paramsDicts)):
			if paramsDicts[ii]['name'] == name:
				chosenParameters = paramsDicts[ii]
				print(f'Selected parameterization {ii}')

	return Params(chosenParameters)