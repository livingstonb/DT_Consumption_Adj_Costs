from model.modelObjects import Params

def load_specifications(locIncomeProcess, index=None, name=None):
	"""
	This function defines the parameterizations and passes the
	one selected by 'index' or 'name' to a new Params object.
	"""

	if (index is None) and (name is None):
		raise Exception ('At least one specification must be chosen')

	adjustCosts = [1,5,10,50,75]

	numExperiments = len(adjustCosts)
	paramsDicts = [dict() for i in range(numExperiments)]

	ii = 0
	for adjustCost in adjustCosts:
		paramsDicts[ii]['name'] = f'adjustCost{adjustCost}'
		paramsDicts[ii]['index'] = ii
		paramsDicts[ii]['adjustCost'] = adjustCost
		paramsDicts[ii]['noPersIncome'] = False
		paramsDicts[ii]['nx'] = 75
		paramsDicts[ii]['nc'] = 400
		paramsDicts[ii]['nSim'] = 5e4
		paramsDicts[ii]['locIncomeProcess'] = locIncomeProcess
		paramsDicts[ii]['timeDiscount'] = 0.9

		ii += 1

	paramsDicts.append({})
	paramsDicts[ii]['name'] = 'fast'
	paramsDicts[ii]['adjustCost'] = 5
	paramsDicts[ii]['noPersIncome'] = True
	paramsDicts[ii]['nx'] = 40
	paramsDicts[ii]['nc'] = 50
	paramsDicts[ii]['nSim'] = 1e5
	paramsDicts[ii]['locIncomeProcess'] = locIncomeProcess
	paramsDicts[ii]['timeDiscount'] = 0.8

	#-----------------------------------------------------#
	#        CREATE PARAMS OBJECT, DO NOT CHANGE          #
	#-----------------------------------------------------#
	if index is not None:
		chosenParameters = paramsDicts[index]
	else:
		for ii in range(len(paramsDicts)):
			if paramsDicts[ii]['name'] == name:
				chosenParameters = paramsDicts[ii]

	return Params(chosenParameters)