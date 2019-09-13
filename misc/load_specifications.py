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
		paramsDicts[ii]['adjustCost'] = adjustCost
		paramsDicts[ii]['nx'] = 40
		paramsDicts[ii]['nc'] = 200
		paramsDicts[ii]['locIncomeProcess'] = locIncomeProcess
		paramsDicts[ii]['timeDiscount'] = 0.8

		ii += 1

	#-----------------------------------------------------#
	#        CREATE PARAMS OBJECT, DO NOT CHANGE          #
	#-----------------------------------------------------#
	if index is not None:
		chosenParameters = Params(paramsDicts[index])
	else:
		chosenParameters = Params([paramsDict[ii] for ii in numExperiments
							if paramsDict[ii]['name'] == name][0])

	return chosenParameters