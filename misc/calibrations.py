from model.Params import Params
import numpy as np

def load_replication(replication):
	target = replication['target']
	acost = replication['adjustCostOn']
	betaHet = replication['betaHeterogeneity']

	if target == 'mean_wealth':
		params_out = {
			'xGridTerm1Wt': 0.01,
			'xGridTerm1Curv': 0.8,
			'xGridCurv': 0.2,
			'xMax': 50,
			'borrowLim': 0,
			'cMin': 1e-6,
			'cMax': 5,
			'cGridTerm1Wt': 0.01,
			'cGridTerm1Curv': 0.8,
			'cGridCurv': 0.2,
		}
		meanw = True
	elif target == 'wealth_lt_1000':
		# Use defaults
		params_out = dict()
		meanw = False

		if betaHet:
			msg = 'beta het only calibrated to mean wealth target'
			raise Exception(msg)
	else:
		raise Exception('Invalid entry for target')

	if betaHet:
		if acost:
			adjCostQ = 0.00043307788064615287
			betaQ = 0.9663253019078122
		else:
			adjCostQ = 0
			betaQ = 0.9663253019078122
		params_out['discount_factor_grid'] = np.array(
			[-0.032, 0, 0.032])
	else:
		if acost and meanw:
			adjCostQ = 1.2861226655132633e-05
			betaQ = 0.9960233991324677
		elif acost and (not meanw):
			adjCostQ = 0.000417
			betaQ = 0.967748
		elif (not acost) and meanw:
			adjCostQ = 0
			betaQ = 0.9960233991324677
		elif (not acost) and (not meanw):
			adjCostQ = 0
			betaQ = 0.9649236559422705

	params_out['timeDiscount'] = betaQ ** 4.0
	params_out['adjustCost'] = adjCostQ * 4.0
	params_out['nc'] = 100
	params_out['nx'] = 100

	print('Replication chosen:')
	print(f'\tBeta heterogeneity = {betaHet}')
	print(f'\tAdjustment cost = {acost}')

	if meanw:
		print('\tTarget mean wealth')
	else:
		print('\tTarget fraction of HHs with w < $1000')

	return params_out

def load_calibration(index):
	"""
	This function sets the parameterizations to be passed
	to a new Params object.
	"""

	default_values = Params()

	params = dict()
	params['index'] = index

	adjustCostOn = (index % 2 == 1)
	index = index // 2

	params['nc'] = 150
	params['nx'] = 150

	params['cal1_options'] = dict()
	params['cal2_options'] = dict()

	if index == 0:
		###########################################################
		##### TARGET WEALTH < $1000 ###############################
		###########################################################
		if adjustCostOn:
			beta0 = 0.9649296443810829
			adjustCost = 0.0005779423819103919
			params['cal2_options']['skip'] = True
		else:
			beta0 = 0.96439492
			adjustCost = 0
			params['cal2_options']['skip'] = True

		params['timeDiscount'] = beta0  ** 4.0
		params['discount_factor_grid'] = np.array([0.0])
		params['name'] = 'Wealth constrained target'
		params['adjustCost'] = adjustCost * 4

		params['cal1_options']['run'] = 'wealth constrained'
		params['cal1_options']['x0'] = np.array([beta0])
		params['cal1_options']['step'] = np.array([0.000015])
		params['cal1_options']['lbounds'] = [0.96]
		params['cal1_options']['ubounds'] = [0.97]
		params['cal1_options']['skip'] = True

		params['cal2_options']['run'] = 'adjustCost'
		params['cal2_options']['x0'] = np.array([adjustCost])
		params['cal2_options']['lbounds'] = [0.0001]
		params['cal2_options']['ubounds'] = [0.001]

	elif index == 1:
		###########################################################
		##### TARGET 3.2 MEAN WEALTH ##############################
		###########################################################
		if adjustCostOn:
			beta0 = 0.9960357026100761
			adjustCost = 1.3438911359499214e-05
			params['cal2_options']['skip'] = True
		else:
			beta0 = 0.99603041
			adjustCost = 0
			params['cal2_options']['skip'] = True

		params['adjustCost'] = adjustCost * 4
		params['timeDiscount'] = beta0 ** 4.0
		params['xMax'] = 50
		params['discount_factor_grid'] = np.array([0.0])

		params['xGridTerm1Wt'] = 0.01
		params['xGridTerm1Curv'] = 0.8
		params['xGridCurv'] = 0.2
		params['borrowLim'] = 0

		params['cMin'] = 1e-6
		params['cMax'] = 5
		params['cGridTerm1Wt'] = 0.01
		params['cGridTerm1Curv'] = 0.8
		params['cGridCurv'] = 0.2

		params['name'] = 'Mean wealth target'

		params['cal1_options']['run'] = 'mean wealth'
		params['cal1_options']['x0'] = np.array([beta0])
		params['cal1_options']['lbounds'] = [0.994]
		params['cal1_options']['ubounds'] = [0.998]
		params['cal1_options']['skip'] = True

		params['cal2_options']['run'] = 'adjustCost'
		params['cal2_options']['x0'] = np.array([adjustCost])
		params['cal2_options']['step'] = np.array([5e-7])
		params['cal2_options']['lbounds'] = [0.00001]
		params['cal2_options']['ubounds'] = [0.001]
		
	elif index == 2:
		###########################################################
		##### TARGET 3.2 MEAN WEALTH W/BETA HETEROGENEITY #########
		###########################################################
		if adjustCostOn:
			beta0 = 0.9983180684037633
			spacing = 0.03055988
			adjustCost = 0.0004035847852844719
			params['cal2_options']['skip'] = True
		else:
			beta0 = 0.9983143584037633
			spacing = 0.03055988
			adjustCost = 0
			params['cal2_options']['skip'] = True

		params['adjustCost'] = adjustCost * 4
		params['timeDiscount'] = beta0 ** 4.0
		params['xMax'] = 50
		params['discount_factor_grid'] = np.array([-2 * spacing, -spacing, 0])

		params['xGridTerm1Wt'] = 0.01
		params['xGridTerm1Curv'] = 0.8
		params['xGridCurv'] = 0.2
		params['borrowLim'] = 0

		params['cMin'] = 1e-6
		params['cMax'] = 5
		params['cGridTerm1Wt'] = 0.01
		params['cGridTerm1Curv'] = 0.8
		params['cGridCurv'] = 0.2
		
		params['name'] = 'Beta heterogeneity'
		params['cal1_options']['run'] = 'beta heterogeneity'
		params['cal1_options']['x0'] = np.array([beta0 -2 * spacing, beta0 - spacing, beta0])
		params['cal1_options']['lbounds'] = [0.9983, 0.030]
		params['cal1_options']['ubounds'] = [0.9984, 0.032]
		params['cal1_options']['skip'] = True

		params['cal2_options']['run'] = 'adjustCost'
		params['cal2_options']['x0'] = np.array([adjustCost])
		params['cal2_options']['lbounds'] = [0.0002]
		params['cal2_options']['ubounds'] = [0.0006]


	print(f"Selected parameterization: {params['name']}")

	return params