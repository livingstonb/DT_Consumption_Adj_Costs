from model.Params import Params
import numpy as np

def load_replication(replication):
	target = replication['target']
	acost = replication['adjustCostOn']
	betaHet = replication['betaHeterogeneity']

	if target == 'mean_wealth':
		params_out = {
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
			adjCostQ = 0.00036446248457166374
			betaQ = 0.9649236559422705
		elif (not acost) and meanw:
			adjCostQ = 0
			betaQ = 0.9960233991324677
		elif (not acost) and (not meanw):
			adjCostQ = 0
			betaQ = 0.9649236559422705

	params_out['timeDiscount'] = betaQ ** 4.0
	params_out['adjustCost'] = adjCostQ * 4.0
	params_out['nc'] = 200
	params_out['nx'] = 200

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

	gridSize = index % 4
	index = index // 4

	if gridSize == 0:
		params['nc'] = 100
		params['nx'] = 100
	elif gridSize == 1:
		params['nc'] = 100
		params['nx'] = 200
	elif gridSize == 2:
		params['nc'] = 200
		params['nx'] = 100
	elif gridSize == 3:
		params['nc'] = 200
		params['nx'] = 200
	else:
		raise Exception('Invalid entry')

	params['adjustCost'] = 0
	params['cal_options'] = dict()

	if index == 0:
		###########################################################
		##### TARGET WEALTH < $1000 ###############################
		###########################################################
		params['timeDiscount'] = 0.9649236559422705  ** 4.0
		params['discount_factor_grid'] = np.array([0.0])
		params['cal_options']['run'] = 'wealth constrained'
		params['name'] = 'Wealth constrained target w/o adj costs'

	elif index == 1:
		###########################################################
		##### TARGET 3.2 MEAN WEALTH ##############################
		###########################################################
		params['timeDiscount'] = 0.9960233991324677 ** 4.0
		params['xMax'] = 50
		params['discount_factor_grid'] = np.array([0.0])

		params['xGridTerm1Wt'] = 0.05
		params['xGridTerm1Curv'] = 0.8
		params['xGridCurv'] = 0.2
		params['borrowLim'] = 0

		params['cMin'] = 1e-6
		params['cMax'] = 5
		params['cGridTerm1Wt'] = 0.05
		params['cGridTerm1Curv'] = 0.9
		params['cGridCurv'] = 0.15

		# Without adjustment costs
		params['name'] = 'Mean wealth target w/o adj costs'
		params['cal_options']['run'] = 'mean wealth'

	elif index == 2:
		###########################################################
		##### TARGET 3.2 MEAN WEALTH W/BETA HETEROGENEITY #########
		###########################################################
		params['timeDiscount'] = 0.9663253019078122 ** 4.0
		params['xMax'] = 50
		params['discount_factor_grid'] = np.array([-0.032, 0, 0.032])

		params['xGridTerm1Wt'] = 0.05
		params['xGridTerm1Curv'] = 0.8
		params['xGridCurv'] = 0.2
		params['borrowLim'] = 0

		params['cMin'] = 1e-6
		params['cMax'] = 5
		params['cGridTerm1Wt'] = 0.05
		params['cGridTerm1Curv'] = 0.9
		params['cGridCurv'] = 0.15
		
		# Without adjustment costs
		params['name'] = 'Beta heterogeneity w/o adj costs'
		params['cal_options']['run'] = 'beta heterogeneity'

	print(f"Selected parameterization: {params['name']}")

	return params