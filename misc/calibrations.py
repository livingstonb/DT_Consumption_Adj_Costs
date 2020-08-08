from model.Params import Params
import numpy as np

def load_custom():
	params = dict()

	return params

def load_parameters(index=0, replication=None):
	"""
	This function sets the parameters, to be passed
	to a new Params object.
	"""

	default_values = Params()

	if replication is not None:
		adjustCostOn = replication['adjustCostOn']
		mode = replication['mode']

		if mode == 'wealth_lt_1000':
			index = 0
		elif mode == 'mean_wealth':
			index = 2
		elif mode == 'beta_het':
			index = 4

		index += adjustCostOn
	else:
		adjustCostOn = (index % 2 == 1)

		if index // 2 == 0:
			mode = 'wealth_lt_1000'
		elif index // 2 == 1:
			mode = 'mean_wealth'
		elif index // 2 == 2:
			mode = 'beta_het'

	params = dict()
	params['index'] = index

	params['nc'] = 150
	params['nx'] = 150

	params['cal1_options'] = dict()
	params['cal2_options'] = dict()

	if mode == 'wealth_lt_1000':
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

		# calibration options
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

	elif mode == 'mean_wealth':
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

		# calibration options
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
		
	elif mode == 'beta_het':
		###########################################################
		##### BOTH WEALTH TARGETS W/BETA HETEROGENEITY ############
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

		# calibration options
		params['cal1_options']['run'] = 'beta heterogeneity'
		params['cal1_options']['x0'] = np.array([beta0 -2 * spacing, beta0 - spacing, beta0])
		params['cal1_options']['lbounds'] = [0.9983, 0.030]
		params['cal1_options']['ubounds'] = [0.9984, 0.032]
		params['cal1_options']['skip'] = True

		params['cal2_options']['run'] = 'adjustCost'
		params['cal2_options']['x0'] = np.array([adjustCost])
		params['cal2_options']['lbounds'] = [0.0002]
		params['cal2_options']['ubounds'] = [0.0006]

	if adjustCostOn:
		params['name'] += ' w/adj cost'
	else:
		params['name'] += ' w/o adj cost'
	print(f"Selected parameterization: {params['name']}")

	return params