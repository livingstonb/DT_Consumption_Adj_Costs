import numpy as np

def parameters():
	defaults = {
		'fastSettings': False,
		'name': 'Unnamed',
		'index': 0,
		'freq': 4,
		'locIncomeProcess': '',
		'nyP': 1,
		'nyT': 1,
		'noTransIncome': False,
		'noPersIncome': False,
		'nz': 1,
		'maxIters': 2e4,
		'tol': 1e-7,
		'nSim': 2e5,
		'tSim': 100,
		'nSectionsGSS': 20,
		'NsimMPC': 2e5,
		'MPCshocks': [-0.081, -0.0405, -0.0081, 0.0081, 0.0405, 0.081, 0],
		'wealthConstraints': [0,0.005,0.01,0.015,0.02,0.05,0.1,0.15],
		'wealthPercentiles': [10,25,50,75,90,99,99.9],
		'xMax': 25,
		'nx': 50,
		'xGridTerm1Wt': 0.01,
		'xGridTerm1Curv': 0.9,
		'xGridCurv': 0.15,
		'borrowLim': 0,
		'nc': 75,
		'cMin': 1e-6,
		'cMax': 5,
		'cGridTerm1Wt': 0.005,
		'cGridTerm1Curv': 0.9,
		'cGridCurv': 0.15,
		'MPCsOutOfNews': False,
		'r': 0.02,
		'R': 1.02,
		'deathProb': 0.02,
		'Bequests': True,
		'riskAver': 1,
		'timeDiscount': 0.8,
		'risk_aver_grid': np.array([0.0], dtype=float),
		'discount_factor_grid': np.array([0.0], dtype=float),
		'govTransfer': 0.0081 * 2.0 * 4.0,
	}

	return defaults