import numpy as np

def printVector(vec):
	for value in vec:
		print(f'\t{value}')

def printLine(newLines=[1, 1]):
	preNL = '\n' * newLines[0]
	postNL = '\n' * newLines[1]
	lines = '-' * 50
	print(preNL + lines + postNL)

def utilityMat(riskaver, con):
	"""
	Utility function for an N-D array.
	"""

	# Replace '1' entries to avoid division by zero
	riskaver_adj = np.where(riskaver == 1, 2, riskaver)
	u = np.where(riskaver == 1,
		np.log(con),
		np.power(con, 1.0 - riskaver_adj) / (1.0 - riskaver_adj))

	return u

def computeAdjBorrLims(nextShock, ymin, borrowLim, R, nlags):
	"""
	Returns a list containing the news-adjusted borrowing limits. Only
	applicable for news of a negative shock. The last element of the
	list is the adjusted borrowing limit if the shock occurs nlags in the future.
	The next-to-last is the adjusted borrowing limit for the period in which
	the shock occurs nlags-1 in the future, etc...

	For news of a positive shock, all elements of the list will equal
	borrowLim.
	"""
	borrLims = [borrowLim]
	nextLim = borrowLim - nextShock
	for ii in range(nlags):
		nextLim = (nextLim - ymin) / R
		borrLims.append(nextLim)

	borrLims = [max(blim, borrowLim) for blim in borrLims]
	return borrLims

def replaceNumpyNan(arr, val):
	arr_out = np.where(np.isnan(arr), val,
		arr)
	return arr_out

def constructCurvedGrid(lbound, ubound, curv, n,
	term1_wt=0, term1_curv=1):
	"""
	Constructs a curved grid equal to the weighted
	sum of two power-spaced grids.
	"""
	grid_out = np.linspace(0, 1, num=n)
	grid_out = grid_out.reshape((n, 1))

	term1 = term1_wt * grid_out ** (1 / term1_curv)
	term2 = grid_out ** (1 / curv)

	grid_out = (term1 + term2) / (1 + term1_wt)
	grid_out = lbound \
		+ (ubound - lbound) * grid_out

	return grid_out

def eval_smoothed(x, y, bounds, q):
	keep = (x > bounds[0]) & (x < bounds[1])
	fp = np.polyfit(x[keep], y[keep], 3)

	return np.polyval(fp, q)