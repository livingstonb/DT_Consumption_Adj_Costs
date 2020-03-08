import numpy as np

def utilityMat(riskaver, con):
	"""
	Utility function for a 4D array.
	"""

	if riskaver.size == 1:
		if riskaver == 1.0:
			return np.log(con)
		else:
			return np.power(con,1.0-riskaver) / (1.0-riskaver)
	else:
		out = np.zeros(con.shape)
		for i in range(riskaver.size):
			if riskaver[i] == 1.0:
				out[:,:,i,:] = np.log(con[:,:,i,:])
			else:
				out[:,:,i,:] = np.power(con[:,:,i,:],1.0-riskaver[i]) / (1.0-riskaver[i])

		return out

def computeAdjBorrLims(nextShock, ymin, borrowLim, R, nlags):
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

def constructCurvedGrid(lbound, ubound, curv, n):
	grid_out = np.linspace(0, 1, num=n)
	grid_out = grid_out.reshape((n, 1))
	grid_out = grid_out ** (1 / curv)
	grid_out = lbound \
		+ (ubound - lbound) * grid_out

	return grid_out