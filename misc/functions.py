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