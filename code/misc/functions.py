import numpy as np

def utility(riskaver,con):
	if riskaver == 1:
		return np.log(con)
	else:
		return con ** (1-riskaver) / (1-riskaver)

def marginalUtility(riskaver,con):
	"""
	Returns the first derivative of the utility function
	"""
	u = con ** (- riskaver)
	return u

def interpolateTransitionProbabilities(grid, vals, extrap=False):
	gridIndices = np.searchsorted(grid,vals)
	probabilities = np.zeros((vals.shape[0],grid.shape[0]))

	for ival in range(vals.shape[0]):
		igrid = gridIndices[ival]
		gridPt1 = grid[igrid-1]
		gridPt2 = grid[igrid]
		Pi = (vals[ival] - gridPt1) / (gridPt2 - gridPt1)

		if extrap or ((Pi >= 0) and (Pi <= 1)):
			probabilities[ival,igrid] = Pi
			probabilities[ival,igrid-1] = 1 - Pi
		elif Pi > 1:
			probabilities[ival,igrid] = 1
		elif Pi < 0:
			probabilities[ival,igrid-1] = 1
		else:
			raise Exception ('Invalid value for Pi')

	return probabilities