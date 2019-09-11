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

def interpolateTransitionProbabilities2D(grid, vals, extrap=False):
	gridIndices = np.searchsorted(grid,vals)
	probabilities = np.zeros((vals.shape[0],vals.shape[1],grid.shape[0]))

	nGrid = grid.shape[0]

	for i in range(vals.shape[0]):
		for j in range(vals.shape[1]):
			igrid = gridIndices[i,j]
			if igrid == 0:
				probabilities[i,j,0] = 1
			elif igrid == nGrid:
				probabilities[i,j,-1] = 1
			else:
				gridPt1 = grid[igrid-1]
				gridPt2 = grid[igrid]
				Pi = (vals[i,j] - gridPt1) / (gridPt2 - gridPt1)

				if extrap or ((Pi >= 0) and (Pi <= 1)):
					probabilities[i,j,igrid] = Pi
					probabilities[i,j,igrid-1] = 1 - Pi
				elif Pi > 1:
					probabilities[i,j,igrid] = 1
				elif Pi < 0:
					probabilities[i,j,igrid-1] = 1
				else:
					raise Exception ('Invalid value for Pi')

	return probabilities

def interpolate1D(grid, pt, extrap=False):
	# returns the two indices between which to interpolate
	# returns the proportions to place on each of the 2 indices
	gridIndex = np.searchsorted(grid, pt)

	if gridIndex == 0:
		gridIndices = [0,1]
		proportions = np.array([1,0])
	elif gridIndex == grid.shape[0]:
		gridIndices = [grid.shape[0]-2,grid.shape[0]-1]
		proportions = np.array([0,1])
	else:
		gridIndices = [gridIndex-1,gridIndex]
		gridPt1 = grid[gridIndices[0]]
		gridPt2 = grid[gridIndices[1]]
		proportion2 = (pt - gridPt1) / (gridPt2 - gridPt1)
		proportions = np.array([1-proportion2,proportion2])

	return gridIndices, proportions

