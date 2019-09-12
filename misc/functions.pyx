import numpy as np
cimport numpy as np

cdef np.ndarray utility(double riskaver, np.ndarray con):
	if riskaver == 1:
		return np.log(con)
	else:
		return con ** (1-riskaver) / (1-riskaver)

cdef np.ndarray marginalUtility(double riskaver, np.ndarray con):
	"""
	Returns the first derivative of the utility function
	"""
	u = con ** (- riskaver)
	return u

cdef long searchSortedSingleInput(double[:] grid, double val):
	cdef long n, midpt, index
	cdef double gridVal

	n = np.size(grid)
	midpt = n // 2

	index = midpt
	while (index > 0) and (index < n):
		gridVal = grid[index]

		if val < gridVal:
			if val >= grid[index-1]:
				return index
			else:
				index -= 1
		else:
			index += 1

cpdef interpolateTransitionProbabilities(grid, vals, extrap=False):
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

cpdef interpolateTransitionProbabilities2D(grid, vals, extrap=False):
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

cdef tuple interpolate1D(double[:] grid, double pt):
	cdef:
		int gridIndex
		double gridPt1, gridPt2, proportion2
		list gridIndices, proportions

	# returns the two indices between which to interpolate
	# returns the proportions to place on each of the 2 indices
	gridIndex = searchSortedSingleInput(grid, pt)

	if gridIndex == 0:
		gridIndices = [0,1]
		proportions = [1,0]
	elif gridIndex == np.size(grid):
		gridIndices = [grid.shape[0]-2,grid.shape[0]-1]
		proportions = [0,1]
	else:
		gridIndices = [gridIndex-1,gridIndex]
		gridPt1 = grid[gridIndices[0]]
		gridPt2 = grid[gridIndices[1]]
		proportion2 = (pt - gridPt1) / (gridPt2 - gridPt1)
		proportions = [1-proportion2,proportion2]

	return (gridIndices, proportions)

cpdef tuple goldenSectionSearch(object f, double a, double b, 
	double goldenRatio, double goldenRatioSq, double tol, tuple args):

	cdef double c, d, diff
	cdef double fc, fd

	diff = b - a

	c = a + diff / goldenRatioSq
	d = a + diff / goldenRatio 

	fc = f(c,*args)
	fd = f(d,*args)

	while abs(c - d) > tol:
		if fc > fd:
			b = d
			d = c
			fd = fc
			diff = diff / goldenRatio
			c = a + diff / goldenRatio
			fc = f(c,*args)
		else:
			a = c
			c = d
			fc = fd
			diff = diff / goldenRatio
			d = a + diff / goldenRatio
			fd = f(d,*args)

	if fc > fd:
		return fc, (a + d) / 2
	else:
		return fd, (c + b) / 2