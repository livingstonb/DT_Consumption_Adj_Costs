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