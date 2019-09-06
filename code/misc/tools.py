def marginalUtility(riskaver,con):
	"""
	Returns the first derivative of the utility function
	"""
	u = con ** (- riskaver)
	return u