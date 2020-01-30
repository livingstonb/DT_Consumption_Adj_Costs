import numpy as np

def constraint_transform(val, lbound, ubound):
	z = np.abs(val) / (1 + np.abs(val))
	z = lbound + (ubound - lbound) * z

	return z

def constraint_transform_inv(z, lbound, ubound):
	val = (z - lbound) / (ubound - lbound)
	val = val / (1 - val)

	return val