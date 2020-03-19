from matplotlib import pyplot as plt
import numpy as np

def polyfit_fraction_below(v_raw, v, cdf_v, q):

	pct_q = np.mean(np.asarray(v_raw) <= q)
	[lb, ub] = poly_interval_transform(pct_q)

	v_lb = np.percentile(v_raw, lb * 100)
	v_ub = np.percentile(v_raw, ub * 100)

	keep = (v >= v_lb) & (v <= v_ub)
	pp = np.polyfit(v[keep], cdf_v[keep], 3)
	gg = np.polyval(pp, q)

	out = {
		'poly': pp,
		'bounds': [v_lb, v_ub],
		'x': q,
		'p_lt': gg,
		'fn': lambda x: np.polyval(pp, x),
	}
	return out

def show_fit(poly_dict):
	fn_handle = poly_dict['fn']
	bds = poly_dict['bounds']
	xx = np.linspace(bds[0], bds[1], 100)
	yy = fn_handle(xx)

	plt.plot(xx, yy)

def poly_interval_transform(x):
	fbound = lambda x: (x + x ** 2.0 + x ** 3.0) / 3.0
	lb = fbound(x)
	ub = 1 - fbound(1 - x)
	return (lb, ub)