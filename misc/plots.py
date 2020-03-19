from matplotlib import pyplot as plt
import numpy as np
import os

def plot_policies(model, grids, params, pnum, outdir):
	cPolicy = model.cChosen

	ixvals = [15, 30, 35, 40, 45, 48]
	xvals = np.array([grids.x_flat[i] for i in ixvals])

	icvals = [15, 30, 40, 50, 60, 73]
	cvals = np.array([grids.c_flat[i] for i in icvals])

	if params.nyP == 1:
		iyP = 0
	else:
		iyP = 5

	fig, ax = plt.subplots(nrows=2,ncols=3)
	fig.suptitle('Consumption function vs. state c (middle yP value)')
	i = 0
	for row in range(2):
		for col in range(3):
			ax[row,col].scatter(grids.c_flat, cPolicy[ixvals[i],:,0,iyP])
			ax[row,col].set_title(f'x = {xvals[i]}')
			ax[row,col].set_xlabel('c, state')
			ax[row,col].set_ylabel('actual consumption')
			i += 1

	pname = f'cpolicy_vs_c_run{pnum}.jpg'
	pname = os.path.join(outdir, pname)
	fig.savefig(pname)

	fig, ax = plt.subplots(nrows=2,ncols=3)
	fig.suptitle('Consumption function vs. state c (middle yP value), zoomed')
	i = 0
	for row in range(2):
		for col in range(3):
			ax[row,col].scatter(grids.c_flat, cPolicy[ixvals[i],:,0,iyP])
			ax[row,col].set_title(f'x = {xvals[i]}')
			ax[row,col].set_xlabel('c, state')
			ax[row,col].set_ylabel('c*, chosen consumption')
			ax[row,col].set_xbound(0, 0.5)
			i += 1

	pname = f'cpolicy_vs_c_zoomed_run{pnum}.jpg'
	pname = os.path.join(outdir, pname)
	fig.savefig(pname)

	fig, ax = plt.subplots(nrows=2,ncols=3)
	fig.suptitle('Consumption function vs. cash-on-hand (middle yP value)')
	i = 0
	for row in range(2):
		for col in range(3):
			ax[row,col].scatter(grids.x_flat, cPolicy[:,icvals[i],0,iyP])
			ax[row,col].set_title(f'c = {cvals[i]}')
			ax[row,col].set_xlabel('x, cash-on-hand')
			ax[row,col].set_ylabel('chosen consumption')
			ax[row,col].set_xlim(0, 5.0)
			ax[row,col].set_ylim(auto=True)
			i += 1

	pname = f'cpolicy_vs_cash_run{pnum}.jpg'
	pname = os.path.join(outdir, pname)
	fig.savefig(pname)

	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	ax.scatter(grids.x_flat, model.inactionRegion[:,0,0,iyP])
	ax.scatter(grids.x_flat, model.cSwitchingPolicy[:,0,0,iyP])
	ax.scatter(grids.x_flat, model.inactionRegion[:,1,0,iyP])

	ax.set_title('Inaction region for consumption (middle yP value)')
	ax.set_xlabel('cash-on-hand, x')
	ax.set_ylabel('consumption')
	ax.legend(['Lower bound of inaction region', 'Optimal c if switching', 
		'Upper bound of inaction region'])
	ax.set_xlim(0, 1.0)
	ax.set_ylim(auto=True)

	pname = f'Inaction_vs_cash_run{pnum}.jpg'
	pname = os.path.join(outdir, pname)
	fig.savefig(pname)

	plt.show()

def plot_mpcs(model, grids, params):
	if not Simulate:
		return

	ishock = 4
	idx_yP = np.asarray(mpcSimulator.finalStates['yPind']) == 5
	idx_yP = idx_yP.reshape((-1,1))
	mpcs = np.asarray(mpcSimulator.mpcs[ishock]).reshape((-1,1))
	cash = np.asarray(mpcSimulator.finalStates['xsim'])
	c = np.asarray(mpcSimulator.finalStates['csim'])

	fig, ax = plt.subplots(nrows=2, ncols=3)
	fig.suptitle('Quarterly MPC vs. initial cash-on-hand')
	i = 0
	for row in range(2):
		for col in range(3):
			idx_c = np.logical_and(c >= cvals[i], c < cvals[i+1])
			x = cash[np.logical_and(idx_yP,idx_c)]
			y = mpcs[np.logical_and(idx_yP,idx_c)]
			ax[row,col].scatter(x,y)
			ax[row,col].set_title(f'{cvals[i]:.3g} <= state c < {cvals[i+1]:.3g}')
			ax[row,col].set_xlabel('cash-on-hand, x')
			ax[row,col].set_ylabel('MPC out of 0.01')
			i += 1

	fig, ax = plt.subplots(nrows=2, ncols=3)
	fig.suptitle('Quarterly MPC vs. consumption state')
	i = 0
	for row in range(2):
		for col in range(3):
			idx_x = np.logical_and(cash >= xvals[i], cash < xvals[i+1])
			x = c[np.logical_and(idx_yP,idx_x)]
			y = mpcs[np.logical_and(idx_yP,idx_x)]
			ax[row,col].scatter(x,y)
			ax[row,col].set_title(f'{xvals[i]:.3g} <= x < {xvals[i+1]:.3g}')
			ax[row,col].set_xlabel('state c')
			ax[row,col].set_ylabel('MPC out of 0.01')
			i += 1

	plt.show()