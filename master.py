import sys
import os
from matplotlib import pyplot as plt
import numpy as np

import pandas as pd

from model import Params, Income, Grid
from misc.load_specifications import load_specifications
from misc import mpcsTable, functions, otherStatistics
from misc.Calibrator import Calibrator
from model.model import Model, ModelWithNews
from model import simulator

from IPython.core.debugger import set_trace

#---------------------------------------------------------------#
#      SET PARAMETERIZATION NUMBER                              #
#---------------------------------------------------------------#
indexSet = False
for arg in sys.argv:
	try:
		paramIndex = int(arg)
		indexSet = True
		break
	except:
		pass

if not indexSet:
	paramIndex = 1

#---------------------------------------------------------------#
#      OR SET PARAMETERIZATION NAME                             #
#---------------------------------------------------------------#
# THIS OVERRIDES paramIndex: TO IGNORE SET TO EMPTY STRING
name = 'target P(assets<1000) and P(MPC>0) = 0.2'

#---------------------------------------------------------------#
#      OPTIONS                                                  #
#---------------------------------------------------------------#
Calibrate = False
Simulate = True # relevant if Calibrate is False
SimulateMPCs = True
Fast = False
PrintGrids = False

basedir = os.getcwd()
outdir = os.path.join(basedir, 'output')
if not os.path.exists(outdir):
	os.mkdir(outdir)

#---------------------------------------------------------------#
#      LOCATION OF INCOME PROCESS                               #
#---------------------------------------------------------------#
locIncomeProcess = os.path.join(
	basedir,'input', 'income_quarterly_b_fixed.mat')

#---------------------------------------------------------------#
#      LOAD PARAMETERS                                          #
#---------------------------------------------------------------#
if name:
	params_dict = load_specifications(locIncomeProcess, name=name)
else:
	params_dict = load_specifications(locIncomeProcess, index=paramIndex)

params_dict['fastSettings'] = Fast
params = Params.Params(params_dict)

#---------------------------------------------------------------#
#      LOAD INCOME PROCESS                                      #
#---------------------------------------------------------------#
income = Income.Income(params, False)
params.addIncomeParameters(income)

#---------------------------------------------------------------#
#      CREATE GRIDS                                             #
#---------------------------------------------------------------#
grids = Grid.Grid(params, income)

if PrintGrids:
	print('Cash grid:')
	functions.printVector(grids.x_flat)

	print('Consumption grid:')
	functions.printVector(grids.c_flat)
	quit()

#---------------------------------------------------------------#
#      INITIALIZE AND SOLVE MODEL                                             #
#---------------------------------------------------------------#
model = Model(params, income, grids)
model.initialize()

if Calibrate:
	calibrator = Calibrator(params.cal_options)
	calibrator.calibrate(params, model, income, grids)
else:
	model.solve()

eqSimulator = simulator.EquilibriumSimulator(params, income, grids, 
	model.cSwitchingPolicy, model.inactionRegion)
if Simulate:
	eqSimulator.simulate()
	finalSimStates = eqSimulator.finalStates
else:
	finalSimStates = []

#-----------------------------------------------------------#
#      SIMULATE MPCs OUT OF AN IMMEDIATE SHOCK              #
#-----------------------------------------------------------#
shockIndices = [0,1,2,3,4,5]

mpcSimulator = simulator.MPCSimulator(
	params, income, grids, 
	model.cSwitchingPolicy,
	model.inactionRegion, 
	shockIndices, finalSimStates)

if Simulate and SimulateMPCs:
	mpcSimulator.simulate()

# # print(eqSimulator.results.dropna().to_string())
# print(mpcSimulator.results.dropna().to_string())
# exit()

#-----------------------------------------------------------#
#      SOLVE FOR POLICY GIVEN SHOCK NEXT PERIOD             #
#-----------------------------------------------------------#
shockIndices_shockNextPeriod = [2,3,4,5]
currentShockIndices = [6] * len(shockIndices_shockNextPeriod) # 6 is shock of 0

cSwitch_shockNextPeriod = np.zeros((params.nx,1,params.nz,params.nyP,len(shockIndices_shockNextPeriod)+1))
inactionRegions_shockNextPeriod = np.zeros((params.nx,2,params.nz,params.nyP,len(shockIndices_shockNextPeriod)+1))

cSwitch_shockNextPeriod[:,:,:,:,-1] = model.cSwitchingPolicy[:,:,:,:,0]
inactionRegions_shockNextPeriod[:,:,:,:,-1] = model.inactionRegion[:,:,:,:,0]
valueBaseline = model.valueFunction
model.interpMat = []

i = 0
for ishock in shockIndices_shockNextPeriod:
	model_shockNextPeriod = ModelWithNews(
		params, income, grids,
		valueBaseline,
		params.MPCshocks[ishock],
		1)

	if SimulateMPCs:
		print(f'Solving for shock of {params.MPCshocks[ishock]} next period')
		model_shockNextPeriod.solve()
		cSwitch_shockNextPeriod[:,:,:,:,i] = model_shockNextPeriod.cSwitchingPolicy[:,:,:,:,0]
		inactionRegions_shockNextPeriod[:,:,:,:,i] = model_shockNextPeriod.inactionRegion[:,:,:,:,0]

	del model_shockNextPeriod
	i += 1

#-----------------------------------------------------------#
#      SOLVE FOR 1-YEAR LOAN                                #
#-----------------------------------------------------------#
cSwitch_loan = np.zeros((params.nx,1,params.nz,params.nyP,2))
inactionRegions_loan = np.zeros((params.nx,2,params.nz,params.nyP,2))

shock = params.MPCshocks[0]
# start with last quarter
model_loan = ModelWithNews(
	params, income, grids,
	valueBaseline,
	shock,
	1)

if SimulateMPCs:
	print(f'Solving for one year loan')
	model_loan.solve()

	for period in range(2, 5):
		model_loan = ModelWithNews(
			params, income, grids,
			model_loan.valueFunction,
			shock,
			period)
		model_loan.solve()

	cSwitch_loan[:,:,:,:,0] = model_loan.cSwitchingPolicy[:,:,:,:,0]
	cSwitch_loan[:,:,:,:,1] = model.cSwitchingPolicy[:,:,:,:,0]
	inactionRegions_loan[:,:,:,:,0] = model_loan.inactionRegion[:,:,:,:,0]
	inactionRegions_loan[:,:,:,:,1] = model.inactionRegion[:,:,:,:,0]
	del model_loan

#-----------------------------------------------------------#
#      SHOCK OF -$500 IN 2 YEARS                            #
#-----------------------------------------------------------#
cSwitch_shock2Years = np.zeros((params.nx,1,params.nz,params.nyP,2))
inactionRegions_shock2Years = np.zeros((params.nx,2,params.nz,params.nyP,2))

ishock = 2
shock = params.MPCshocks[ishock]
model_shock2Years = ModelWithNews(
	params, income, grids,
	valueBaseline,
	shock,
	1)

if SimulateMPCs:
	model_shock2Years.solve()

	for period in range(2, 9):
		shock = 0.0
		model_shock2Years = ModelWithNews(
			params, income, grids,
			model_shock2Years.valueFunction,
			shock,
			period)
		model_shock2Years.solve()

	cSwitch_shock2Years[:,:,:,:,0] = model_shock2Years.cSwitchingPolicy[:,:,:,:,0]
	cSwitch_shock2Years[:,:,:,:,1] = model.cSwitchingPolicy[:,:,:,:,0]
	inactionRegions_shock2Years[:,:,:,:,0] = model_shock2Years.inactionRegion[:,:,:,:,0]
	inactionRegions_shock2Years[:,:,:,:,1] = model.inactionRegion[:,:,:,:,0]
	del model_shock2Years

#-----------------------------------------------------------#
#      SIMULATE MPCs OUT OF NEWS                            #
#-----------------------------------------------------------#
currentShockIndices = [6] * len(shockIndices_shockNextPeriod)
mpcNewsSimulator_shockNextPeriod = simulator.MPCSimulatorNews(
	params, income, grids,
	cSwitch_shockNextPeriod, inactionRegions_shockNextPeriod,
	shockIndices_shockNextPeriod, currentShockIndices,
	finalSimStates, periodsUntilShock=1)

shockIndices_shock2Years = [2]
currentShockIndices = [6]
mpcNewsSimulator_shock2Years = simulator.MPCSimulatorNews(
	params, income, grids,
	cSwitch_shock2Years, inactionRegions_shock2Years,
	shockIndices_shock2Years, currentShockIndices,
	finalSimStates, periodsUntilShock=8)

shockIndices_loan = [0]
currentShockIndices = [5]
mpcNewsSimulator_loan = simulator.MPCSimulatorNews_Loan(
	params, income, grids,
	cSwitch_loan, inactionRegions_loan,
	shockIndices_loan, currentShockIndices,
	finalSimStates, periodsUntilShock=4)

if SimulateMPCs:
	mpcNewsSimulator_shockNextPeriod.simulate()
	mpcNewsSimulator_shock2Years.simulate()
	mpcNewsSimulator_loan.simulate()

#-----------------------------------------------------------#
#      RESULTS                                              #
#-----------------------------------------------------------#
# find fractions of households that respond to one, both, or neither of
# two treatments
if Simulate:
	otherStatistics.saveWealthGroupStats(
		mpcSimulator, mpcNewsSimulator_shockNextPeriod,
		mpcNewsSimulator_shock2Years, mpcNewsSimulator_loan,
		finalSimStates, outdir, paramIndex, params)

	# parameters
	print('\nSelected parameters:\n')
	print(params.series.to_string())

	# put main results into a Series
	print('\nResults from simulation:\n')
	print(eqSimulator.results.dropna().to_string())

	print('\nMPCS:\n')
	print(mpcSimulator.results.dropna().to_string())

	print('\nMPCS out of news:\n')
	print(mpcNewsSimulator_shockNextPeriod.results.dropna().to_string())

	print('\nMPCS out of future loss:\n')
	print(mpcNewsSimulator_shock2Years.results.dropna().to_string())
	print(mpcNewsSimulator_loan.results.dropna().to_string())

	name_series = pd.Series({'Experiment':params.name})
	index_series = pd.Series({'Index':params.index})
	results = pd.concat([	name_series,
							index_series,
							params.series, 
							eqSimulator.results.dropna(),
							mpcSimulator.results.dropna(),
							mpcNewsSimulator_shockNextPeriod.results.dropna(),
							mpcNewsSimulator_shock2Years.results.dropna(),
							mpcNewsSimulator_loan.results.dropna(),
							])

	savepath = os.path.join(outdir,f'run{paramIndex}.pkl')
	results.to_pickle(savepath)

	savepath = os.path.join(outdir,f'run{paramIndex}_statistics.csv')
	results.to_csv(savepath, index_label=params.name, header=True)

	mpcs_table = mpcsTable.create(params, mpcSimulator, 
		mpcNewsSimulator_shockNextPeriod,
		mpcNewsSimulator_shock2Years,
		mpcNewsSimulator_loan,
		)
	savepath = os.path.join(outdir,f'run{paramIndex}_mpcs_table.csv')
	# mpcs_table.to_excel(savepath, freeze_panes=(0,0), index_label=params.name)
	mpcs_table.to_csv(savepath, index_label=params.name, header=True)

#-----------------------------------------------------------#
#      PLOTS                                                #
#-----------------------------------------------------------#
ixvals = [0,params.nx//8,params.nx//6,params.nx//4,params.nx//3,params.nx//2,params.nx-1]
xvals = np.array([grids.x_flat[i] for i in ixvals])

icvals = [0,params.nc//6,params.nc//5,params.nc//4,params.nc//3,params.nc//2,params.nc-1]
cvals = np.array([grids.c_flat[i] for i in icvals])

def plot_policies():
	cPolicy = model.cChosen

	print(xvals)

	if params.nyP == 1:
		iyP = 0
	else:
		iyP = 5

	fig, ax = plt.subplots(nrows=2,ncols=3)
	fig.suptitle('Consumption function vs. state c')
	i = 0
	for row in range(2):
		for col in range(3):
			ax[row,col].scatter(grids.c_flat,cPolicy[ixvals[i],:,0,iyP])
			ax[row,col].set_title(f'x = {xvals[i]}')
			ax[row,col].set_xlabel('c, state')
			ax[row,col].set_ylabel('actual consumption')
			i += 1

	fig, ax = plt.subplots(nrows=2,ncols=3)
	fig.suptitle('Consumption function vs. state c, zoomed')
	i = 0
	for row in range(2):
		for col in range(3):
			ax[row,col].scatter(grids.c_flat,cPolicy[ixvals[i],:,0,iyP])
			ax[row,col].set_title(f'x = {xvals[i]}')
			ax[row,col].set_xlabel('c, state')
			ax[row,col].set_ylabel('actual consumption')
			ax[row,col].set_xbound(0, 0.5)
			i += 1

	fig, ax = plt.subplots(nrows=2,ncols=3)
	fig.suptitle('EMAX vs. state c')
	i = 0
	for row in range(2):
		for col in range(3):
			ax[row,col].scatter(grids.c_flat,model.EMAX[ixvals[i],:,0,iyP])
			ax[row,col].set_title(f'x = {xvals[i]}')
			ax[row,col].set_xlabel('c, state')
			ax[row,col].set_ylabel('EMAX')
			i += 1

	fig, ax = plt.subplots(nrows=2,ncols=3)
	fig.suptitle('Consumption function vs. assets')
	i = 0
	for row in range(2):
		for col in range(3):
			ax[row,col].scatter(grids.x_flat,cPolicy[:,icvals[i],0,iyP])
			ax[row,col].set_title(f'c = {cvals[i]}')
			ax[row,col].set_xlabel('x, cash-on-hand')
			ax[row,col].set_ylabel('actual consumption')
			i += 1

	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	ax.scatter(grids.x_flat, model.inactionRegion[:,0,0,iyP])
	ax.scatter(grids.x_flat, model.cSwitchingPolicy[:,0,0,iyP])
	ax.scatter(grids.x_flat, model.inactionRegion[:,1,0,iyP])

	ax.set_title('Inaction region for consumption')
	ax.set_xlabel('cash-on-hand, x')
	ax.set_ylabel('consumption')
	ax.legend(['Lower bound of inaction region', 'Desired c if switching', 
		'Upper bound of inaction region'])

	plt.show()

def plot_mpcs():
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