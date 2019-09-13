import sys
import os
from matplotlib import pyplot as plt
import numpy as np

# declare repository directory
# basedir = '/Users/brianlivingston/Documents/GitHub/DT_Consumption_Adj_Costs'
# basedir = 'C:\\Users\\Brian-laptop\\Documents\\GitHub\\DT_Consumption_Adj_Costs'
basedir = os.getcwd()

builddir = os.path.join(basedir,'build')
sys.path.append(builddir)

from model import modelObjects
from misc.load_specifications import load_specifications
from model.model import Model
from model.simulator import EquilibriumSimulator

#---------------------------------------------------#
#                   SET OPTIONS                     #
#---------------------------------------------------#

# create params object
locIncomeProcess = os.path.join(basedir,'input','IncomeGrids','quarterly_b.mat')
params = load_specifications(locIncomeProcess, index=0)
# params = modelObjects.Params(paramsDict)

# create income object
income = modelObjects.Income(params)

params.addIncomeParameters(income)

# create grids
grids = modelObjects.Grid(params,income)
# import pdb; pdb.set_trace()

# initialize and solve for policy functions
model = Model(params,income,grids)
model.solve()
# model.plotPolicyFunctions()

cSwitch = np.asarray(model.valueSwitch) - params.adjustCost > np.asarray(model.valueNoSwitch)
cPolicy = cSwitch * np.asarray(model.cSwitchingPolicy) + (~cSwitch) * grids.c['matrix']

# ixvals = [0,10,20,40]
# xvals = np.array([grids.x['wide'][i,0,0,5] for i in ixvals])

# fig, ax = plt.subplots(nrows=2,ncols=2)
# i = 0
# for row in range(2):
# 	for col in range(2):
# 		ax[row,col].plot(grids.c['matrix'][ixvals[i],:,0,5],cPolicy[ixvals[i],:,0,5])
# 		i += 1

# plt.show()

# # solve for stationary distribution
eqSimulator = EquilibriumSimulator(params, income, grids, model)
eqSimulator.simulate()

