
import sys
import os

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
locIncomeProcess = os.path.join(basedir,'input','IncomeGrids','incomeDebug.mat')
params = load_specifications(locIncomeProcess, index=2)
# params = modelObjects.Params(paramsDict)

# create income object
income = modelObjects.Income(params)

params.addIncomeParameters(income)

# create grids
grids = modelObjects.Grid(params,income)

# initialize and solve for policy functions
model = Model(params,income,grids)
model.solve()

# solve for stationary distribution
eqSimulator = EquilibriumSimulator(params, income, grids, model)
eqSimulator.simulate()