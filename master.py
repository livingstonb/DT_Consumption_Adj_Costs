
import sys
import os

# declare repository directory
# basedir = '/Users/brianlivingston/Documents/GitHub/DT_Consumption_Adj_Costs'
# basedir = 'C:\\Users\\Brian-laptop\\Documents\\GitHub\\DT_Consumption_Adj_Costs'
basedir = '/mnt/c/Users/Brian-laptop/Documents/GitHub/DT_Consumption_Adj_Costs'
codedir = os.path.join(basedir,'code')
sys.path.append(codedir)

builddir = os.path.join(basedir,'build')
sys.path.append('build')

from model import modelObjects
from parameterizations.load_specifications import load_specifications
from build.model import Model

#---------------------------------------------------#
#                   SET OPTIONS                     #
#---------------------------------------------------#

# create params object
locIncomeProcess = os.path.join(basedir,'input','IncomeGrids','quarterly_b.mat')
params = load_specifications(locIncomeProcess, index=2)
# params = modelObjects.Params(paramsDict)

# create income object
income = modelObjects.Income(params)

params.addIncomeParameters(income)

# create grids
grids = modelObjects.Grid(params,income)

# initialize model
model = Model(params,income,grids)
model.solve()