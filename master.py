
import sys
import os

# declare repository directory
# basedir = '/Users/brianlivingston/Documents/GitHub/DT_Consumption_Adj_Costs'
basedir = 'C:\\Users\\Brian-laptop\\Documents\\GitHub\\DT_Consumption_Adj_Costs'
codedir = os.path.join(basedir,'code')
sys.path.append(codedir)

from model_setup import modelObjects
from model_setup import load_specifications

from model.model import Model

#---------------------------------------------------#
#                   SET OPTIONS                     #
#---------------------------------------------------#

# create params object
locIncomeProcess = os.path.join(basedir,'input','IncomeGrids','quarterly_b.mat')
paramsDict = load_specifications(locIncomeProcess, index=1)
if len(sys.argv) > 1
	params = modelObjects.Params(paramsDict,sys.argv[1])
else
	params = modelObjects.Params(paramsDict,1)

# create income object
income = modelObjects.Income(params)

params.addIncomeParameters(income)

# create grids
grids = modelObjects.Grid(params,income)

# initialize model
model = Model(params,income,grids)
model.solveEGP()