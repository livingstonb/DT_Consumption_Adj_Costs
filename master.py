
import sys
import os

basedir = '/Users/brianlivingston/Documents/GitHub/DT_Consumption_Adj_Costs'
codedir = os.path.join(basedir,'code')
sys.path.append(codedir)

from setup.params import Params
from setup.income import Income
from setup.grid import Grid

# create params object
locIncomeProcess = os.path.join(basedir,'input/IncomeGrids/quarterly_b.mat')
paramsDict = {'locIncomeProcess': locIncomeProcess}
params = Params(paramsDict)

# create income object
income = Income(params)

params.addIncomeParameters(income)

# create grids
grids = Grid(params,income)