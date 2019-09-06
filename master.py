
import sys
import os

basedir = '/Users/brianlivingston/Documents/GitHub/DT_Consumption_Adj_Costs'
codedir = os.path.join(basedir,'code')
sys.path.append(codedir)

from setup.params import Params
from setup.income import Income

locIncomeProcess = os.path.join(basedir,'input/IncomeGrids/quarterly_b.mat')
paramsDict = {'locIncomeProcess': locIncomeProcess}
params = Params(paramsDict)

income = Income(params)