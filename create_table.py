import os
import pandas as pd

# basedir = os.getcwd()
basedir = '/home/brian/Documents/GitHub/DT_Consumption_Adj_Costs/'
outdir = os.path.join(basedir,'output')

pkls = []
files = os.listdir(outdir)
for file in files:
	if file.endswith('.pkl'):
		pkls.append(file)

pkls = sorted(pkls)
pkls = [os.path.join(outdir,pkl) for pkl in pkls]

series_list = []
for pkl in pkls:
	series_list.append(pd.read_pickle(pkl))

df = pd.concat(series_list, axis=1)

xlpath = os.path.join(outdir,'table.xlsx')
df.to_excel(xlpath, freeze_panes=(0,0), header=False)