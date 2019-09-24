import pandas as pd
import numpy as np

def create(p, mpcSimulator, mpcNewsSimulator):

	index = 	[	
					'GAIN',
					'  $500',
					'  $2500',
					'  $5000',
					'LOSS',
					'  $500',
					'  $2500',
					'  $5000',
					'NEWS-GAIN',
					'  $500 in 3 months',
					'  $2500 in 3 months',
					'  $5000 in 3 months',
					'NEWS-LOSS',
					'  $500 in 3 months',
					'  $2500 in 3 months',
					'  $5000 in 3 months',
					'LOAN',
					'  $500',
					'  $2500',
					'  $5000',
				]

	colnames = [	
					'E[MPC]',
					'Share MPC < 0',
					'Share MPC = 0',
					'Share MPC > 0',
					'E[MPC|MPC>0]',
					'Median(MPC|MPC>0)',
				]
	
	mean_mpc = [	
					np.nan,
					mpcSimulator.results[f'E[Q1 MPC] out of {p.MPCshocks[3]}'],
					mpcSimulator.results[f'E[Q1 MPC] out of {p.MPCshocks[4]}'],
					mpcSimulator.results[f'E[Q1 MPC] out of {p.MPCshocks[5]}'],
					np.nan,
					mpcSimulator.results[f'E[Q1 MPC] out of {p.MPCshocks[0]}'],
					mpcSimulator.results[f'E[Q1 MPC] out of {p.MPCshocks[1]}'],
					mpcSimulator.results[f'E[Q1 MPC] out of {p.MPCshocks[2]}'],
					np.nan,
					mpcNewsSimulator.results[f'E[Q1 MPC] out of news of {p.MPCshocks[3]} shock'],
					mpcNewsSimulator.results[f'E[Q1 MPC] out of news of {p.MPCshocks[4]} shock'],
					mpcNewsSimulator.results[f'E[Q1 MPC] out of news of {p.MPCshocks[5]} shock'],
					np.nan,
					np.nan,
					np.nan,
					np.nan,
					np.nan,
					np.nan,
					np.nan,
					np.nan,
				]

	neg_mpc_share = [	
						np.nan,
						mpcSimulator.results[f'P(Q1 MPC < 0) for shock of {p.MPCshocks[3]}'],
						mpcSimulator.results[f'P(Q1 MPC < 0) for shock of {p.MPCshocks[4]}'],
						mpcSimulator.results[f'P(Q1 MPC < 0) for shock of {p.MPCshocks[5]}'],
						np.nan,
						mpcSimulator.results[f'P(Q1 MPC < 0) for shock of {p.MPCshocks[0]}'],
						mpcSimulator.results[f'P(Q1 MPC < 0) for shock of {p.MPCshocks[1]}'],
						mpcSimulator.results[f'P(Q1 MPC < 0) for shock of {p.MPCshocks[2]}'],
						np.nan,
						mpcNewsSimulator.results[f'P(Q1 MPC < 0) for news of {p.MPCshocks[3]} shock'],
						mpcNewsSimulator.results[f'P(Q1 MPC < 0) for news of {p.MPCshocks[4]} shock'],
						mpcNewsSimulator.results[f'P(Q1 MPC < 0) for news of {p.MPCshocks[5]} shock'],
						np.nan,
						np.nan,
						np.nan,
						np.nan,
						np.nan,
						np.nan,
						np.nan,
						np.nan,
					]

	zero_mpc_share = [	
						np.nan,
						mpcSimulator.results[f'P(Q1 MPC = 0) for shock of {p.MPCshocks[3]}'],
						mpcSimulator.results[f'P(Q1 MPC = 0) for shock of {p.MPCshocks[4]}'],
						mpcSimulator.results[f'P(Q1 MPC = 0) for shock of {p.MPCshocks[5]}'],
						np.nan,
						mpcSimulator.results[f'P(Q1 MPC = 0) for shock of {p.MPCshocks[0]}'],
						mpcSimulator.results[f'P(Q1 MPC = 0) for shock of {p.MPCshocks[1]}'],
						mpcSimulator.results[f'P(Q1 MPC = 0) for shock of {p.MPCshocks[2]}'],
						np.nan,
						mpcNewsSimulator.results[f'P(Q1 MPC = 0) for news of {p.MPCshocks[3]} shock'],
						mpcNewsSimulator.results[f'P(Q1 MPC = 0) for news of {p.MPCshocks[4]} shock'],
						mpcNewsSimulator.results[f'P(Q1 MPC = 0) for news of {p.MPCshocks[5]} shock'],
						np.nan,
						np.nan,
						np.nan,
						np.nan,
						np.nan,
						np.nan,
						np.nan,
						np.nan,
						]

	pos_mpc_share = [	
						np.nan,
						mpcSimulator.results[f'P(Q1 MPC > 0) for shock of {p.MPCshocks[3]}'],
						mpcSimulator.results[f'P(Q1 MPC > 0) for shock of {p.MPCshocks[4]}'],
						mpcSimulator.results[f'P(Q1 MPC > 0) for shock of {p.MPCshocks[5]}'],
						np.nan,
						mpcSimulator.results[f'P(Q1 MPC > 0) for shock of {p.MPCshocks[0]}'],
						mpcSimulator.results[f'P(Q1 MPC > 0) for shock of {p.MPCshocks[1]}'],
						mpcSimulator.results[f'P(Q1 MPC > 0) for shock of {p.MPCshocks[2]}'],
						np.nan,
						mpcNewsSimulator.results[f'P(Q1 MPC > 0) for news of {p.MPCshocks[3]} shock'],
						mpcNewsSimulator.results[f'P(Q1 MPC > 0) for news of {p.MPCshocks[4]} shock'],
						mpcNewsSimulator.results[f'P(Q1 MPC > 0) for news of {p.MPCshocks[5]} shock'],
						np.nan,
						np.nan,
						np.nan,
						np.nan,
						np.nan,
						np.nan,
						np.nan,
						np.nan,
						]

	mean_cond_mpc = [	
						np.nan,
						mpcSimulator.results[f'E[Q1 MPC | MPC > 0] out of {p.MPCshocks[3]}'],
						mpcSimulator.results[f'E[Q1 MPC | MPC > 0] out of {p.MPCshocks[4]}'],
						mpcSimulator.results[f'E[Q1 MPC | MPC > 0] out of {p.MPCshocks[5]}'],
						np.nan,
						mpcSimulator.results[f'E[Q1 MPC | MPC > 0] out of {p.MPCshocks[0]}'],
						mpcSimulator.results[f'E[Q1 MPC | MPC > 0] out of {p.MPCshocks[1]}'],
						mpcSimulator.results[f'E[Q1 MPC | MPC > 0] out of {p.MPCshocks[2]}'],
						np.nan,
						mpcNewsSimulator.results[f'E[Q1 MPC | MPC > 0] out of news of {p.MPCshocks[3]} shock'],
						mpcNewsSimulator.results[f'E[Q1 MPC | MPC > 0] out of news of {p.MPCshocks[4]} shock'],
						mpcNewsSimulator.results[f'E[Q1 MPC | MPC > 0] out of news of {p.MPCshocks[5]} shock'],
						np.nan,
						np.nan,
						np.nan,
						np.nan,
						np.nan,
						np.nan,
						np.nan,
						np.nan,
						]

	med_cond_mpc = [
						np.nan,
						mpcSimulator.results[f'Median(Q1 MPC | MPC > 0) out of {p.MPCshocks[3]}'],
						mpcSimulator.results[f'Median(Q1 MPC | MPC > 0) out of {p.MPCshocks[4]}'],
						mpcSimulator.results[f'Median(Q1 MPC | MPC > 0) out of {p.MPCshocks[5]}'],
						np.nan,
						mpcSimulator.results[f'Median(Q1 MPC | MPC > 0) out of {p.MPCshocks[0]}'],
						mpcSimulator.results[f'Median(Q1 MPC | MPC > 0) out of {p.MPCshocks[1]}'],
						mpcSimulator.results[f'Median(Q1 MPC | MPC > 0) out of {p.MPCshocks[2]}'],
						np.nan,
						mpcNewsSimulator.results[f'Median(Q1 MPC | MPC > 0) out of news of {p.MPCshocks[3]} shock'],
						mpcNewsSimulator.results[f'Median(Q1 MPC | MPC > 0) out of news of {p.MPCshocks[4]} shock'],
						mpcNewsSimulator.results[f'Median(Q1 MPC | MPC > 0) out of news of {p.MPCshocks[5]} shock'],
						np.nan,
						np.nan,
						np.nan,
						np.nan,
						np.nan,
						np.nan,
						np.nan,
						np.nan,
					]


	i = 0
	cols = [mean_mpc, neg_mpc_share, zero_mpc_share, pos_mpc_share, mean_cond_mpc, med_cond_mpc]
	cols_series = []
	for col in cols:
		cols_series.append(pd.Series(col, index=index, name=colnames[i]))
		i += 1

	df = pd.concat(cols_series, axis=1)

	return df
	