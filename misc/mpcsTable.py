import pandas as pd
import numpy as np

def create(p, mpcSimulator, mpcNewsSimulator_shockNextPeriod,
	mpcNewsSimulator_shock2Years, mpcNewsSimulator_loan):

	index = 	[	
					'GAIN',
					'  $500',
					'  $2500',
					'  $5000',
					'LOSS',
					'  $500',
					'NEWS-GAIN',
					'  $500 in 3 months',
					'  $5000 in 3 months',
					'NEWS-LOSS',
					'  $500 in 3 months',
					'  $500 in 2 years',
					'LOAN',
					'  $5000 for 1 year',
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
					mpcSimulator.results[f'E[Q1 MPC] out of {p.MPCshocks[2]}'],
					np.nan,
					mpcNewsSimulator_shockNextPeriod.results[f'E[Q1 MPC] out of news of {p.MPCshocks[3]} shock in 1 quarter(s)'],
					mpcNewsSimulator_shockNextPeriod.results[f'E[Q1 MPC] out of news of {p.MPCshocks[5]} shock in 1 quarter(s)'],
					np.nan,
					mpcNewsSimulator_shockNextPeriod.results[f'E[Q1 MPC] out of news of {p.MPCshocks[2]} shock in 1 quarter(s)'],
					mpcNewsSimulator_shock2Years.results[f'E[Q1 MPC] out of news of {p.MPCshocks[2]} shock in 8 quarter(s)'],
					np.nan,
					mpcNewsSimulator_loan.results[f'E[Q1 MPC] out of {p.MPCshocks[5]} loan for 4 quarter(s)'],
				]

	neg_mpc_share = [	
						np.nan,
						mpcSimulator.results[f'P(Q1 MPC < 0) for shock of {p.MPCshocks[3]}'],
						mpcSimulator.results[f'P(Q1 MPC < 0) for shock of {p.MPCshocks[4]}'],
						mpcSimulator.results[f'P(Q1 MPC < 0) for shock of {p.MPCshocks[5]}'],
						np.nan,
						mpcSimulator.results[f'P(Q1 MPC < 0) for shock of {p.MPCshocks[2]}'],
						np.nan,
						mpcNewsSimulator_shockNextPeriod.results[f'P(Q1 MPC < 0) for news of {p.MPCshocks[3]} shock in 1 quarter(s)'],
						mpcNewsSimulator_shockNextPeriod.results[f'P(Q1 MPC < 0) for news of {p.MPCshocks[5]} shock in 1 quarter(s)'],
						np.nan,
						mpcNewsSimulator_shockNextPeriod.results[f'P(Q1 MPC < 0) for news of {p.MPCshocks[2]} shock in 1 quarter(s)'],
						mpcNewsSimulator_shock2Years.results[f'P(Q1 MPC < 0) for news of {p.MPCshocks[2]} shock in 8 quarter(s)'],
						np.nan,
						mpcNewsSimulator_loan.results[f'P(Q1 MPC < 0) for {-p.MPCshocks[0]} loan for 4 quarter(s)'],
					]

	zero_mpc_share = [	
						np.nan,
						mpcSimulator.results[f'P(Q1 MPC = 0) for shock of {p.MPCshocks[3]}'],
						mpcSimulator.results[f'P(Q1 MPC = 0) for shock of {p.MPCshocks[4]}'],
						mpcSimulator.results[f'P(Q1 MPC = 0) for shock of {p.MPCshocks[5]}'],
						np.nan,
						mpcSimulator.results[f'P(Q1 MPC = 0) for shock of {p.MPCshocks[2]}'],
						np.nan,
						mpcNewsSimulator_shockNextPeriod.results[f'P(Q1 MPC = 0) for news of {p.MPCshocks[3]} shock in 1 quarter(s)'],
						mpcNewsSimulator_shockNextPeriod.results[f'P(Q1 MPC = 0) for news of {p.MPCshocks[5]} shock in 1 quarter(s)'],
						np.nan,
						mpcNewsSimulator_shockNextPeriod.results[f'P(Q1 MPC = 0) for news of {p.MPCshocks[2]} shock in 1 quarter(s)'],
						mpcNewsSimulator_shock2Years.results[f'P(Q1 MPC = 0) for news of {p.MPCshocks[3]} shock in 8 quarter(s)'],
						np.nan,
						mpcNewsSimulator_loan.results[f'P(Q1 MPC = 0) for {-p.MPCshocks[0]} loan for 4 quarter(s)'],
						]

	pos_mpc_share = [	
						np.nan,
						mpcSimulator.results[f'P(Q1 MPC > 0) for shock of {p.MPCshocks[3]}'],
						mpcSimulator.results[f'P(Q1 MPC > 0) for shock of {p.MPCshocks[4]}'],
						mpcSimulator.results[f'P(Q1 MPC > 0) for shock of {p.MPCshocks[5]}'],
						np.nan,
						mpcSimulator.results[f'P(Q1 MPC > 0) for shock of {p.MPCshocks[2]}'],
						np.nan,
						mpcNewsSimulator_shockNextPeriod.results[f'P(Q1 MPC > 0) for news of {p.MPCshocks[3]} shock in 1 quarter(s)'],
						mpcNewsSimulator_shockNextPeriod.results[f'P(Q1 MPC > 0) for news of {p.MPCshocks[5]} shock in 1 quarter(s)'],
						np.nan,
						mpcNewsSimulator_shockNextPeriod.results[f'P(Q1 MPC > 0) for news of {p.MPCshocks[2]} shock in 1 quarter(s)'],
						mpcNewsSimulator_shock2Years.results[f'P(Q1 MPC > 0) for news of {p.MPCshocks[2]} shock in 8 quarter(s)'],
						np.nan,
						mpcNewsSimulator_loan.results[f'P(Q1 MPC > 0) for {-p.MPCshocks[0]} loan for 4 quarter(s)'],
						]

	mean_cond_mpc = [	
						np.nan,
						mpcSimulator.results[f'E[Q1 MPC | MPC > 0] out of {p.MPCshocks[3]}'],
						mpcSimulator.results[f'E[Q1 MPC | MPC > 0] out of {p.MPCshocks[4]}'],
						mpcSimulator.results[f'E[Q1 MPC | MPC > 0] out of {p.MPCshocks[5]}'],
						np.nan,
						mpcSimulator.results[f'E[Q1 MPC | MPC > 0] out of {p.MPCshocks[2]}'],
						np.nan,
						mpcNewsSimulator_shockNextPeriod.results[f'E[Q1 MPC | MPC > 0] out of news of {p.MPCshocks[3]} shock in 1 quarter(s)'],
						mpcNewsSimulator_shockNextPeriod.results[f'E[Q1 MPC | MPC > 0] out of news of {p.MPCshocks[5]} shock in 1 quarter(s)'],
						np.nan,
						mpcNewsSimulator_shockNextPeriod.results[f'E[Q1 MPC | MPC > 0] out of news of {p.MPCshocks[2]} shock in 1 quarter(s)'],
						mpcNewsSimulator_shock2Years.results[f'E[Q1 MPC | MPC > 0] out of news of {p.MPCshocks[2]} shock in 8 quarter(s)'],
						np.nan,
						mpcNewsSimulator_loan.results[f'E[Q1 MPC | MPC > 0] out of {-p.MPCshocks[0]} loan for 4 quarter(s)'],
						]

	med_cond_mpc = [
						np.nan,
						mpcSimulator.results[f'Median(Q1 MPC | MPC > 0) out of {p.MPCshocks[3]}'],
						mpcSimulator.results[f'Median(Q1 MPC | MPC > 0) out of {p.MPCshocks[4]}'],
						mpcSimulator.results[f'Median(Q1 MPC | MPC > 0) out of {p.MPCshocks[5]}'],
						np.nan,
						mpcSimulator.results[f'Median(Q1 MPC | MPC > 0) out of {p.MPCshocks[2]}'],
						np.nan,
						mpcNewsSimulator_shockNextPeriod.results[f'Median(Q1 MPC | MPC > 0) out of news of {p.MPCshocks[3]} shock in 1 quarter(s)'],
						mpcNewsSimulator_shockNextPeriod.results[f'Median(Q1 MPC | MPC > 0) out of news of {p.MPCshocks[5]} shock in 1 quarter(s)'],
						np.nan,
						mpcNewsSimulator_shockNextPeriod.results[f'Median(Q1 MPC | MPC > 0) out of news of {p.MPCshocks[2]} shock in 1 quarter(s)'],
						mpcNewsSimulator_shock2Years.results[f'Median(Q1 MPC | MPC > 0) out of news of {p.MPCshocks[2]} shock in 8 quarter(s)'],
						np.nan,
						mpcNewsSimulator_loan.results[f'Median(Q1 MPC | MPC > 0) out of {-p.MPCshocks[0]} loan for 4 quarter(s)'],
					]


	i = 0
	cols = [mean_mpc, neg_mpc_share, zero_mpc_share, pos_mpc_share, mean_cond_mpc, med_cond_mpc]
	cols_series = []
	for col in cols:
		cols_series.append(pd.Series(col, index=index, name=colnames[i]))
		i += 1

	df = pd.concat(cols_series, axis=1)

	return df
	