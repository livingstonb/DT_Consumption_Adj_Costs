import numpy as np
import pandas as pd
import os

def saveWealthGroupStats(mpcSimulator, mpcNewsSimulator_shockNextPeriod,
	mpcNewsSimulator_shock2Years, mpcNewsSimulator_loan, finalSimStates,
	outdir, paramIndex, params):

	mpcs_over_states = dict()
	mpcs_over_states['$500 GAIN'] = mpcSimulator.mpcs[3]
	mpcs_over_states['$2500 GAIN'] = mpcSimulator.mpcs[4]
	mpcs_over_states['$5000 GAIN'] = mpcSimulator.mpcs[5]
	mpcs_over_states['$500 LOSS'] = mpcSimulator.mpcs[2]
	mpcs_over_states['$500 NEWS-GAIN'] = mpcNewsSimulator_shockNextPeriod.mpcs[3]
	mpcs_over_states['$5000 NEWS-GAIN'] = mpcNewsSimulator_shockNextPeriod.mpcs[5]
	mpcs_over_states['$500 NEWS-LOSS'] = mpcNewsSimulator_shockNextPeriod.mpcs[2]
	mpcs_over_states['$500 NEWS-LOSS IN 2 YEARS'] = mpcNewsSimulator_shock2Years.mpcs[2]
	mpcs_over_states['$5000 LOAN'] = mpcNewsSimulator_loan.mpcs[0]

	index = []
	treatmentResponses = pd.DataFrame()
	for pair in combinations(mpcs_over_states.keys(), 2):
		key = pair[0] + ', ' + pair[1]
		index.append(key)
		thisTreatmentPair = {
			'Response to 1 only' : ((mpcs_over_states[pair[0]] > 0)  & (mpcs_over_states[pair[1]] == 0) ).mean(),
			'Response to 2 only' : ((mpcs_over_states[pair[0]] == 0)  & (mpcs_over_states[pair[1]] > 0) ).mean(),
			'Response to both' : ((mpcs_over_states[pair[0]] > 0)  & (mpcs_over_states[pair[1]] > 0) ).mean(),
			'Response to neither' : ((mpcs_over_states[pair[0]] == 0)  & (mpcs_over_states[pair[1]] == 0) ).mean(),
		}
		treatmentResponses = treatmentResponses.append(thisTreatmentPair, ignore_index=True)

	treatmentResponses.index = index
	savepath = os.path.join(outdir, f'run{paramIndex}_treatment_responses.csv')
	# treatmentResponses.to_excel(savepath, freeze_panes=(0,0), index_label=params.name)
	treatmentResponses.to_csv(savepath, index_label=params.name)

	# find fractions responding in certain wealth groups
	group1 = np.asarray(finalSimStates['asim']) <= 0.081
	group2 = (np.asarray(finalSimStates['asim']) > 0.081) & (np.asarray(finalSimStates['asim'])<= 0.486)
	group3 = (np.asarray(finalSimStates['asim']) > 0.486) & (np.asarray(finalSimStates['asim']) <= 4.05)
	group4 = (np.asarray(finalSimStates['asim']) > 4.05)

	groups = [group1, group2, group3, group4]

	groupLabels = [
		'$0-$5000',
		'$5000-$30,000',
		'$30,000-$250,000',
		'$250,000+',
	]

	treatments = [
		('$500 GAIN', '$500 NEWS-GAIN'),
		('$5000 GAIN', '$5000 NEWS-GAIN'),
		('$500 GAIN', '$500 LOSS'),
		('$500 LOSS', '$500 NEWS-LOSS'),	
	]

	# loop over income groups
	for i in range(4):	
		index = []
		treatmentResults = []

		for pair in treatments:
			thisTreatmentPair = dict()
			
			index.append(pair[0] + ', ' + pair[1])

			mpcs_treatment1 = mpcs_over_states[pair[0]][groups[i].flatten()]
			mpcs_treatment2 = mpcs_over_states[pair[1]][groups[i].flatten()]
			thisTreatmentPair['Response to 1 only'] =  (
				(mpcs_treatment1 > 0) & (mpcs_treatment2 == 0)).mean()
			thisTreatmentPair['Response to 2 only'] =  (
				(mpcs_treatment1 == 0) & (mpcs_treatment2 > 0)).mean()
			thisTreatmentPair['Response to both'] = (
				(mpcs_treatment1 > 0) & (mpcs_treatment2 > 0)).mean()
			thisTreatmentPair['Response to neither'] = (
				(mpcs_treatment1 == 0) & (mpcs_treatment2 == 0)).mean()

			treatmentResults.append(thisTreatmentPair)
		
		thisGroup = pd.DataFrame(data=treatmentResults, index=index)
		savepath = os.path.join(outdir, f'run{paramIndex}_wealthgroup{i+1}_responses.csv')
		# thisGroup.to_excel(savepath, freeze_panes=(0,0), index_label=params.name)
		thisGroup.to_csv(savepath, index_label=params.name)