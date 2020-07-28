import sys
import main

#---------------------------------------------------------------#
#      FUNCTIONS                                                #
#---------------------------------------------------------------#
def set_from_cmd_arg(cmd_line_args):
	"""
	Returns the first integer-valued command-line
	argument passed, if available.
	"""
	for arg in cmd_line_args:
		try:
			return int(arg)
		except:
			pass

	return None

#---------------------------------------------------------------#
#      CHOOSE RUN OPTIONS                                       #
#---------------------------------------------------------------#
runOptions = dict()
runOptions['Calibrate'] = False # not for replication materials
runOptions['Simulate'] = True
runOptions['SimulateMPCs'] = True
runOptions['MPCsNews'] = True
runOptions['Fast'] = False # run w/small grids for debugging
runOptions['PrintGrids'] = False
runOptions['MakePlots'] = False

#---------------------------------------------------------------#
#      CHOOSE REPLICATION                                       #
#---------------------------------------------------------------#
# If desired, choose a replication experiment
# Redefine replication = None to ignore
replication = dict()
replication['target'] = 'mean_wealth' # Either 'mean_wealth' or 'wealth_lt_1000'
replication['adjustCostOn'] = True # True or False
replication['betaHeterogeneity'] = True # True or False
# replication = None

#---------------------------------------------------------------#
#      OR CHOOSE AN INDEX                                       #
#---------------------------------------------------------------#
# Otherwise, choose a parameter number (overrided by the above)
paramIndex = 0

#---------------------------------------------------------------#
#      HOUSEKEEPING                                             #
#---------------------------------------------------------------#
indexPassed = set_from_cmd_arg(sys.argv)

if indexPassed is not None:
	paramIndex = indexPassed
	replication = None

main.main(paramIndex=paramIndex,
	runopts=runOptions, replication=replication)