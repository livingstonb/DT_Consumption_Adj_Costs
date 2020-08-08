import sys
import main

#---------------------------------------------------------------#
#      FUNCTIONS                                                #
#---------------------------------------------------------------#
def set_from_cmd_arg(cmd_line_args):
	"""
	Returns the first integer-valued command-line argument passed,
	if available. Used to run code on the server.
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
runOptions['PrintGrids'] = False # prints the grids and terminates
runOptions['MakePlots'] = False # may be out of date

#---------------------------------------------------------------#
#      CHOOSE REPLICATION                                       #
#---------------------------------------------------------------#
# If desired, choose a replication experiment
# Redefine replication = None to ignore
replication = dict()
replication['mode'] = 'mean_wealth' # 'mean_wealth', 'wealth_lt_1000', 'beta_het'
replication['adjustCostOn'] = True # True or False
# replication = None

#---------------------------------------------------------------#
#      OR CHOOSE AN INDEX                                       #
#---------------------------------------------------------------#
# Otherwise, choose a parameterization by number
# Overridden by selecting a replication or passing an argument from
# the command line
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