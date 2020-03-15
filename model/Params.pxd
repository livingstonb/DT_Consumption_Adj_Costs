
cdef class Params:
	"""
	This class stores the parameters of the model.

	The default values are set in the class constructor
	and to override these values, a dictionary is passed
	into the constructor containing parameter values
	that need to be overriden (e.g. {'tSim':200}).

	Variables such as returns and the probability of
	death must be set on an annual basis. The method
	adjustToQuarterly() adjusts these variables to
	quarterly values.
	"""
	cdef:
		public str name, locIncomeProcess
		public int index, freq, nyP, nyT, nz
		public long maxIters, nSim, tSim
		public bint MPCsOutOfNews, Bequests, fastSettings
		public bint noTransIncome, noPersIncome
		public long NsimMPC
		public list MPCshocks, wealthConstraints, wealthPercentiles
		public int xMax, nx
		public double xGridTerm1Wt, xGridTerm1Curv
		public double xGridCurv, borrowLim
		public int nc, nshocks, n_discountFactor, n_riskAver
		public double cMin, cMax, cGridCurv
		public double cGridTerm1Wt, cGridTerm1Curv
		public double r, R, deathProb, govTransfer, tol
		public double riskAver, adjustCost, timeDiscount
		public object risk_aver_grid, discount_factor_grid
		public object series, cal_options
