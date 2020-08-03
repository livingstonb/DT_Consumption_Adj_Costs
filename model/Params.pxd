
cdef class Params:
	cdef:
		public str name
		public int index, freq, nyP, nyT, nz
		public long maxIters, nSim, tSim
		public bint MPCsOutOfNews, Bequests, fastSettings
		public bint noTransIncome, noPersIncome
		public long NsimMPC
		public list MPCshocks, wealthConstraints, wealthPercentiles
		public int xMax, nx, nSectionsGSS
		public double xGridTerm1Wt, xGridTerm1Curv
		public double xGridCurv, borrowLim
		public int nc, nshocks, n_discountFactor, n_riskAver
		public double cMin, cMax, cGridCurv
		public double cGridTerm1Wt, cGridTerm1Curv
		public double r, R, deathProb, govTransfer, tol
		public double riskAver, adjustCost, timeDiscount
		public object risk_aver_grid, discount_factor_grid
		public object series
		public dict cal1_options, cal2_options
