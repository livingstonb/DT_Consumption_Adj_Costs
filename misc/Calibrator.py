import numpy as np
from scipy import optimize
from model import simulator

class Calibrator:
	def __init__(self, cal_options):

		self.variables = cal_options['variables']
		self.target_names = cal_options['target_names']
		self.target_values = cal_options['target_values']
		self.solver = cal_options['solver']
		self.nvars = len(self.variables)

		self.solver_kwargs = dict()

		self.set_bounds(cal_options)

		lbounds = []
		ubounds = []
		for ivar in range(self.nvars):

			bound = cal_options['bounds'][ivar]
			lbounds.append(bound[0])
			ubounds.append(bound[1])

		self.xbounds = optimize.Bounds(
			lbounds, ubounds, keep_feasible=True)

		self.set_solver_options(cal_options)
		self.set_target_transformation()

	def set_solver_options(self, cal_options):
		bounds = cal_options['bounds']

		if self.solver == 'root_scalar':
			self.solver_kwargs['bracket'] = bounds[0]
			self.solver_kwargs['method'] = 'brentq'
			self.solver_kwargs['xtol'] = 1e-7
			self.solver_kwargs['rtol'] = 1e-9
		elif self.solver == 'minimize_scalar':
			self.solver_kwargs['bounds'] = bounds[0]
			self.solver_kwargs['method'] = 'bounded'
		elif self.solver == 'minimize':
			self.solver_kwargs['bounds'] = self.xbounds
			self.solver_kwargs['method'] = 'SLSQP'
		elif self.solver == 'least_squares':
			self.solver_kwargs['bounds'] = self.xbounds
			self.solver_kwargs['method'] = 'dogbox'
			self.solver_kwargs['verbose'] = 1
			self.solver_kwargs['diff_step'] = 1e-5
		elif self.solver == 'differential_evolution':
			self.solver_kwargs['bounds'] = self.xbounds

	def optimize(self, p, model):
		self.p = p
		self.model = model

		requires_x0 = [
			'minimize_scalar', 'least_squares',
			'minimize',
		]

		solver_args = [self.optim_handle]
		if self.solver in requires_x0:
			x0 = np.zeros(self.nvars)
			for ivar in range(self.nvars):
				var = self.variables(ivar)
				x0[ivar] = p.getParam(var)

			solver_args.append(x0)

		if self.solver == 'root_scalar'
			scipy_solver = optimize.root_scalar
		elif self.solver == 'minimize_scalar'
			scipy_solver = optimize.minimize_scalar
		elif self.solver == 'minimize':
			scipy_solver = optimize.minimize
		elif self.solver == 'least_squares':
			scipy_solver = optimize.least_squares
		elif self.solver == 'differential_evolution':
			scipy_solver = optimize.differential_evolution

		opt_results = scipy_solver(*solver_args,
			**self.solver_kwargs)

		return opt_results

	def optim_handle(self, x):
		for ivar in range(self.nvars):
			var = self.variables(ivar)
			if var == 'timeDiscount':
				self.p.resetDiscountRate(x[ivar])
			elif var == 'adjustCost':
				self.p.resetAdjustCost(x[ivar])
			else:
				self.p.setParam(var, x[ivar])

		self.model.solve()

		eqSimulator = simulator.EquilibriumSimulator(
			params, income, grids, model.cSwitchingPolicy,
			model.valueDiff)
		eqSimulator.simulate()

		yvals = np.zeros(self.nvars)
		for it in range(self.nvars):
			target = self.target_names[it]
			yvals[ivar] = eqSimulator.results[target] \
				- self.target_values[it]

		return self.transform_y(yvals)

	def set_target_transformation(self):
		leave_unchanged = [
			'root_scalar', 'least_squares',
		]

		take_norm = [
			'minimize_scalar', 'minimize',
		]

		take_abs = [
			'differential_evoluation',
		]

		if self.solver in leave_unchanged:
			self.transform_y = lambda x: x
		elif self.solver in take_norm:
			self.transform_y = lambda x: np.norm(x)
		elif self.solver in take_abs:
			self.transform_y = lambda x: np.abs(x)