import numpy as np
from scipy import optimize
from model import simulator
from misc import functions
from IPython.core.debugger import set_trace

class Calibrator:
	def __init__(self, cal_options):

		self.variables = cal_options['variables']
		self.target_names = cal_options['target_names']
		self.target_values = cal_options['target_values']
		self.target_types = cal_options['target_types']
		self.solver = cal_options['solver']
		self.scale = cal_options['scale']
		self.nvars = len(self.variables)
		self.iteration = 0

		self.solver_kwargs = dict()

		self.set_bounds(cal_options)
		self.set_solver_options(cal_options)
		self.set_target_transformation()

	def scale_for_solver(self, x):
		x_scaled = np.zeros(self.nvars)
		for ivar in range(self.nvars):
			x_scaled[ivar] = x[ivar] * self.scale[ivar]

		return x_scaled

	def unscale(self, x_scaled):
		x = np.zeros(self.nvars)
		for ivar in range(self.nvars):
			x[ivar] = x_scaled[ivar] / self.scale[ivar]

		return x

	def set_bounds(self, cal_options):
		lbounds = []
		ubounds = []
		for ivar in range(self.nvars):
			bound = cal_options['bounds'][ivar]
			lbounds.append(bound[0])
			ubounds.append(bound[1])

		lbounds = self.scale_for_solver(lbounds)
		ubounds = self.scale_for_solver(ubounds)

		self.xbounds_BoundsObj = optimize.Bounds(
			lbounds, ubounds, keep_feasible=True)
		self.xbounds = (lbounds, ubounds)

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
			self.solver_kwargs['bounds'] = self.xbounds_BoundsObj
			self.solver_kwargs['method'] = 'SLSQP'
		elif self.solver == 'least_squares':
			self.solver_kwargs['bounds'] = self.xbounds
			self.solver_kwargs['method'] = 'dogbox'
			self.solver_kwargs['verbose'] = 1
			self.solver_kwargs['diff_step'] = 1e-5
			self.solver_kwargs['gtol'] = None
		elif self.solver == 'differential_evolution':
			self.solver_kwargs['bounds'] = self.xbounds_BoundsObj

	def calibrate(self, p, model, income, grids):
		self.p = p
		self.model = model
		self.income = income
		self.grids = grids

		requires_x0 = [
			'minimize_scalar', 'least_squares',
			'minimize',
		]

		solver_args = [self.optim_handle]
		if self.solver in requires_x0:
			x0 = np.zeros(self.nvars)
			for ivar in range(self.nvars):
				var = self.variables[ivar]
				x0[ivar] = p.getParam(var)

			x0 = self.scale_for_solver(x0)
			solver_args.append(x0)

		if self.solver == 'root_scalar':
			scipy_solver = optimize.root_scalar
		elif self.solver == 'minimize_scalar':
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

	def optim_handle(self, x_scaled):
		x = self.unscale(x_scaled)
		for ivar in range(self.nvars):
			var = self.variables[ivar]
			vchange = self.p.getParam(var) - x[ivar]
			self.p.setParam(var, x[ivar])

			if self.iteration > 0:
				iterStr = f'For iteration {self.iteration}'
				if vchange == 0:
					print(f'{iterStr}, {var} was not changed')
				elif vchange > 0:
					print(f'{iterStr}, {var} was increased by {vchange}')
				else:
					print(f'{iterStr}, {var} was decreased by {np.abs(vchange)}')

		if self.iteration == 0:
			functions.printLine()
			print('Beginning calibration')
			functions.printLine()

		self.model.solve()

		eqSimulator = simulator.EquilibriumSimulator(
			self.p, self.income, self.grids,
			self.model.cSwitchingPolicy,
			self.model.inactionRegion)
		eqSimulator.simulate()

		if 'MPC' in self.target_types:
			shockIndices = [3]

			mpcSimulator = simulator.MPCSimulator(
				self.p, self.income, self.grids, 
				self.model.cSwitchingPolicy,
				self.model.inactionRegion, 
				shockIndices, eqSimulator.finalStates)

			mpcSimulator.simulate()

		yvals = np.zeros(self.nvars)
		values = [None] * self.nvars
		for it in range(self.nvars):
			target = self.target_names[it]

			if self.target_types[it] == 'Equilibrium':
				values[it] = eqSimulator.results[target]
			elif self.target_types[it] == 'MPC':
				values[it] = mpcSimulator.results[target]

			yvals[ivar] = values[it] - self.target_values[it]

		self.printIterationResults(x, values)
		self.iteration += 1
		return self.transform_y(yvals)

	def printIterationResults(self, x, values):
		functions.printLine()

		print('\nAt this solver iteration, parameters were:')
		for ivar in range(self.nvars):
			var = self.variables[ivar]
			print(f'\t{var}\n\t\t= {x[ivar]}')

		print('\nResults were:')
		for ivar in range(self.nvars):
			target = self.target_names[ivar]
			target_val = self.target_values[ivar]
			print(f'\t{target}\n\t\t= {values[ivar]} (desired = {target_val})')

		functions.printLine()

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