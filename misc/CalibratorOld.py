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
		self.weights = cal_options['weights']
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

		self.constraints = []
		if 'sbounds' in cal_options:
			for ivar in range(self.nvars):
				sbounds = cal_options['sbounds'][ivar]
				hbounds = cal_options['bounds'][ivar]
				newConstraint = Constraint(sbounds, hbounds)
				self.constraints.append(newConstraint)

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
			self.solver_kwargs['method'] = 'L-BFGS-B'
			self.solver_kwargs['options'] = {
				'gtol': 1.0e-4,
				'ftol': 1.0e-7,
			}
		elif self.solver == 'least_squares':
			self.solver_kwargs['bounds'] = self.xbounds
			# self.solver_kwargs['method'] = 'dogbox'
			self.solver_kwargs['verbose'] = 1
			# self.solver_kwargs['diff_step'] = 1e-7
			self.solver_kwargs['gtol'] = None
			self.solver_kwargs['loss'] = 'soft_l1'
		elif self.solver == 'differential_evolution':
			self.solver_kwargs['bounds'] = self.xbounds_BoundsObj
		elif self.solver == 'root':
			pass

	def calibrate(self, p, model, income, grids):
		self.p = p
		self.model = model
		self.income = income
		self.grids = grids

		requires_x0 = [
			'minimize_scalar', 'least_squares',
			'minimize', 'root',
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
		elif self.solver == 'root':
			scipy_solver = optimize.root

		opt_results = scipy_solver(*solver_args,
			**self.solver_kwargs)

		return opt_results

	def optim_handle(self, x_scaled):
		x = self.unscale(x_scaled)

		z = []
		for ivar in range(self.nvars):
			if self.solver == 'root':
				x[ivar], tmp = self.constraints[ivar].check_value(x[ivar])
				z.append(tmp)

			var = self.variables[ivar]
			vchange = x[ivar] - self.p.getParam(var)
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

		yvals = np.zeros(self.nvars) # + len(z)
		values = [None] * self.nvars
		for ivar in range(self.nvars):
			target = self.target_names[ivar]

			if self.target_types[ivar] == 'Equilibrium':
				values[ivar] = eqSimulator.results[target]
			elif self.target_types[ivar] == 'MPC':
				values[ivar] = mpcSimulator.results[target]

			yvals[ivar] = self.weights[ivar] * (values[ivar] - self.target_values[ivar])

		# ii = self.nvars
		# for ivar in range(self.nvars):
		# 	if self.solver == 'root':
		# 		yvals[ii] = z[ivar]
		# 		ii += 1

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
			'root',
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
			self.transform_y = lambda x: np.linalg.norm(x)
		elif self.solver in take_abs:
			self.transform_y = lambda x: np.abs(x)

class Constraint:
	def __init__(self, sbounds, hbounds):
		self.sbounds = sbounds
		self.hbounds = hbounds

	def check_value(self, x):
		dl = max(self.sbounds[0] - x, 0)
		dh = max(x - self.sbounds[1], 0)

		z = max(dl, dh)
		devz = np.exp(-z)

		if z == 0:
			x_sc = x
		elif dl > 0:
			# Lower bound violated
			x_s = devz * self.sbounds[0]
			x_h = (1 - devz) * self.hbounds[0]
			x_sc = x_s + x_h
			z = -z
		elif dh > 0:
			# Upper bound violated
			x_s = devz * self.sbounds[1]
			x_h = (1 - devz) * self.hbounds[1]
			x_sc = x_s + x_h

		return (x_sc, z)

# class Options:
# 	def __init__(self):
# 		self.name = name
# 		self.bounds = bounds
# 		self.lb = bounds[0]
# 		self.ub = bounds[1]
# 		self.v0 = v0

# class OptimVariable:
# 	def __init__(self, name, bounds, v0):
# 		self.name = name
# 		self.bounds = bounds
# 		self.lb = bounds[0]
# 		self.ub = bounds[1]
# 		self.v0 = v0
