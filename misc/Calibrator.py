import numpy as np
from scipy import optimize
from model import simulator
from misc import functions
from IPython.core.debugger import set_trace

class Calibrator:
	def __init__(self, variables, targets, solverOpts):

		self.nvars = len(variables)
		self.variables = variables
		self.targets = targets
		self.target_types = [targets[i].type for i in range(self.nvars)]
		self.solverOpts = solverOpts
		self.scipy_kwargs = solverOpts.solver_kwargs
		self.iteration = 0

		self.set_x0()
		self.set_bounds()

	# 	self.constraints = []
	# 	if 'sbounds' in cal_options:
	# 		for ivar in range(self.nvars):
	# 			sbounds = cal_options['sbounds'][ivar]
	# 			hbounds = cal_options['bounds'][ivar]
	# 			newConstraint = Constraint(sbounds, hbounds)
	# 			self.constraints.append(newConstraint)

	# 	return x

	def set_x0(self):
		if self.solverOpts.requiresInitialCond:
			x0 = [self.variables[i].x0 for i in range(self.nvars)]
			self.x0 = np.array(x0)
		else:
			self.x0 = None

	def set_bounds(self):
		requiresBoundsObj = [
			'minimize',
			'differential_evolution',
		]

		requiresPair = [
			'least_squares',
		]

		requiresBracket = [
			'root_scalar',
		]

		lbounds = [self.variables[i].lb for i in range(self.nvars)]
		ubounds = [self.variables[i].ub for i in range(self.nvars)]

		if self.solverOpts.solver in requiresBoundsObj:
			self.scipy_kwargs['bounds'] = optimize.Bounds(
				lbounds, ubounds, keep_feasible=True)
		elif self.solverOpts.solver in requiresPair:
			self.scipy_kwargs['bounds'] = (lbounds, ubounds)
		elif self.solverOpts.solver in requiresBracket:
			bracket = [self.variables[i].bracket for i in range(self.nvars)]
			self.scipy_kwargs['bracket'] = bracket

	def calibrate(self, p, model, income, grids):
		self.p = p
		self.model = model
		self.income = income
		self.grids = grids

		scipy_args = [self.optim_handle]
		if self.x0 is not None:
			scipy_args.append(self.x0)

		if self.solverOpts.solver == 'root_scalar':
			scipy_solver = optimize.root_scalar
		elif self.solverOpts.solver == 'minimize_scalar':
			scipy_solver = optimize.minimize_scalar
		elif self.solverOpts.solver == 'minimize':
			scipy_solver = optimize.minimize
		elif self.solverOpts.solver == 'least_squares':
			scipy_solver = optimize.least_squares
		elif self.solverOpts.solver == 'differential_evolution':
			scipy_solver = optimize.differential_evolution
		elif self.solverOpts.solver == 'root':
			scipy_solver = optimize.root

		opt_results = scipy_solver(
			*scipy_args,
			**self.scipy_kwargs)

		return opt_results

	def optim_handle(self, x_scaled):
		x = np.copy(x_scaled)
		for ivar in range(self.nvars):
			x[ivar] = self.variables[ivar].unscale(x_scaled[ivar])

		z = []
		for ivar in range(self.nvars):
			# if self.solver == 'root':
			# 	x[ivar], tmp = self.constraints[ivar].check_value(x[ivar])
			# 	z.append(tmp)

			var = self.variables[ivar].name
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
			target = self.targets[ivar].name

			if self.target_types[ivar] == 'Equilibrium':
				values[ivar] = eqSimulator.results[target]
			elif self.target_types[ivar] == 'MPC':
				values[ivar] = mpcSimulator.results[target]

			dy = values[ivar] - self.targets[ivar].value
			yvals[ivar] = dy * self.targets[ivar].weight

		# ii = self.nvars
		# for ivar in range(self.nvars):
		# 	if self.solver == 'root':
		# 		yvals[ii] = z[ivar]
		# 		ii += 1

		self.printIterationResults(x, values)
		self.iteration += 1
		return self.solverOpts.transform_y(yvals)

	def printIterationResults(self, x, values):
		functions.printLine()

		print('\nAt this solver iteration, parameters were:')
		for ivar in range(self.nvars):
			var = self.variables[ivar].name
			print(f'\t{var}\n\t\t= {x[ivar]}')

		print('\nResults were:')
		for ivar in range(self.nvars):
			target = self.targets[ivar].name
			target_val = self.targets[ivar].value
			print(f'\t{target}\n\t\t= {values[ivar]} (desired = {target_val})')

		functions.printLine()

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

class OptimVariable:
	def __init__(self, name, bounds, x0, scale=1.0):
		self.name = name
		self.bounds = bounds
		self.bracket = bounds
		self.lb = bounds[0]
		self.ub = bounds[1]
		self.xscale = scale
		self.x0 = self.scale(x0)

	def scale(self, x):
		return x * self.xscale

	def unscale(self, x_scaled):
		return x_scaled / self.xscale

class OptimTarget:
	def __init__(self, name, value, target_type, weight=1.0):
		self.name = name
		self.value = value
		self.type = target_type
		self.weight = weight

class SolverOptions:
	def __init__(self, solver, solver_kwargs=None):
		self.solver = solver

		if solver_kwargs is None:
			self.solver_kwargs = self.default_kwargs()
		else:
			self.solver_kwargs = solver_kwargs

		self.requiresInitialCond = self.checkIfInitialCondNeeded()
		self.set_target_transformation()

	def default_kwargs(self):
		solver_kwargs = dict()
		solver_kwargs['options'] = {'disp': True}

		if self.solver == 'root_scalar':
			solver_kwargs['method'] = 'brentq'
			solver_kwargs['xtol'] = 1e-7
			solver_kwargs['rtol'] = 1e-9
		elif self.solver == 'minimize_scalar':
			solver_kwargs['method'] = 'bounded'
		elif self.solver == 'minimize':
			solver_kwargs['method'] = 'L-BFGS-B'
			solver_kwargs['options'].update(
				{
				'eps': 2.0e-7,
				'maxiter': 50,
				# 'gtol': 2.0e-5,
				# 'ftol': 1.0e-7,
				}
			)
		elif self.solver == 'least_squares':
			solver_kwargs['verbose'] = 1
			solver_kwargs['gtol'] = None
		elif self.solver == 'differential_evolution':
			pass
		elif self.solver == 'root':
			pass

		return solver_kwargs

	def checkIfInitialCondNeeded(self):
		requires_x0 = [
			'minimize_scalar',
			'least_squares',
			'minimize',
			'root',
		]
		return (self.solver in requires_x0)

	def set_target_transformation(self):
		leave_unchanged = [
			'root_scalar',
			'least_squares',
			'root',
		]

		take_norm = [
			'minimize_scalar',
			'minimize',
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