import numpy as np
from scipy import optimize
from model import simulator
from misc import functions

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

	def set_x0(self):
		if self.solverOpts.solver == 'root_scalar':
			self.x0 = [self.variables[0].x0, self.variables[0].x1]
		elif self.solverOpts.requiresInitialCond:
			x0 = [self.variables[i].x0 for i in range(self.nvars)]
			self.x0 = np.array(x0)
		else:
			self.x0 = None

	def set_bounds(self):
		lbounds = [self.variables[i].lb for i in range(self.nvars)]
		ubounds = [self.variables[i].ub for i in range(self.nvars)]

		if self.solverOpts.solver  == 'minimize':
			self.scipy_kwargs['bounds'] = optimize.Bounds(
				lbounds, ubounds, keep_feasible=True)
		elif self.solverOpts.solver == 'least_squares':
			self.scipy_kwargs['bounds'] = (lbounds, ubounds)
		elif self.solverOpts.solver == 'root_scalar':
			self.scipy_kwargs['bracket'] = self.variables[0].bracket
			self.scipy_kwargs['x0'] = self.x0[0]
			self.scipy_kwargs['x1'] = self.x0[1]
		elif self.solverOpts.solver == 'minimize_scalar':
			self.scipy_kwargs['bounds'] = self.variables[0].bracket

	def calibrate(self, p, model, income, grids):
		self.p = p
		self.model = model
		self.income = income
		self.grids = grids

		scipy_args = [self.optim_handle]
		if (self.x0 is not None) and (self.solverOpts.solver != 'root_scalar'):
			scipy_args.append(self.x0)

		if self.solverOpts.solver == 'minimize':
			scipy_solver = optimize.minimize
		elif self.solverOpts.solver == 'least_squares':
			scipy_solver = optimize.least_squares
		elif self.solverOpts.solver == 'root_scalar':
			scipy_solver = optimize.root_scalar
		elif self.solverOpts.solver == 'minimize_scalar':
			scipy_solver = optimize.minimize_scalar

		opt_results = scipy_solver(
			*scipy_args,
			**self.scipy_kwargs)

		return opt_results

	def optim_handle(self, x_scaled):
		x = np.copy(x_scaled)
		if np.ndim(x) == 0:
			x = np.array([x])

		for ivar in range(self.nvars):
			x[ivar] = self.variables[ivar].unscale(x[ivar])

		for ivar in range(self.nvars):
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
			self.p, self.income, self.grids)
		eqSimulator.initialize(self.model.cSwitchingPolicy,
			self.model.inactionRegion)
		eqSimulator.simulate()

		if 'MPC' in self.target_types:
			shockIndices = [3]

			mpcSimulator = simulator.MPCSimulator(
				self.p, self.income, self.grids,
				shockIndices)
			mpcSimulator.initialize(self.model.cSwitchingPolicy,
				self.model.inactionRegion, eqSimulator.finalStates)
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

class OptimVariable:
	def __init__(self, name, bounds, x0, x1=None, scale=1.0):
		self.name = name
		self.xscale = scale
		self.x0 = self.scale(x0)
		self.x1 = self.scale(x1)
		self.lb = self.scale(bounds[0])
		self.ub = self.scale(bounds[1])
		self.bracket = [self.lb, self.ub]

	def scale(self, x):
		if x is not None:
			return x * self.xscale
		else:
			return None

	def unscale(self, x_scaled):
		return x_scaled / self.xscale

class OptimTarget:
	def __init__(self, name, value, target_type, weight=1.0):
		self.name = name
		self.value = value
		self.type = target_type
		self.weight = weight

class SolverOptions:
	def __init__(self, solver, solver_kwargs=None,
		other_opts=None):
		self.solver = solver

		if other_opts is not None:
			self.set_other_opts(other_opts)

		if solver_kwargs is None:
			self.solver_kwargs = self.default_kwargs()
		else:
			self.solver_kwargs = solver_kwargs

		self.requiresInitialCond = True
		self.set_target_transformation()

	def set_other_opts(self, other_opts):
		self.other_opts = {
			'norm_deg': 2,
			'norm_raise_to': 1,
		}
		self.other_opts.update(other_opts)

	def default_kwargs(self):
		solver_kwargs = dict()
		solver_kwargs['options'] = {'disp': True}

		if self.solver == 'minimize':
			solver_kwargs['method'] = 'L-BFGS-B'
			solver_kwargs['options'].update(
				{
				'eps': 2.0e-6,
				'maxiter': 100,
				}
			)
		elif self.solver == 'least_squares':
			solver_kwargs['verbose'] = 1
			solver_kwargs['gtol'] = None
		elif self.solver == 'root_scalar':
			solver_kwargs['method'] = 'secant'
		elif self.solver == 'minimize_scalar':
			solver_kwargs['method'] = 'bounded'

		return solver_kwargs

	def set_target_transformation(self):
		if self.solver == 'least_squares':
			self.transform_y = lambda x: x
		elif self.solver == 'minimize':
			ndg = self.other_opts['norm_deg']
			npow = self.other_opts['norm_raise_to']
			self.transform_y = lambda x: np.power(
				np.linalg.norm(x, ndg), npow)
		elif self.solver == 'root_scalar':
			self.transform_y = lambda x: x
		elif self.solver == 'minimize_scalar':
			self.transform_y = lambda x: np.linalg.norm(x)