import numpy as np
from scipy import optimize
from model import simulator
from misc import functions

class Calibrator:
	def __init__(self, p, model, income, grids):
		self.p = p
		self.model = model
		self.income = income
		self.grids = grids
		self.step = None
		self.ftol = 1.0e-5

	def calibrate(self):
		bracket = [self.lbounds[0], self.ubounds[0]]

		optimize.root_scalar(
			self.optim_handle, x0=self.x0, bracket=bracket,
			)

	def simulate(self):
		self.model.solve()

		eqSimulator = simulator.EquilibriumSimulator(
			self.p, self.income, self.grids)
		eqSimulator.initialize(self.model.cSwitchingPolicy,
			self.model.inactionRegion)
		eqSimulator.simulate()

		return eqSimulator

class Calibrator1(Calibrator):
	def __init__(self, p, model, income, grids):
		self.lbounds = [0.96]
		self.ubounds = [0.999]
		self.x0 = np.array([0.99])

		super().__init__(p, model, income, grids)

	def optim_handle(self, x):
		self.p.setParam('timeDiscount', x, True)

		eqSimulator = self.simulate()

		z = eqSimulator.results['Mean wealth'] - 3.2
		return z

class Calibrator2(Calibrator):
	def __init__(self, p, model, income, grids):
		self.lbounds = [0.95]
		self.ubounds = [0.99]
		self.x0 = np.array([0.97])
		
		super().__init__(p, model, income, grids)
		self.step = np.array([0.00002])

	def optim_handle(self, x):
		self.p.setParam('timeDiscount', x, True)

		eqSimulator = self.simulate()

		z = eqSimulator.results['Wealth <= $1000'] - 0.23
		print(f'Wealth constrained = {z + 0.23}')
		return z

class Calibrator3(Calibrator):
	def __init__(self, p, model, income, grids):
		self.lbounds = [0.96, 0.01]
		self.ubounds = [0.9995, 0.05]
		self.x0 = np.array([0.9985, 0.032])
		
		super().__init__(p, model, income, grids)
		self.step = np.array([0.00002, 0.00002])

	def optim_handle(self, x):
		# self.p.setParam('timeDiscount', x[0], True)
		# self.p.setParam('discount_factor_grid', np.array([x[0]- 2 * x[1], x[0] - x[1], x[0]]), True)
		# self.p.setParam('timeDiscount', x[2], True)
		self.p.setParam('discount_factor_grid', np.array([x[0], x[1], x[2]]), True)
		self.model.p = self.p

		eqSimulator = self.simulate()

		z = np.zeros(2)
		z[0] = eqSimulator.results['Mean wealth'] - 3.2
		z[1] = eqSimulator.results['Wealth <= $1000'] - 0.23

		print(f'Mean wealth = {z[0] + 3.2}')
		print(f'Wealth constrained = {z[1] + 0.23}')

		return np.linalg.norm(z)

	def calibrate(self):
		A1 = np.array([[1, -2, 1]])
		lb1 = np.array([0.0])
		ub1 = np.array([0.0])
		constraint1 = optimize.LinearConstraint(A1, lb1, ub1)

		A2 = np.array([[0, 0, 1], [1, 0, 0]])
		lb2 = np.array([0.998, 0.932])
		ub2 = np.array([0.9986, np.inf])
		constraint2 = optimize.LinearConstraint(A2, lb2, ub2, keep_feasible=True)

		optimize.minimize(self.optim_handle, self.x0, method='trust-constr',
			constraints=[constraint1, constraint2])

class Calibrator4(Calibrator):
	def __init__(self, p, model, income, grids):
		self.lbounds = [1e-6]
		self.ubounds = [2e-3]
		self.x0 = np.array([1e-4])

		super().__init__(p, model, income, grids)
		self.step = np.array([2.5e-6])
		self.ftol = 1.0e-7

	def optim_handle(self, x):
		self.p.setParam('adjustCost', x, True)
		self.model.p = self.p

		eqSimulator = self.simulate()

		shockIndices = [3]
		mpcSimulator = simulator.MPCSimulator(
			self.p, self.income, self.grids, shockIndices)
		mpcSimulator.initialize(self.model.cSwitchingPolicy,
			self.model.inactionRegion, eqSimulator.finalStates)
		mpcSimulator.simulate()

		targeted_stat = f'P(Q1 MPC > 0) for shock of 0.0081'
		z = mpcSimulator.results[targeted_stat] - 0.2

		print(f'P(MPC > 0) = {z + 0.2}')

		return z