import numpy as np
from scipy import optimize
from model import simulator
from misc import functions

# def generate_Calibrator(index, p, model, income, grids):
# 	if index == 

class Calibrator:
	def __init__(self, p, model, income, grids):
		self.p = p
		self.model = model
		self.income = income
		self.grids = grids
		self.step = None

	def calibrate(self):
		boundsObj = optimize.Bounds(self.lbounds, self.ubounds,
			keep_feasible=True)
		optimize.minimize(self.optim_handle, self.x0, bounds=boundsObj,
			method='SLSQP', jac='2-point',
			options={'finite_diff_rel_step': None})

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
		super().__init__(p, model, income, grids)
		self.lbounds = [0.95]
		self.ubounds = [0.999]
		self.x0 = np.array([0.98])

	def optim_handle(self, x):
		self.p.setParam('timeDiscount', x, True)

		eqSimulator = self.simulate()

		z = np.linalg.norm(eqSimulator.results['Mean wealth'] - 3.2)
		return z

class Calibrator2(Calibrator):
	def __init__(self, p, model, income, grids):
		super().__init__(p, model, income, grids)
		self.lbounds = [0.96, 0.01]
		self.ubounds = [0.9995, 0.05]
		self.x0 = np.array([0.995, 0.03])
		self.step = [0.01, 0.01]

	def optim_handle(self, x):
		self.p.setParam('timeDiscount', x[0], True)
		self.p.setParam('discount_factor_grid', np.array([x[0]- 2 * x[1], x[0] - x[1], x[0]]), True)
		self.model.p = self.p

		eqSimulator = self.simulate()

		z = np.zeros(2)
		z[0] = eqSimulator.results['Mean wealth'] - 3.2
		z[1] = eqSimulator.results['Wealth <= $1000'] - 0.23

		print(f'Mean wealth = {z[0] + 3.2}')
		print(f'Wealth constrained = {z[1] + 0.23}')

		return np.linalg.norm(z)

class Calibrator3(Calibrator):
	def __init__(self, p, model, income, grids):
		super().__init__(p, model, income, grids)
		self.lbounds = [1e-6]
		self.ubounds = [5e-3]
		self.x0 = np.array([1e-4])

	def optim_handle(self, x):
		self.p.setParam('adjustCost', x[0], True)
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

		print(f'P(MPC > 0) = {z[0] + 0.2}')

		return np.linalg.norm(z)

	