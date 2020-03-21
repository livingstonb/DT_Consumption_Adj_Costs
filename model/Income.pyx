from scipy.io import loadmat
import numpy as np
cimport numpy as np

from misc.ergodicdist import ergodicdist

cdef class Income:
	def __init__(self, params, locIncome, normalize=True):
		self.p = params
		self.normalize = normalize

		self.readIncome(locIncome)

		self.createOtherIncomeVariables()
		
	def readIncome(self, locIncome):
		matFile = loadmat(locIncome)

		# persistent component
		if self.p.noPersIncome:
			self.nyP = 1
			self.yPdist = np.array([1.0])
			self.yPgrid = np.array([1.0])
			self.yPtrans = np.array([[1.0]])
		else:
			self.logyPgrid = np.asarray(matFile['logyPgrid']).flatten()
			self.nyP = self.logyPgrid.size

			# self.yPdist = np.asarray(matFile['yPdist']).flatten()
			self.yPtrans = matFile['yPtrans']
			self.yPdist = ergodicdist(self.yPtrans).flatten()

			self.yPgrid = np.exp(self.logyPgrid)

		if self.normalize:
			mu_yP = np.dot(self.yPdist.T, self.yPgrid)
			self.yPgrid = np.asarray(self.yPgrid) / mu_yP
			self.logyPgrid = np.log(self.yPgrid)

		self.yPcumdist = np.cumsum(self.yPdist)
		self.yPcumdistT = np.transpose(self.yPcumdist)
		self.yPcumtrans = np.cumsum(self.yPtrans, axis=1)

		# transitory income
		if self.p.noTransIncome:
			self.nyT = 1
			self.yTdist = np.array([1.0])
			self.yTgrid = np.array([1.0])
		else:
			self.logyTgrid = np.asarray(matFile['logyTgrid']).flatten()
			self.nyT = self.logyTgrid.size

			self.yTdist = np.asarray(matFile['yTdist']).flatten()
			self.yTgrid = np.exp(self.logyTgrid)

		if self.normalize:
			mu_yT = np.dot(self.yTdist.T, self.yTgrid)
			self.yTgrid = np.asarray(self.yTgrid) / mu_yT
			self.yTgrid = np.asarray(self.yTgrid) / self.p.freq
			self.logyTgrid = np.log(self.yTgrid)

		self.yTcumdist = np.cumsum(self.yTdist)
		self.yTcumdistT = np.transpose(self.yTcumdist)

	def createOtherIncomeVariables(self):
		# matrix of all income values, dimension nyP by nyT
		self.ymat = np.matmul(
			np.reshape(self.yPgrid, (-1, 1)),
			np.reshape(self.yTgrid, (1, -1))
			)

		self.ymin = np.min(np.asarray(self.ymat).flatten())

		self.ydist = np.matmul(
			np.reshape(self.yPdist, (-1, 1)),
			np.reshape(self.yTdist, (1, -1))
			)
		self.meany = np.matmul(
			np.reshape(self.ydist, (1, -1)),
			np.reshape(self.ymat, (-1, 1))
			)