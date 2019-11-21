from scipy.io import loadmat
import numpy as np
cimport numpy as np

cdef class Income:
	def __init__(self, params):
		self.p = params

		self.readIncome()

		self.createOtherIncomeVariables()
		
	def readIncome(self):
		matFile = loadmat(self.p.locIncomeProcess)

		# persistent component
		if self.p.noPersIncome:
			self.nyP = 1
			self.yPdist = np.array([1.0])
			self.yPgrid = np.array([1.0])
			self.yPtrans = np.array([[1.0]])
		else:
			self.logyPgrid = np.asarray(matFile['discmodel1']['logyPgrid'][0][0]).flatten()
			self.nyP = self.logyPgrid.size

			self.yPdist = np.asarray(matFile['discmodel1']['yPdist'][0][0]).flatten()
			self.yPtrans = matFile['discmodel1']['yPtrans'][0][0]

			self.yPgrid = np.exp(self.logyPgrid)

		self.yPgrid = np.asarray(self.yPgrid) / np.dot(self.yPdist.T,self.yPgrid)
		self.yPcumdist = np.cumsum(self.yPdist)
		self.yPcumdistT = np.transpose(self.yPcumdist)
		self.yPcumtrans = np.cumsum(self.yPtrans,axis=1)

		# transitory income
		if self.p.noTransIncome:
			self.nyT = 1
			self.yTdist = np.array([1.0])
			self.yTgrid = np.array([1.0])
		else:
			self.logyTgrid = np.asarray(matFile['discmodel1']['logyTgrid'][0][0]).flatten()
			self.nyT = self.logyTgrid.size

			self.yTdist = np.asarray(matFile['discmodel1']['yTdist'][0][0]).flatten()
			self.yTgrid = np.exp(self.logyTgrid)

		self.yTgrid = np.asarray(self.yTgrid) / np.dot(self.yTdist.T,self.yTgrid)
		self.yTgrid = np.asarray(self.yTgrid) / self.p.freq

		self.yTcumdist = np.cumsum(self.yTdist)
		self.yTcumdistT = np.transpose(self.yTcumdist)

	def createOtherIncomeVariables(self):
		# matrix of all income values, dimension nyP by nyT
		self.ymat = np.matmul(
			np.reshape(self.yPgrid, (-1, 1)),
			np.reshape(self.yTgrid, (1, -1))
			)

		self.ymin = np.min(np.asarray(self.ymat).flatten())