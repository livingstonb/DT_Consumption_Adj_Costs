from scipy.io import loadmat
import numpy as np
import numpy.matlib as matlib

class Income:
	"""
	This class stores income variables.

	Mean annual income is normalized to 1 by
	normalizing persistent income to have mean
	1 and transitory income to have mean 1 if
	frequency is annual and 1/4 if frequency
	is quarterly.
	"""

	def __init__(self, params):
		self.p = params

		self.readIncome()

		self.createOtherIncomeVariables()
		
	def readIncome(self):
		matFile = loadmat(self.p.locIncomeProcess)

		# persistent component
		self.logyPgrid = matFile['discmodel1']['logyPgrid'][0][0]
		self.nyP = self.logyPgrid.size
		self.logyPgrid = self.logyPgrid.reshape((self.nyP,-1))

		self.yPdist = matFile['discmodel1']['yPdist'][0][0].reshape((self.nyP,-1))
		self.yPtrans = matFile['discmodel1']['yPtrans'][0][0]

		self.yPgrid = np.exp(self.logyPgrid)
		self.yPgrid = self.yPgrid / np.dot(self.yPdist.T,self.yPgrid)
		self.yPcumdist = np.cumsum(self.yPdist)
		self.yPcumtrans = np.cumsum(self.yPtrans,axis=1)

		# transitory income
		self.logyTgrid = matFile['discmodel1']['logyTgrid'][0][0]
		self.nyT = self.logyTgrid.size
		self.logyTgrid = self.logyPgrid.reshape((self.nyT,-1))

		self.yTdist = matFile['discmodel1']['yTdist'][0][0].reshape((self.nyT,-1))
		
		self.yTgrid = np.exp(self.logyTgrid)
		self.yTgrid = self.yTgrid / np.dot(self.yTdist.T,self.yTgrid)
		self.yTgrid = self.yTgrid / self.p.freq
		self.yTcumdist = np.cumsum(self.yTdist)

	def createOtherIncomeVariables(self):
		# matrix of all income values, dimension nyP by nyT
		self.ymat = np.matmul(self.yPgrid,self.yTgrid.T)

