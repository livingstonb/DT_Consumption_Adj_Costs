from scipy.io import loadmat
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
		
	def readIncome(self)
		matFile = loadmat(params.locIncomeProcess)

		# persistent component
		self.logyPgrid = matFile['discmodel1']['logyPgrid']
		self.yPdist = matFile['discmodel1']['yPdist']
		self.yPtrans = matFile['discmodel1']['yPtrans']

		self.nyP = self.yPtrans.size
		self.yPgrid = np.exp(self.logyPgrid)
		self.yPgrid = self.yPgrid / np.dot(self.yPdist,self.yPgrid)
		self.yPcumdist = np.cumsum(self.yPdist)
		self.yPcumtrans = np.cumsum(self.yPtrans,axis=1)

		# transitory income
		self.logyTgrid = matFile['discmodel1']['logyTgrid']
		self.yTdist = matFile['discmodel1']['yTdist']

		self.nyT = self.yTtrans.size
		self.yTgrid = np.exp(self.logyTgrid)
		self.yTgrid = self.yTgrid / np.dot(self.yTdist,self.yTgrid)
		self.yTgrid = self.yTgrid / self.params.freq
		self.yTcumdist = np.cumsum(self.yTdist)

	def createOtherIncomeVariables(self)
		self.ymat = np.matmul(self.yPgrid,obj.yTgrid.T)