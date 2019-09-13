from misc import functions
import numpy as np

# gr = (np.sqrt(5) + 1) / 2
# grsq = gr ** 2

# fn = lambda x: x ** 3

# print(functions.goldenSectionSearch(fn, -1, 0.5, 
# 	gr, grsq, 1e-5, tuple()))

cgrid = np.array([0.001,0.05,1,2])
for cval in [-0.1,0.001,0.05,1.2,3]:
	ind2 = functions.searchSortedSingleInput(cgrid, cval)
	ind1 = ind2 - 1
	weights = functions.getInterpolationWeights(cgrid, cval, ind2)
	print(f'cval is {cval}')
	print(f'index is {ind2}')
	print(np.asarray(weights))

cvals = np.array([-0.1,0.001,0.05,1.2,3])
print(np.asarray(functions.searchSortedMultipleInput(cgrid,cvals)))