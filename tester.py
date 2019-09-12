from misc.functions import goldenSectionSearch
import numpy as np

gr = (np.sqrt(5) + 1) / 2
grsq = gr ** 2

fn = lambda x: x ** 3

print(goldenSectionSearch(fn, -1, 0.5, 
	gr, grsq, 1e-5, tuple()))