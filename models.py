from functions import *

df = pd.read_csv('data/cleaned_ratio_data')

'''
Erics Model
'''

def neck(tshirt, height_inches, weight_pounds):
	necksize = .2104 * tshirt + -0.0874 * height_inches + 0.0216 *\
	weight_pounds + 17.2057
	return necksize

def sleeve(tshirt, height_inches, weight_pounds, build=None, suit=None, inseam=None):
	if not build and not suit and not inseam:
		sleevesize = 0.1425 * tshirt + 0.1835 * height_inches + 0.0073 *\
		weight_pounds + 1.303
	else:
		sleevesize = 0.1859 * tshirt + 0.3179 * height_inches + 0.0048 *\
		weight_pounds + 9.5991
	return sleevesize

# changing build to 1/2/3/4