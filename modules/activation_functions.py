# Neural Network - Activation Functions

# ================================================================================================ #
# Imports

import numpy as np

# ================================================================================================ #

def tanh(x):
	return np.tanh(x)

def tanh_prime(x):
	return 1-np.tanh(x)**2

# ================================================================================================ #

# {'name': (function, derivative)}
activation_functions = {
	'tanh': (tanh, tanh_prime),
}