# Neural Network - Main

# ================================================================================================ #
# Imports

from math import log10, sin, cos, tan, pi
from random import randint
from time import time

from rust import Network

# ================================================================================================ #



# ================================================================================================ #

inputs = [
	[0.0, 0.0],
	[0.0, 1.0],
	[1.0, 0.0],
	[1.0, 1.0]
]

outputs = [
	[0.0],
	[1.0],
	[1.0],
	[0.0]
]

network = Network([2, 3, 3, 1])
network.train_until(inputs, outputs, .01, 10_000, 1.0)
predictions = network.predict(inputs)
print(predictions)