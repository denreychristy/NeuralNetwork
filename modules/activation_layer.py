# Neural Network - Activation Layer

# ================================================================================================ #
# Imports

from typing import Callable

from modules.layer import Layer

# ================================================================================================ #

class ActivationLayer(Layer):
	def __init__(self, activation_tuple: tuple[Callable, Callable]):
		self.activation: Callable = activation_tuple[0]
		self.activation_prime: Callable = activation_tuple[1]
	
	# ================================================== #
	# Class Methods

	# ================================================== #
	# Dunder Methods

	# ================================================== #
	# Property Methods

	# ================================================== #
	# Set Methods

	# ================================================== #
	# Other Methods

	def forward_propagation(self, input_data):
		self.input = input_data
		self.output = self.activation(self.input)
		return self.output

	def backward_propagation(self, output_error, learning_rate):
		return self.activation_prime(self.input) * output_error

# ================================================================================================ #