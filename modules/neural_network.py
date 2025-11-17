# Neural Network - Neural Network

# ================================================================================================ #
# Imports

from time	import time
from typing	import Callable, Optional

from activation_functions	import activation_functions
from activation_layer		import ActivationLayer
from fully_connected_layer	import FullyConnectedLayer
from layer					import Layer
from loss_functions			import loss_functions

# ================================================================================================ #

class NeuralNetwork:
	def __init__(self, layers: list[int],
		loss_tuple: tuple[Callable, Callable] = loss_functions['mse']):
		
		self.layers: list[Layer] = []
		for i in range(len(layers) - 1):
			self.add_layer(FullyConnectedLayer(layers[i], layers[i + 1]))
			self.add_layer(ActivationLayer(activation_functions['tanh']))

		self.set_loss_function(loss_tuple)

		self.current_error_rate: Optional[float] = None
	
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

	def add_layer(self, layer):
		self.layers.append(layer)

	def set_loss_function(self, loss_tuple: tuple[Callable, Callable]):
		self.loss = loss_tuple[0]
		self.loss_prime = loss_tuple[1]

	def predict(self, input_data):
		samples = len(input_data)
		result = []

		for i in range(samples):
			output = input_data[i]
			for layer in self.layers:
				output = layer.forward_propagation(output)
			result.append(output)

		return result

	def train(self, x_train, y_train, epochs: int = 10_000, learning_rate: float = .01,
		max_time_in_seconds: Optional[float] = None, error_threshold: Optional[float] = None,
		print_progress: bool = False, print_progress_time_interval: float = 1.0):
		
		start_time = time()
		last_print_time = None
		samples = len(x_train)

		err = 0
		for i in range(epochs):
			if max_time_in_seconds is not None:
				if time() > (start_time + max_time_in_seconds):
					break

			err = 0
			for j in range(samples):
				output = x_train[j]
				for layer in self.layers:
					output = layer.forward_propagation(output)

				err += self.loss(y_train[j], output)
				error = self.loss_prime(y_train[j], output)

				for layer in reversed(self.layers):
					error = layer.backward_propagation(error, learning_rate)

			# calculate average error on all samples
			err /= samples
			if print_progress:
				if last_print_time is None:
					print(f'epoch {i + 1} / {epochs}\terror = {round(err, 4)}')
					last_print_time = time()
				else:
					if time() > (last_print_time + print_progress_time_interval):
						print(f'epoch {i + 1} / {epochs}\terror = {round(err, 4)}')
						last_print_time = time()
			
			# Break if error threshold is reached
			if error_threshold is not None:
				if err <= error_threshold:
					print(f'epoch {i + 1} / {epochs}\terror = {round(err, 4)}')
					self.current_error_rate = err
					return
			
		self.current_error_rate = err

# ================================================================================================ #