# Neural Network - Main

# ================================================================================================ #
# Imports

import numpy as np

from NeuralNetwork.modules.neural_network import NeuralNetwork

# ================================================================================================ #

if __name__ == '__main__':
	network = NeuralNetwork([2, 3, 1])

	input_data = np.array([
		[[False, False]],
		[[True, False]],
		[[False, True]],
		[[True, True]]
	])

	output_data = np.array([
		[[False]],
		[[True]],
		[[True]],
		[[False]]
	])

	network.train(input_data, output_data, 100_000, .01, 30.0, .0001, True)

	error = network.current_error_rate
	if error is not None:
		print()
		print(f'Current error rate: {round(error, 4)}')

	print()
	for x, y in zip(input_data, output_data):
		print(f'Actual: {y[0][0]} | Predicted: {round(network.predict(x)[0][0][0])}')