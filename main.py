# Neural Network - Main

# ================================================================================================ #
# Imports

import numpy as np

from modules.neural_network import NeuralNetwork

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

	training_time = 10.0

	network.train(input_data, output_data, 1_000_000, .001, training_time, None, False)

	error = network.current_error_rate
	if error is not None:
		print()
		print(f'Current error rate: {round(error, 4)}')

	print()
	for x, y in zip(input_data, output_data):
		print(f'Actual: {y[0][0]} | Predicted: {round(network.predict(x)[0][0][0])}')
	
	print()
	print(f'Error Rate By 5 Minutes: {network.error_rate_by_5_minutes}')

	print()
	related_network = network.create_related_network()
	related_network.train(input_data, output_data, 1_000_000, .001, training_time, None, False)
	print(f'Related network structure: {related_network.structure}')
	print(related_network.current_error_rate)

	if related_network.current_error_rate is not None:
		if network.current_error_rate is not None:
			if related_network.current_error_rate < network.current_error_rate:
				print('The new related network is better.')
			else:
				print('The first network was better.')