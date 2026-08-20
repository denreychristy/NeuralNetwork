# Neural Network - Main

# ================================================================================================ #
# Imports

from math import log10, sin, cos, tan, pi
from random import randint
from time import time

from rust import Network

# ================================================================================================ #

def format_predictions(predictions: list[list[float]]) -> list[list[float]]:
	for s, sample in enumerate(predictions):
		for v, value in enumerate(sample):
			predictions[s][v] = round(predictions[s][v], 2)
	return predictions

def create_optimum_network(inputs: list[list[float]], outputs: list[list[float]]):
	input_size = len(inputs[0])
	output_size = len(outputs[0])

	results = []
	for _ in range(10):
		internal_layers = randint(1, randint(1, 5))
		structure = [input_size]
		for _ in range(internal_layers):
			lb = min(structure[-1], output_size + 1)
			ub = 2 * max(structure[-1], output_size)
			structure.append(randint(lb, ub))
		structure.append(output_size)

		network = Network(structure)
		training_start = time()
		epochs = network.train_until(inputs, outputs, 0.0001, 100_000, 1.0)
		elapsed = time() - training_start
		results.append({
			'structure': tuple(structure),
			'epochs': epochs,
			'elapsed': elapsed
		})

	print()
	tested_structures = set([x['structure'] for x in results])
	structures_dict = {}
	for structure in tested_structures:
		total_epochs = 0
		total_elapsed = 0
		times_tested = 0
		for example in results:
			if tuple(example['structure']) == structure:
				total_epochs += example['epochs']
				total_elapsed += example['elapsed']
				times_tested += 1
		average_epochs = total_epochs / times_tested
		average_elapsed = total_elapsed / times_tested
		statistics = (log10(sum(structure)), log10(average_epochs), average_elapsed)
		structures_dict[structure] = sum(statistics)
	results = list(structures_dict.items())
	results.sort(key = lambda structure: structure[1])
	print(results[0])

# ================================================================================================ #

inputs = [
	[cos(x * pi), sin(x * pi)] for x in [0.0, 0.5, 1.0, 1.5]
]

outputs = [
	(x + y) for (x, y) in inputs
]