class Network:
	def __init__(self, structure: list[int]) -> None: ...
	def predict(self, input_data: list[list[float]]) -> list[list[float]]: ...
	def train(
		self,
		input_data: list[list[float]],
		target: list[list[float]],
		epochs: int,
		learning_rate: float,
		update_frequency: int = 1000,
	) -> float: ...

	def train_until(
			self,
			input_data: list[list[float]],
			target: list[list[float]],
			loss_threshold: float,
			max_epochs: int,
			learning_rate: float,
			update_frequency: int = 1000,
		) -> int: ...