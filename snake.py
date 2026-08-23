# Neural Network - Snake

# ================================================================================================ #
# Imports

from enum import Enum, IntEnum
from random import choice, randint
from typing import Optional

import pygame as pg

from rust import Network

# ================================================================================================ #
# Constants

GRID_SIZE: int = 10
CELL_SIZE: int = 20
MAX_DISTANCE: float = ((GRID_SIZE) ** 2 + (GRID_SIZE) ** 2) ** .5

# ================================================================================================ #

class Cell(Enum):
	EMPTY = 0
	FOOD = 1
	SNAKE = -1
	HEAD = -0.5

	@property
	def to_float(self) -> float:
		if self == Cell.EMPTY:
			return 0.0
		if self == Cell.FOOD:
			return 1.0
		if self == Cell.SNAKE:
			return -1.0
		if self == Cell.HEAD:
			return -0.5
		return 0.0

# ================================================================================================ #

class Direction(IntEnum):
	LEFT = 0
	UP = 1
	RIGHT = 2
	DOWN = 3

	@classmethod
	def as_list(cls) -> list['Direction']:
		return [cls.LEFT, cls.UP, cls.RIGHT, cls.DOWN]

	@classmethod
	def random(cls) -> 'Direction':
		return choice(cls.as_list())

	def turn_left(self) -> 'Direction':
		return Direction((self - 1) % 4)

	def turn_right(self) -> 'Direction':
		return Direction((self + 1) % 4)

	def to_one_hot(self) -> list[int]:
		result = [0, 0, 0, 0]
		result[int(self)] = 1
		return result

# ================================================================================================ #

class Grid:
	def __init__(self):
		self.vector: list[Cell] = [Cell.EMPTY for _ in range(GRID_SIZE * GRID_SIZE)]
		self.snake: list[tuple[int, int]] = [self.random_empty_coords()]
		self.direction: Direction = Direction.random()

		self.place_food()

	def set_cell(self, coords: tuple[int, int], cell: Cell) -> None:
		index = self.coords_to_index(coords)
		self.vector[index] = cell

	def get_cell(self, coords: tuple[int, int]) -> Optional[Cell]:
		# Out of Boundsa
		x, y = coords
		if x < 0 or x >= GRID_SIZE or y < 0 or y >= GRID_SIZE:
			return None

		index = self.coords_to_index(coords)
		return self.vector[index]

	def random_empty_index(self) -> int:
		options = []
		for i, cell in enumerate(self.vector):
			if cell == Cell.EMPTY:
				options.append(i)
		return choice(options)

	def random_empty_coords(self) -> tuple[int, int]:
		index = self.random_empty_index()
		coords = self.index_to_coords(index)
		return coords

	def index_to_coords(self, index: int) -> tuple[int, int]:
		return (index % GRID_SIZE, index // GRID_SIZE)

	def coords_to_index(self, coords: tuple[int, int]) -> int:
		x, y = coords
		return y * GRID_SIZE + x

	def place_food(self) -> None:
		index = self.random_empty_index()
		self.vector[index] = Cell.FOOD

	def distance_to_food(self) -> float:
		hx, hy = self.snake[-1]
		fx, fy = self.index_to_coords(self.vector.index(Cell.FOOD))
		distance = ((hx - fx) ** 2 + (hy - fy) ** 2) ** .5
		return distance

	def move_snake(self) -> tuple[bool, bool]:
		hx, hy = self.snake[-1]
		self.set_cell((hx, hy), Cell.SNAKE)

		if self.direction == Direction.LEFT:
			hx -= 1
		elif self.direction == Direction.UP:
			hy -= 1
		elif self.direction == Direction.RIGHT:
			hx += 1
		elif self.direction == Direction.DOWN:
			hy += 1
		cell = self.get_cell((hx, hy))
		if cell is None:
			return (False, False)
		
		if cell == Cell.SNAKE:
			return (False, False)

		food = False
		if cell == Cell.EMPTY:
			tx, ty = self.snake.pop(0)
			self.set_cell((tx, ty), Cell.EMPTY)

		if cell == Cell.FOOD:
			self.place_food()
			food = True

		self.snake.append((hx, hy))
		self.set_cell((hx, hy), Cell.HEAD)

		return (True, food)

	def update(self) -> tuple[bool, bool]:
		result = self.move_snake()

		return result

	def display(self, target_surf: pg.surface.Surface) -> None:
		for y in range(GRID_SIZE):
			for x in range(GRID_SIZE):
				cell = self.get_cell((x, y))
				if cell == Cell.EMPTY: continue

				color = (0, 0, 255) if cell == Cell.FOOD else (0, 255, 0)
				pg.draw.rect(
					target_surf,
					color,
					(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
				)

# ================================================================================================ #

class Snake:
	def __init__(self):
		pg.init()

		self.clock = pg.time.Clock()
		self.fps = 60

		window_size = GRID_SIZE * CELL_SIZE
		self.window_surf = pg.display.set_mode((window_size, window_size))
		self.window_rect = self.window_surf.get_rect()

		self.grid = Grid()

		self.network = Network([GRID_SIZE * GRID_SIZE + 4, 128, 128, 128, 128, 64, 32, 16, 8, 4, 2, 1])
		self.input_history = []
		self.output_history = []
		self.distance_to_food = MAX_DISTANCE
		self.record_distance = MAX_DISTANCE

		self.flag_run: bool = False

	def reset(self):
		self.grid = Grid()
		self.distance_to_food = MAX_DISTANCE
		self.record_distance = MAX_DISTANCE

	def run(self):
		self.flag_run = True
		while self.flag_run:
			self.clock.tick(self.fps)
			self.user_input()
			self.update()
			self.display()
		pg.quit()

	def user_input(self):
		for event in pg.event.get():
			if event.type == pg.QUIT:
				self.flag_run = False

	def update(self):
		success, food = self.grid.update()

		if food:
			print()
			print(" HE FOUND THE FOOD! ")
			print()

		if not success:
			self.reset()
			return

		current_distance_to_food = self.grid.distance_to_food()

		if self.input_history and (food or current_distance_to_food < self.distance_to_food):
			if current_distance_to_food < self.record_distance:
				epochs = 100
				self.record_distance = current_distance_to_food
			elif food:
				epochs = 1000
			else:
				epochs = 0
			self.network.train(self.input_history, self.output_history, epochs, 0.1)

		while len(self.input_history) > 10_000:
			self.input_history.pop(0)
			self.output_history.pop(0)

		self.distance_to_food = current_distance_to_food

		input_data = [cell.to_float for cell in self.grid.vector] + self.grid.direction.to_one_hot()
		self.input_history.append(input_data)

		if randint(0, 100) < 10:
			prediction = randint(-1, 1)
		else:
			prediction = self.network.predict([input_data])[0][0]
		if -1.0 <= prediction < -.33:
			self.grid.direction = self.grid.direction.turn_left()
			self.output_history.append([-1.0])
		elif .33 <= prediction <= 1.0:
			self.grid.direction = self.grid.direction.turn_right()
			self.output_history.append([1.0])
		else:
			self.output_history.append([0.0])

	def display(self):
		self.window_surf.fill((31, 31, 31))
		self.grid.display(self.window_surf)
		pg.display.flip()

# ================================================================================================ #

if __name__ == '__main__':
	game = Snake()
	game.run()