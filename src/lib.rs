// Rust Module

// ============================================================================================== //
// Imports

use rand::RngExt;
use ndarray::Array2;
use pyo3::prelude::*;

// ============================================================================================== //
// PyModule Function

#[pymodule]
fn rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
	m.add_class::<Network>()?;
	Ok(())
}

// ============================================================================================== //
// Matrix Struct

#[derive(Clone)]
struct Matrix {
	array: Array2<f32>,
}

impl Matrix {
	fn add(&self, other: &Matrix) -> Self {
		Self { array: &self.array + &other.array }
	}

	fn dot(&self, other: &Matrix) -> Self {
		Self { array: self.array.dot(&other.array) }
	}

	fn from_data(data: Vec<Vec<f32>>) -> Self {
		let rows = data.len();
		let cols = data[0].len();
		let flat_data: Vec<f32> = data.into_iter().flatten().collect();
		let array = Array2::from_shape_vec(
			(rows, cols),
			flat_data
		)
			.expect("Dimension mismatch!");

		Self { array }
	}

	fn transpose(&self) -> Self {
		Self { array: self.array.t().to_owned() }
	}

	fn random(rows: usize, cols: usize, lb: f32, ub: f32) -> Self {
		let mut rng = rand::rng();
		Self { array: Array2::from_shape_fn((rows, cols), |_| rng.random_range(lb..=ub)) }
	}

	fn scale(&self, factor: f32) -> Self {
		Self { array: &self.array * factor }
	}

	fn zeros(rows: usize, cols: usize) -> Self {
		Self { array: Array2::<f32>::zeros((rows, cols)) }
	}

	fn to_vec(&self) -> Vec<Vec<f32>> {
		self.array
			.rows()
			.into_iter()
			.map(|row| row.to_vec())
			.collect()
	}

	fn mul(&self, other: &Matrix) -> Self {
		Self { array: &self.array * &other.array }
	}
}

// ============================================================================================== //
// Propagation Trait

trait Propagation {
	fn forward(&mut self, input: Matrix) -> Matrix;
	fn backward(&mut self, output_error: Matrix, learning_rate: f32) -> Matrix;
}

// ============================================================================================== //
// Fully Connected Layer

struct FullyConnectedLayer {
	input: Matrix,
	output: Matrix,
	weights: Matrix,
	bias: Matrix,
}

impl FullyConnectedLayer {
	fn new(input_size: usize, output_size: usize) -> Self {
		Self {
			input: Matrix::zeros(1, input_size),
			output: Matrix::zeros(1, output_size),
			weights: Matrix::random(input_size, output_size, 0.0, 1.0),
			bias: Matrix::random(1, output_size, -1.0, 1.0),
		}
	}
}

impl Propagation for FullyConnectedLayer {
	fn forward(&mut self, input_data: Matrix) -> Matrix {
		self.input = input_data;
		self.output = self.input.dot(&self.weights).add(&self.bias);
		self.output.clone()
	}

	fn backward(&mut self, output_error: Matrix, learning_rate: f32) -> Matrix {
		let input_error = output_error.dot(&self.weights.transpose());
		let weights_error = self.input.transpose().dot(&output_error);

		self.weights = self.weights.add(&weights_error.scale(-1.0 * learning_rate));
		self.bias = self.bias.add(&output_error.scale(-1.0 * learning_rate));

		input_error
	}
}

// ============================================================================================== //
// Activation Layer

struct ActivationLayer {
	input: Matrix,
	output: Matrix,
	activation: fn(&Matrix) -> Matrix,
	activation_prime: fn(&Matrix) -> Matrix,
}

impl ActivationLayer {
	fn new(input_size: usize, output_size: usize, activation: fn(&Matrix) -> Matrix,
		activation_prime: fn(&Matrix) -> Matrix) -> Self {
		
		Self {
			input: Matrix::zeros(1, input_size),
			output: Matrix::zeros(1, output_size),
			activation,
			activation_prime,
		}
	}
}

impl Propagation for ActivationLayer {
	fn forward(&mut self, input: Matrix) -> Matrix {
		self.input = input;
		self.output = (self.activation)(&self.input);
		self.output.clone()
	}

	fn backward(&mut self, output_error: Matrix, _learning_rate: f32) -> Matrix {
		let act_prime = (self.activation_prime)(&self.input);
		act_prime.mul(&output_error) 
	}
}

// ============================================================================================== //
// Activation Functions

fn vector_tanh(matrix: &Matrix) -> Matrix {
	Matrix { array: matrix.array.mapv(|x| x.tanh()) }
}

fn vector_tanh_prime(matrix: &Matrix) -> Matrix {
	Matrix { array: matrix.array.mapv(|x| 1.0 - x.tanh().powi(2)) }
}

// ============================================================================================== //
// Loss Functions

fn mse(y_true: &Matrix, y_pred: &Matrix) -> f32 {
	(&y_true.array - &y_pred.array)
		.mapv(|x| x.powi(2))
		.mean()
		.unwrap_or(0.0)
}

fn mse_prime(y_true: &Matrix, y_pred: &Matrix) -> Matrix {
	let n = y_true.array.len() as f32;
	let array = (&y_pred.array - &y_true.array).mapv(|x| 2.0 * x / n);

	Matrix { array }
}

// ============================================================================================== //
// Layer Enum

enum Layer {
	FullyConnectedLayer(FullyConnectedLayer),
	ActivationLayer(ActivationLayer),
}

impl Propagation for Layer {
	fn forward(&mut self, input: Matrix) -> Matrix {
		match self {
			Layer::FullyConnectedLayer(l) => l.forward(input),
			Layer::ActivationLayer(l) => l.forward(input),
		}
	}

	fn backward(&mut self, output_error: Matrix, learning_rate: f32) -> Matrix {
		match self {
			Layer::FullyConnectedLayer(l) => l.backward(output_error, learning_rate),
			Layer::ActivationLayer(l) => l.backward(output_error, learning_rate),
		}
	}
}

// ============================================================================================== //
// Network Struct (Python Bound)

#[pyclass] // 2. Mark struct for PyO3
struct Network {
	structure: Vec<usize>,
	layers: Vec<Layer>,
}

#[pymethods] // 3. Expose methods to Python
impl Network {
	#[new] // Maps to Network(structure) in Python
	fn new(structure: Vec<usize>) -> Self {
		let mut layers: Vec<Layer> = Vec::new();
		for l in 0..(structure.len() - 1) {
			layers.push(Layer::FullyConnectedLayer(FullyConnectedLayer::new(
				structure[l],
				structure[l + 1],
			)));
			layers.push(Layer::ActivationLayer(ActivationLayer::new(
				structure[l + 1],
				structure[l + 1],
				vector_tanh,
				vector_tanh_prime,
			)));
		}

		Self { structure, layers }
	}

	// Pass nested list from Python: net.predict([[0.5, 0.2]]) -> [[0.81]]
	fn predict(&mut self, input_data: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
		let mut current = Matrix::from_data(input_data);
		for layer in &mut self.layers {
			current = layer.forward(current);
		}
		current.to_vec()
	}

	#[pyo3(signature = (input_data, target, epochs, learning_rate, update_frequency = 1000))]
	fn train(
		&mut self,
		input_data: Vec<Vec<f32>>,
		target: Vec<Vec<f32>>,
		epochs: u32,
		learning_rate: f32,
		update_frequency: usize,
	) -> f32 {
		let inputs = Matrix::from_data(input_data);
		let outputs = Matrix::from_data(target);

		let mut err = 0.0;
		for e in 1..=epochs {
			// Forward pass
			let predictions = self.predict_from_matrix(&inputs);
			err = mse(&outputs, &predictions);

			// Compute initial error gradient dE/dY
			let mut error = mse_prime(&outputs, &predictions);

			// Backpropagate backwards through layers
			for layer in self.layers.iter_mut().rev() {
				error = layer.backward(error, learning_rate);
			}

			if (e as usize) % update_frequency == 0 || e == epochs {
				println!("Epoch {}: error {}", e, err);
			}
		}

		err
	}

	#[pyo3(signature = (input_data, target, loss_threshold, max_epochs, learning_rate,
		update_frequency = 1000))]
	fn train_until(
		&mut self,
		input_data: Vec<Vec<f32>>,
		target: Vec<Vec<f32>>,
		loss_threshold: f32,
		max_epochs: usize,
		learning_rate: f32,
		update_frequency: usize,
	) -> usize {
		let inputs = Matrix::from_data(input_data);
		let outputs = Matrix::from_data(target);

		let mut epoch = 1;
		let mut current_loss = f32::MAX;

		while (current_loss > loss_threshold) & (epoch <= max_epochs) {
			// Forward pass
			let predictions = self.predict_from_matrix(&inputs);
			current_loss = mse(&outputs, &predictions);

			// Compute initial error gradient dE/dY (Renamed to `gradient`)
			let mut gradient = mse_prime(&outputs, &predictions);

			// Backpropagate backwards through layers
			for layer in self.layers.iter_mut().rev() {
				gradient = layer.backward(gradient, learning_rate);
			}

			if epoch % update_frequency == 0 {
				println!("Epoch {}: MSE Loss {}", epoch, current_loss);
			}

			epoch += 1;
		}

		println!("Target reached at Epoch {}: MSE Loss {}", epoch, current_loss);

		epoch
	}
}

impl Network {
	fn predict_from_matrix(&mut self, inputs: &Matrix) -> Matrix {
		let mut current = inputs.clone();
		for layer in &mut self.layers {
			current = layer.forward(current);
		}

		current
	}
}