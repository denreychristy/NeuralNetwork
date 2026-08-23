use ndarray::Array2;
use pyo3::prelude::*;
use rand::RngExt;

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

	fn mul(&self, other: &Matrix) -> Self {
		Self { array: &self.array * &other.array }
	}

	fn scale(&self, factor: f32) -> Self {
		Self { array: &self.array * factor }
	}

	fn transpose(&self) -> Self {
		Self { array: self.array.t().to_owned() }
	}

	fn from_data(data: Vec<Vec<f32>>) -> Self {
		let rows = data.len();
		let cols = if rows > 0 { data[0].len() } else { 0 };
		let flat_data: Vec<f32> = data.into_iter().flatten().collect();
		let array = Array2::from_shape_vec((rows, cols), flat_data)
			.expect("Dimension mismatch in Matrix::from_data");

		Self { array }
	}

	fn random(rows: usize, cols: usize, lb: f32, ub: f32) -> Self {
		let mut rng = rand::rng();
		Self {
			array: Array2::from_shape_fn((rows, cols), |_| rng.random_range(lb..=ub)),
		}
	}

	fn zeros(rows: usize, cols: usize) -> Self {
		Self { array: Array2::<f32>::zeros((rows, cols)) }
	}

	fn to_vec(&self) -> Vec<Vec<f32>> {
		(0..self.array.nrows())
			.map(|r| (0..self.array.ncols()).map(|c| self.array[[r, c]]).collect())
			.collect()
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
	weights: Matrix,
	bias: Matrix,
}

impl FullyConnectedLayer {
	fn new(input_size: usize, output_size: usize) -> Self {
		Self {
			input: Matrix::zeros(1, input_size),
			weights: Matrix::random(input_size, output_size, -1.0, 1.0),
			bias: Matrix::random(1, output_size, -1.0, 1.0),
		}
	}
}

impl Propagation for FullyConnectedLayer {
	fn forward(&mut self, input_data: Matrix) -> Matrix {
		self.input = input_data;
		let mut out = self.input.dot(&self.weights);
		
		// Broadcast 1xN bias across B rows of the output
		for r in 0..out.array.nrows() {
			for c in 0..out.array.ncols() {
				out.array[[r, c]] += self.bias.array[[0, c]];
			}
		}
		out
	}

	fn backward(&mut self, output_error: Matrix, learning_rate: f32) -> Matrix {
		let input_error = output_error.dot(&self.weights.transpose());
		let weights_error = self.input.transpose().dot(&output_error);

		// Sum gradients along columns for batch bias updates
		let mut bias_error = Matrix::zeros(1, self.bias.array.ncols());
		for c in 0..output_error.array.ncols() {
			let sum: f32 = output_error.array.column(c).sum();
			bias_error.array[[0, c]] = sum;
		}

		self.weights = self.weights.add(&weights_error.scale(-learning_rate));
		self.bias = self.bias.add(&bias_error.scale(-learning_rate));

		input_error
	}
}

// ============================================================================================== //
// Activation Layer

struct ActivationLayer {
	input: Matrix,
	activation: fn(&Matrix) -> Matrix,
	activation_prime: fn(&Matrix) -> Matrix,
}

impl ActivationLayer {
	fn new(
		activation: fn(&Matrix) -> Matrix,
		activation_prime: fn(&Matrix) -> Matrix,
	) -> Self {
		Self {
			input: Matrix::zeros(1, 1),
			activation,
			activation_prime,
		}
	}
}

impl Propagation for ActivationLayer {
	fn forward(&mut self, input: Matrix) -> Matrix {
		self.input = input;
		(self.activation)(&self.input)
	}

	fn backward(&mut self, output_error: Matrix, _learning_rate: f32) -> Matrix {
		let act_prime = (self.activation_prime)(&self.input);
		act_prime.mul(&output_error) // Hadamard product for element-wise activation
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

#[pyclass]
struct Network {
	#[allow(dead_code)]
	structure: Vec<usize>,
	layers: Vec<Layer>,
}

#[pymethods]
impl Network {
	#[new]
	fn new(structure: Vec<usize>) -> Self {
		let mut layers: Vec<Layer> = Vec::new();
		for l in 0..(structure.len() - 1) {
			layers.push(Layer::FullyConnectedLayer(FullyConnectedLayer::new(
				structure[l],
				structure[l + 1],
			)));
			layers.push(Layer::ActivationLayer(ActivationLayer::new(
				vector_tanh,
				vector_tanh_prime,
			)));
		}

		Self { structure, layers }
	}

	fn predict(&mut self, input_data: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
		let input_matrix = Matrix::from_data(input_data);
		let result_matrix = self.forward_internal(&input_matrix);
		result_matrix.to_vec()
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
			let predictions = self.forward_internal(&inputs);
			err = mse(&outputs, &predictions);

			let mut error = mse_prime(&outputs, &predictions);

			for layer in self.layers.iter_mut().rev() {
				error = layer.backward(error, learning_rate);
			}

			if (e as usize) % update_frequency == 0 || e == epochs {
				println!("Epoch {}: MSE Loss {}", e, err);
			}
		}

		err
	}
}

impl Network {
	fn forward_internal(&mut self, inputs: &Matrix) -> Matrix {
		let mut current = inputs.clone();
		for layer in &mut self.layers {
			current = layer.forward(current);
		}
		current
	}
}