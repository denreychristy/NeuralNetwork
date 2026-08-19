// Rust Module

// ============================================================================================== //
// Imports

use std::assert_eq;

use rand::RngExt;

use ndarray::{Array1, s};
use pyo3::prelude::*;

// ============================================================================================== //
// PyModule Function

#[pymodule]
fn rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}

// ============================================================================================== //
// Helper Functions

fn random_vector(size: usize, lb: f32, ub: f32) -> Vec<f32> {
	let mut rng = rand::rng();
	(0..size).map(|_| rng.random_range(lb..=ub)).collect()
}

fn vector_dot(vec_a: &[f32], vec_b: &[f32]) -> f32 {
	let arr_a = Array1::from_vec(vec_a.to_vec());
	let arr_b = Array1::from_vec(vec_b.to_vec());

	arr_a.dot(&arr_b)
}

// ============================================================================================== //
// Matrix

struct Matrix {
	rows: usize,
	cols: usize,
	array: Array1<f32>
}

impl Matrix {
	fn new(rows: usize, cols: usize, array: Array1<f32>) -> Self {
		Self {rows, cols, array}
	}

	fn from_data(data: Vec<Vec<f32>>) -> Self {
		let rows: usize = data.len();
		let cols: usize = data[0].len();
		let mut vector: Vec<f32> = Vec::with_capacity(rows * cols);
		for r in 0..rows {
			for c in 0..cols {
				vector.push(data[r][c]);
			}
		}
		let array: Array1<f32> = Array1::from_vec(vector);

		Self {rows, cols, array}
	}

	fn get(&self, row: usize, col: usize) -> f32 {
		self.array[row * self.cols + col]
	}

	fn get_row(&self, row: usize) -> Self {
		let rows: usize = 1;
		let cols: usize = self.cols;
		let start: usize = row * self.cols;
		let stop: usize = (row + 1) * self.cols;
		let array: Array1<f32> = self.array.slice(s![start..stop]).to_owned();

		Self {rows, cols, array}
	}

	fn dot(&self, other: &Matrix) {
		// TODO!
	}
}

// ============================================================================================== //
// Propogation Trait

trait Propogation {
	// ================================================== //

	fn forward(&mut self, input: Vec<f32>) {}

	// ================================================== //

	fn backward(&mut self, output_error: Vec<f32>, learning_rate: f32) {}

	// ================================================== //
}

// ============================================================================================== //
// Fully Connected Layer Struct

struct FullyConnectedLayer {
	input_size: usize,
	output_size: usize,
	inputs: Vec<f32>,
	outputs: Vec<f32>,
	weights: Vec<f32>, // One for every node in this layer * every node in the next layer
	bias: Vec<f32>, // One for every node in the next layer
	input_error: Vec<f32> // One for every node in this layer
}

impl FullyConnectedLayer {
	// ================================================== //

	fn new(input_size: usize, output_size: usize) -> Self {
		Self {
			input_size,
			output_size,
			inputs: Vec::with_capacity(input_size),
			outputs: Vec::with_capacity(output_size),
			weights: random_vector(input_size * output_size, -1.0, 1.0),
			bias: random_vector(input_size * output_size, -1.0, 1.0),
			input_error: Vec::with_capacity(input_size)
		}
	}

	// ================================================== //

	fn get_weight(&self, from_node: usize, to_node: usize) -> f32 {
		// *from* is rows, *to* is cols
		let index: usize = from_node * self.input_size + to_node;

		self.weights[index]
	}

	// ================================================== //

	fn get_bias(&self, from_node: usize) -> f32 {
		// This vector is 1-dimensional
		self.bias[from_node]
	}

	// ================================================== //
}

impl Propogation for FullyConnectedLayer {
	// ================================================== //

	fn forward(&mut self, input: Vec<f32>) {
		assert_eq!(self.input_size, input.len(), "Wrong dimension for forward()!");
		
		for i in 0..self.input_size {
			self.inputs[i] = input[i];
		}

		// for every node in the next layer,
		for o in 0..self.output_size {
			self.outputs[o] = 0.0; // (zero out the node)
			// take each node in this layer
			for i in 0..self.input_size {
				// and add up this node's value times the corresponding weight
				self.outputs[o] += self.inputs[i] * self.get_weight(i, o);
				// plus the bias
				self.outputs[o] += self.get_bias(i);
			}
		}
	}

	// ================================================== //

	fn backward(&mut self, output_error: Vec<f32>, learning_rate: f32) {
		// Output Error | this vector is length self.output_size

		// Inputs Error | this vector is length self.input_size
		for i in 0..self.input_size {
			self.input_error[i] = 0.0;
			for o in 0..self.output_size {
				self.input_error[i] += output_error[o] * self.get_weight(i, o);
			}
		}
		
		// Weights Error
		let mut weights_error: f32 = 0.0;
		for i in 0..self.input_size {
			for o in 0..self.output_size {
				weights_error += self.inputs[i] * output_error[o];
			}
		}

		// Update parameters
		for w in 0..self.weights.len() {
			self.weights[w] -= learning_rate * weights_error;
		}
		for o in 0..self.output_size {
			self.bias[o] -= learning_rate * output_error[o];
		}
	}

	// ================================================== //
}

// ============================================================================================== //
// Activation Layer Struct

struct ActivationLayer {
	input_size: usize,
	output_size: usize,
	inputs: Vec<f32>,
	outputs: Vec<f32>,
	activation: fn(&Vec<f32>) -> Vec<f32>,
	activation_prime: fn(&Vec<f32>) -> Vec<f32>,
	input_error: Vec<f32>
}

impl ActivationLayer {
	// ================================================== //

	fn new(input_size: usize, output_size: usize, activation: fn(&Vec<f32>) -> Vec<f32>,
		activation_prime: fn(&Vec<f32>) -> Vec<f32>) -> Self {

		Self {
			input_size,
			output_size,
			inputs: Vec::with_capacity(input_size),
			outputs: Vec::with_capacity(output_size),
			activation,
			activation_prime,
			input_error: Vec::with_capacity(input_size)
		}
	}
}

impl Propogation for ActivationLayer {
	// ================================================== //

	fn forward(&mut self, input: Vec<f32>) {
		assert_eq!(self.input_size, input.len(), "Wrong dimension for forward()!");
		let activated_inpts = (self.activation)(&self.inputs);
		for i in 0..self.input_size {
			self.inputs[i] = input[i];
			self.outputs[i] = activated_inpts[i];
		}
	}

	// ================================================== //

	fn backward(&mut self, output_error: Vec<f32>, learning_rate: f32) {
		let activated_prime_outputs = (self.activation_prime)(&self.outputs);
		for i in 0..self.input_size {
			self.input_error[i] = activated_prime_outputs[i] * output_error[i];
		}
	}

	// ================================================== //
}

// ============================================================================================== //
// Activation Functions

fn vector_tanh(vector: &Vec<f32>) -> Vec<f32> {
	(0..vector.len())
		.map(|v| vector[v].tanh())
		.collect()
}

fn vector_tanh_prime(vector: &Vec<f32>) -> Vec<f32> {
	(0..vector.len())
		.map(|v| 1.0 - vector[v].tanh().powf(2.0))
		.collect()
}

// ============================================================================================== //
// Layer Enum

enum Layer {
	FullyConnectedLayer(FullyConnectedLayer),
	ActivationLayer(ActivationLayer)
}

// ============================================================================================== //
// Network Struct

struct Network {
	structure: Vec<usize>,
	layers: Vec<Layer>
}

impl Network {
	// ================================================== //

	fn new(structure: Vec<usize>) -> Self {
		let mut layers: Vec<Layer> = Vec::new();
		for l in 0..(structure.len() - 1) {
			layers.push(Layer::FullyConnectedLayer(FullyConnectedLayer::new(structure[l], structure[l + 1])));
			layers.push(
				Layer::ActivationLayer(ActivationLayer::new(
					structure[l + 1],
					structure[l + 1],
					vector_tanh,
					vector_tanh_prime
				))
			);
		}

		Self {
			structure,
			layers
		}
	}

	// ================================================== //
}

// ============================================================================================== //