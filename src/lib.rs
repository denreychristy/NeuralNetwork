
use std::assert_eq;

use pyo3::prelude::*;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    Ok(())
}

// ============================================================================================== //
// Matrix Struct

#[derive(Clone, Debug)]
pub struct Matrix {
	pub rows: usize,
	pub cols: usize,
	pub data: Vec<f32>
}

impl Matrix {
	// ================================================== //

	pub fn new(rows: usize, cols: usize, data: Vec<f32>) -> Self {
		assert_eq!(rows * cols, data.len(), "Dimension mismatch in Matrix::new().");
		Self {rows, cols, data}
	}

	// ================================================== //
	
	pub fn zeros(rows: usize, cols: usize) -> Self {
		Self {
			rows,
			cols,
			data: vec![0.0; rows * cols]
		}
	}

	// ================================================== //

	pub fn random(rows: usize, cols: usize, seed: &mut u64) -> Self {
		let mut data = Vec::with_capacity(rows * cols);
		let scale = (2.0 / rows as f32).sqrt();

		for _ in 0..(rows * cols) {
			*seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
			let rnd = ((*seed >> 33) as f32) / (u32::MAX as f32);
			data.push((rnd * 2.0 - 1.0) * scale);
		}

		Self {rows, cols, data}
	}

	// ================================================== //

	pub fn dot(&self, other: &Matrix) -> Matrix {
		assert_eq!(self.cols, other.rows, "Dimension mismatch in Matrix::dot().");
		let mut result = Matrix::zeros(self.rows, other.cols);
		
		for i in 0..self.cols {
			for k in 0..self.cols {
				let r = self.data[i * self.cols + k];
				for j in 0..other.cols {
					result.data[i * other.cols + j] += r * other.data[k * other.cols + j];
				}
			}
		}

		result
	}

	// ================================================== //

	fn transpose(&self) -> Self {
		let mut result = Matrix::zeros(self.cols, self.rows);
		for r in 0..self.rows {
			for c in 0..self.cols {
				result.data[c * self.rows + r] = self.data[r * self.cols + c];
			}
		}

		result
	}
}