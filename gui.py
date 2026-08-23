import json
import tkinter as tk
from tkinter import messagebox, ttk
# Import your compiled PyO3 module
import rust


class NeuralNetApp(tk.Tk):

	def __init__(self):
		super().__init__()
		self.title("Rust Neural Network GUI")
		self.geometry("800x700")

		self.net = None
		self._build_ui()

	def _build_ui(self):
		notebook = ttk.Notebook(self)
		notebook.pack(fill="both", expand=True, padx=10, pady=10)

		# Tab 1: Configuration & Training
		train_tab = ttk.Frame(notebook)
		notebook.add(train_tab, text="1. Model & Training")
		self._build_train_tab(train_tab)

		# Tab 2: Inference / Prediction
		predict_tab = ttk.Frame(notebook)
		notebook.add(predict_tab, text="2. Inference")
		self._build_predict_tab(predict_tab)

	def _build_train_tab(self, parent):
		# --- Network Setup ---
		net_frame = ttk.LabelFrame(parent, text=" Network Architecture ")
		net_frame.pack(fill="x", padx=10, pady=5)

		ttk.Label(
			net_frame, text="Structure (e.g., 2, 4, 1 for 2-in, 4-hidden, 1-out):"
		).grid(row=0, column=0, padx=5, pady=5, sticky="w")
		self.struct_entry = ttk.Entry(net_frame, width=30)
		self.struct_entry.insert(0, "2, 4, 1")
		self.struct_entry.grid(row=0, column=1, padx=5, pady=5)

		ttk.Button(
			net_frame, text="Initialize Network", command=self.init_network
		).grid(row=0, column=2, padx=5, pady=5)

		# --- Hyperparameters ---
		hp_frame = ttk.LabelFrame(parent, text=" Hyperparameters ")
		hp_frame.pack(fill="x", padx=10, pady=5)

		ttk.Label(hp_frame, text="Learning Rate:").grid(
			row=0, column=0, padx=5, pady=5
		)
		self.lr_entry = ttk.Entry(hp_frame, width=10)
		self.lr_entry.insert(0, "0.1")
		self.lr_entry.grid(row=0, column=1, padx=5, pady=5)

		ttk.Label(hp_frame, text="Epochs:").grid(
			row=0, column=2, padx=5, pady=5
		)
		self.epochs_entry = ttk.Entry(hp_frame, width=10)
		self.epochs_entry.insert(0, "10000")
		self.epochs_entry.grid(row=0, column=3, padx=5, pady=5)

		ttk.Label(hp_frame, text="Update Freq:").grid(
			row=0, column=4, padx=5, pady=5
		)
		self.freq_entry = ttk.Entry(hp_frame, width=10)
		self.freq_entry.insert(0, "1000")
		self.freq_entry.grid(row=0, column=5, padx=5, pady=5)

		# --- Data Input ---
		data_frame = ttk.LabelFrame(parent, text=" Data Input (2D JSON Array) ")
		data_frame.pack(fill="both", expand=True, padx=10, pady=5)

		ttk.Label(data_frame, text="Input Data (X):").grid(
			row=0, column=0, sticky="w", padx=5
		)
		self.x_text = tk.Text(data_frame, height=6, width=40)
		self.x_text.insert(
			"1.0", "[\n  [0.0, 0.0],\n  [0.0, 1.0],\n  [1.0, 0.0],\n  [1.0, 1.0]\n]"
		)
		self.x_text.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

		ttk.Label(data_frame, text="Target Data (Y):").grid(
			row=0, column=1, sticky="w", padx=5
		)
		self.y_text = tk.Text(data_frame, height=6, width=40)
		self.y_text.insert(
			"1.0", "[\n  [0.0],\n  [1.0],\n  [1.0],\n  [0.0]\n]"
		)
		self.y_text.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")

		data_frame.columnconfigure(0, weight=1)
		data_frame.columnconfigure(1, weight=1)

		# --- Actions & Status ---
		ttk.Button(parent, text="Start Training", command=self.train_network).pack(
			pady=5
		)

		self.status_label = ttk.Label(
			parent, text="Status: Network not initialized", font=("Arial", 10, "bold")
		)
		self.status_label.pack(pady=5)

	def _build_predict_tab(self, parent):
		ttk.Label(
			parent, text="Inference Input (2D JSON Array):"
		).pack(anchor="w", padx=10, pady=5)

		self.pred_input_text = tk.Text(parent, height=6)
		self.pred_input_text.insert(
			"1.0", "[\n  [0.0, 1.0],\n  [1.0, 1.0]\n]"
		)
		self.pred_input_text.pack(fill="x", padx=10, pady=5)

		ttk.Button(parent, text="Run Prediction", command=self.predict).pack(
			pady=5
		)

		ttk.Label(parent, text="Prediction Output:").pack(
			anchor="w", padx=10, pady=5
		)
		self.pred_output_text = tk.Text(parent, height=8, state="disabled")
		self.pred_output_text.pack(fill="both", expand=True, padx=10, pady=5)

	def init_network(self):
		try:
			struct_str = self.struct_entry.get()
			structure = [int(x.strip()) for x in struct_str.split(",") if x.strip()]
			self.net = rust.Network(structure)
			self.status_label.config(
				text=f"Status: Network initialized with structure {structure}"
			)
			messagebox.showinfo("Success", "Network initialized successfully!")
		except Exception as e:
			messagebox.showerror("Error", f"Failed to initialize network: {e}")

	def train_network(self):
		if not self.net:
			messagebox.showwarning(
				"Warning", "Please initialize the network first."
			)
			return

		try:
			x_data = json.loads(self.x_text.get("1.0", tk.END))
			y_data = json.loads(self.y_text.get("1.0", tk.END))
			lr = float(self.lr_entry.get())
			epochs = int(self.epochs_entry.get())
			freq = int(self.freq_entry.get())

			self.status_label.config(text="Status: Training in progress...")
			self.update_idletasks()

			# Call PyO3 exposed train method
			self.net.train(
				input_data=x_data,
				target=y_data,
				epochs=epochs,
				learning_rate=lr,
				update_frequency=freq,
			)

			self.status_label.config(text="Status: Training Complete!")
			messagebox.showinfo("Success", "Training completed successfully!")
		except Exception as e:
			self.status_label.config(text="Status: Training failed")
			messagebox.showerror("Error", f"Failed during training: {e}")

	def predict(self):
		if not self.net:
			messagebox.showwarning(
				"Warning", "Please initialize and train the network first."
			)
			return

		try:
			raw_data = json.loads(self.pred_input_text.get("1.0", tk.END))

			# Ensure input is a 2D array of floats
			pred_data = [[float(val) for val in row] for row in raw_data]
			print(pred_data)

			# Call Rust network
			results = self.net.predict(pred_data)

			self.pred_output_text.config(state="normal")
			self.pred_output_text.delete("1.0", tk.END)
			self.pred_output_text.insert("1.0", json.dumps(results, indent=2))
			self.pred_output_text.config(state="disabled")
		except Exception as e:
			messagebox.showerror("Error", f"Failed to perform prediction: {e}")


if __name__ == "__main__":
	app = NeuralNetApp()
	app.mainloop()