# neuralnetwork-model


# 🧠 Simple Neural Network for Iris Flower Classification

This project demonstrates a basic feed-forward neural network built in PyTorch for classifying the Iris dataset. The repository includes code for model creation, dataset handling, training, and evaluation, with a Jupyter Notebook to showcase usage.

---

## 🗂 Project Structure

- **`model.py`**: Defines the `NeuralNetwork` model class, with functions for training and evaluating the model.
- **`data_utils.py`**: Contains `BuildDataset`, which loads and preprocesses the Iris dataset and returns DataLoader objects for training and testing.
- **`example_notebook.ipynb`**: A Jupyter notebook example, demonstrating how to use the model and dataset functions to train and evaluate the neural network.

---

## 📊 Dataset

This project uses the [Iris dataset](https://archive.ics.uci.edu/ml/datasets/Iris), which contains:
- **150 samples** of iris flowers, each with **4 features**:
  - Sepal length
  - Sepal width
  - Petal length
  - Petal width
- **3 classes** for classification: Setosa, Versicolor, and Virginica.

---

## ⚙️ Requirements

- **Python 3.8+**
- **Libraries**:
  - PyTorch
  - scikit-learn
  
Install dependencies with:
```bash
pip install torch scikit-learn
