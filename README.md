# neuralnetwork-model


# Simple Neural Network for Iris Flower Classification

This project demonstrates a basic feed-forward neural network built in PyTorch for classifying the Iris dataset. The repository includes code for model creation, dataset handling, training, and evaluation, with a Jupyter Notebook to showcase usage.

---

## Project Structure

- **`model.py`**: Defines the `NeuralNetwork` model class, with functions for training and evaluating the model.
- **`data_utils.py`**: Contains `BuildDataset`, which loads and preprocesses the Iris dataset and returns DataLoader objects for training and testing.
- **`run.py`**: Demonstrates how to use the model and dataset functions to train and evaluate the neural network.

---

## Dataset

This project uses the [Iris dataset](https://archive.ics.uci.edu/ml/datasets/Iris), which contains:
- **150 samples** of iris flowers, each with **4 features**:
  - Sepal length
  - Sepal width
  - Petal length
  - Petal width
- **3 classes** for classification: Setosa, Versicolor, and Virginica.

---

##  Requirements

- **Python 3.8+**
- **Libraries**:
  - PyTorch
  - scikit-learn
  
Install dependencies with:
```bash
pip install torch scikit-learn



## Contact

Feel free to reach out if you have questions or feedback! You can contact me via kevinabizeiddaou@gmail.com.

## Acknowledgments
This project uses the popular Iris dataset, often used for machine learning demonstrations and provided by the UCI Machine Learning Repository.

## Conclusion
In this project, we implemented a simple feed-forward neural network to classify the Iris dataset, showcasing the fundamentals of neural network design and usage with PyTorch. The example provided in this repository serves as a foundation for further experimentation and learning in machine learning and neural networks.

