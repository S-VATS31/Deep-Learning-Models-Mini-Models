# Shaan Vats' Machine Learning Repository

Welcome to my collection of machine learning architectures and projects! This repository showcases various machine learning models and implementations using **PyTorch**, along with other low-level techniques that I’ve developed.

---

## What You’ll Find Here

- **State-of-the-Art ML Architectures**: Explore a variety of models implemented from scratch or based on cutting-edge research papers.
- **PyTorch Implementations**: Code focused on deep learning with PyTorch, including custom neural networks, optimization techniques, and more.
- **Low-level Implementations**: Projects that implement algorithms from the ground up, providing an in-depth understanding of the core principles without relying on high-level libraries.

---

## Features

- **Reproducible Results**: Every project aims for reproducibility with clear instructions.
- **Clear Documentation**: Each project is documented to ensure easy understanding and usage.
- **Modular Code**: The code is designed to be modular, making it easy to adapt and extend for your own purposes.

---

## Important Projects

### Project 1: Traffic Volume Prediction Using Neural Networks

This project demonstrates predicting traffic volume using a custom deep learning model built with **PyTorch**. The model is trained on the **Metro Interstate Traffic Volume dataset** and uses a fully connected neural network to make predictions based on various features like time of day, weather, and holidays.

#### Key Features:
- **Data Preprocessing**: Extracts features from `date_time`, handles missing values, encodes categorical variables, and normalizes the dataset.
- **Neural Network**: A fully connected network with 4 hidden layers, ReLU activation, and Xavier weight initialization. The output is a single neuron predicting traffic volume.
- **Regularization**: Implements **L1 regularization** to reduce overfitting.
- **Evaluation**: Uses **MSE loss** and evaluates the model using **MAE, RMSE**, and **R² metrics**.

#### Training:
The model is trained using the **AdamW optimizer** with a **Cosine Annealing Learning Rate Scheduler** for 6000 iterations.

#### Results:
The model’s performance is evaluated on test data, and predictions are visualized against actual values with a scatter plot.

#### Troubles and Overcoming:
During the initial training of the model, I was surprised to find that the untrained model performed almost as well as the trained version in predicting traffic volume! This led me to dive deeper into the model’s design. I spent countless hours reviewing research papers and exploring the finer details of loss functions, optimization algorithms, tensor shapes, and more. After some intense experimentation, the model finally began to generate traffic volume predictions within a highly impressive range, validating the effort and adjustments made along the way!

---

### Dependencies:
- PyTorch
- Pandas
- Numpy
- Matplotlib
- Scikit-learn

---

### Project 2: Low-Level Backpropagation Implementation

This project demonstrates the implementation of backpropagation from scratch, focusing on the core concepts of **forward** and **backward propagation**. It provides a deeper understanding of the inner workings of neural networks by bypassing high-level libraries like PyTorch or TensorFlow. A low-level implementation like this requires extensive knowledge of the **Chain Rule of Calculus** and the proper **matrix shapes** for debugging purposes.

#### Key Features:
- **Neural Network Architecture**: 
  - A fully connected network consisting of:
    - Input layer
    - Hidden layer
    - Output layer
- **Forward Propagation**: 
  - Implements a **weighted sum** of inputs followed by a **sigmoid activation** function.
- **Backpropagation**: 
  - Custom implementation of the backpropagation algorithm, including:
    - Gradient computation
    - Parameter updates
- **Activation Functions**: 
  - Uses the **sigmoid** activation function and its derivative for gradient calculation.
- **Parameter Initialization**: 
  - Implements **Xavier initialization** to prevent issues like vanishing/exploding gradients.
- **Training Loop**: 
  - The network is trained using **gradient descent** with a learning rate (`alpha`) of `1e-4`.

#### Training Process:
- **Forward Pass**: Propagates the inputs through the network.
- **Backward Pass**: Updates the weights using the computed gradients.

#### Troubles and Overcoming:
This project was a massive challenge to debug. It not only tested my calculus skills but also pushed me to think critically about visualizing high-dimensional tensors. I spent hours debugging matrix dimensions, using techniques like transpositions, to ensure everything worked as expected. This project was an invaluable learning experience, as it highlighted the difference between having a theoretical understanding and actually having to implement and program those concepts.

---

### Dependencies:
- Numpy
