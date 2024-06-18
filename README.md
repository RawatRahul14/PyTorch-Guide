# PyTorch-Guide

Welcome to the PyTorch Guide! This repository contains tutorials on how to implement and use PyTorch, a powerful Deep Learning framework.

#### Table of content
* [Introduction](#introduction)
* [Applications](#applications)
    * Computer Vision
    * Natural Language Processing
    * Reinforcement Learning
    * Recommendation System
    * Time Series Analysis
    * Autonomous Systems
* [Installtion](#installation)
* [Tensors](#tensors)
* [Neural Networks](#neural-networks)

## Introduction:
PyTorch is an open-source Deep Learning framework developed by the Facebook's AI Research lab (FAIR). It is widely used in the applications such as Computer Vision (CV) and Natural Language Processing (NLP). 
* **Tensor Computation :** Tensors are the fundamental data structure used in PyTorch for computation. They are similar to NumPy arrays and also provide GPU acceleration.
* **AutoGrad :** Automatic differentiation for creating and training Deep Neural Networks.

## Applications:
PyTorch is used in various domains due to its flexibility, ease of use, and powerful features. 

Some key applications of PyTorch:
1. **Computer Vision (CV)**
    * Image Classification: It is used to make models for classifying images into different categories.
    * Object Detection: YOLO algorithm uses Convolutional Layers (or CNN) to detect objects in an image and is implemented on PyTorch.
    * Generative Models: PyTorch is used to create Generative Adversarial Networks (GANs) for various tasks like style transfer and image super-resolution.
2. **Natural Langauge Processing (NLP)**
    * Text Classification: PyTorch is used to make models to analyze text and classify into categories, such as spam detection.
    * Machine Translation: Architecture models like "Atention is all you need" is implemented in PyTorch due to its flexibilty and ease of use. 
    * Speech Recognition: PyTorch aids in building models that can convert speech into text.
3. **Reinforcement Learning (RL)**
    * Robotics: PyTorch helps in building models which can interact with their environment and learn from it to make better decisions.
    * Game Playing: PyTorch is used to make Agents that can play games like chess by learning optimal strategies through RL algorithms. 
4. **Recommendation System**
    * Collaborative Filtering: PyTorch helps in implementing techniques like vector factorization for recommending items to find similarities between users and items.
    * Content Recommendation: Models can be created using PyTorch to recommend movies, products, music, etc. based on user prefrences.
5. **Time Series Analysis**
    * Anomaly Detection: Model can find unusual patterns or anomaly in the time-series data.
    * Forecasting: Pytorch can be used to make models to predict future values of the time-series data, such as Stock Price, Weather.
6. **Autonomous Systems**
    * Self-driving cars: PyTorch is used to develop decision making systems for self driving cars which helps them to navigate through their environment safely.

## Installation

1. Ensure you are using Python verion 3.6 or higher.


2. Install PyTorch using pip command.
    ```sh
    pip install torch torchvision
    ```


3. Verify the installation
    ```py
    import torch
    print(torch.__version__)
    ```

## Tensors
Tensors are the fundamental data structures used in PyTorch to store, manipulate and do computation, They are similar to NumPy arrays but can utilize GPU acceleration for faster computations.

They can be thought as multidimensional arrays. In mathematics, they known as matrices.

* [Manipulating Tensors](01-Tensors.ipynb)

## Neural Networks

Neural Networks are the fundamental part of the Deep Learning, it allows the model to learn complex patterns in data. PyTorch provides a flexible framework for building and training neural networks.

[Python Notebook for the below code](02-Simple-NN.ipynb)

1. **Importing the required packages.**
    ```py
    import torch
    import torch.nn as nn
    import torch.optim as optim
    ```
2. **Defining a Simple Neural Network:**

    A PyTorch neural network model is made by creating a class which inherits from `nn.Module`. The layers are defined in the `__init__` method and the feed forward is defined in the `forward` method.
    ```py
    class SimpleNN(nn.Module):
        def __init__(self):
            super(SimpleNN, self).__init__()
            # fc1: 1st fully connected layer
            # fc2: 2nd fullu connected layer
            self.fc1 = nn.Linear(10, 50) # Input Layer: 10 nodes, Hidden Layer: 50 nodes
            self.fc2 = nn.Linear(50, 1) # Output Layer: 1 node
        def forward(self, x):
            # Applying the 1st layer
            x = self.fc1(x)
            # Applying the activation function
            x = torch.relu(x)
            # Applying the 2nd layer
            x = self.fc2(x)
            return x
    ```

3. **Creating a model Instance**

    Initiate the model and define the loss function.
    ```py
    # Initializing the model
    model = SimpleNN()
    # Mean Squarred Error loss
    criterion = nn.MSELoss() 
    # SGD optimizer with the 0.01 learning rate
    optimizer = optim.SGD(model.parameters(), lr = 0.01)
    ```

4. **Training the model**
    ```py
    # Example data
    inputs = torch.randn(100, 10)  # 100 samples, each with 10 features
    targets = torch.randn(100, 1)  # 100 target values

    # Training loop
    num_epochs = 1000
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()  # Clear the gradients
        loss.backward()        # Compute the gradients
        optimizer.step()       # Update the parameters

        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}') 
    ```