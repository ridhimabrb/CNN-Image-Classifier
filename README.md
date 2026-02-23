# CNN Image Classifier

This project implements a Convolutional Neural Network (CNN) for image classification using the CIFAR-10 dataset.

## Dataset
CIFAR-10 consists of 60,000 color images of size 32x32 belonging to 10 classes:
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

## Training and Testing Strategy

The CIFAR-10 dataset is pre-split into:
- 50,000 training images (80%)
- 10,000 testing images (20%)

## Technologies Used
- Python
- TensorFlow / Keras
- NumPy
- Matplotlib

## Model Architecture
- Convolution + ReLU
- Max Pooling
- Fully Connected layers
- Softmax output layer

## Results
Achieved approximately 70–75% validation accuracy on CIFAR-10.
![alt text](image.png)

## How the CNN Learns Visual Features

Unlike traditional machine learning models, this CNN does not rely on manually engineered features. Instead, it automatically learns hierarchical visual representations directly from raw image pixels.
- Early convolution layers learn low-level features such as edges, corners, and color gradients.
- Intermediate layers combine these features to detect shapes and textures.
- Deeper layers learn high-level object-specific features such as wheels, faces, or wings.

This hierarchical feature learning is what makes CNNs effective for image classification tasks.

## Optimization and Learning Process

The model is trained using the Adam optimizer, which combines the advantages of momentum and adaptive learning rates.

During training:
- The loss function (Sparse Categorical Cross-Entropy) measures prediction error.
- Backpropagation computes gradients of the loss with respect to model parameters.
- The optimizer updates weights to minimize the loss iteratively.

This process is repeated across multiple epochs until convergence.

## Future Improvements

- Apply data augmentation to improve generalization
- Introduce batch normalization and dropout
- Extend the model to classify custom user-uploaded images
- Deploy the model using a web interface (e.g., Streamlit)
