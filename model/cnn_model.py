import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn_model():
    model = models.Sequential()

    # First Convolution Block
    model.add(layers.Conv2D(32, (3,3), activation='relu',
                            input_shape=(32,32,3)))
    model.add(layers.MaxPooling2D((2,2)))

    # Second Convolution Block
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))

    # Third Convolution Block
    model.add(layers.Conv2D(64, (3,3), activation='relu'))

    # Fully Connected Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    return model