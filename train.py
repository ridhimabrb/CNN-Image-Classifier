import tensorflow as tf
from tensorflow.keras import datasets
from model.cnn_model import build_cnn_model

# Load CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

# Normalize pixel values
X_train, X_test = X_train / 255.0, X_test / 255.0

# Build model
model = build_cnn_model()

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(
    X_train, y_train,
    epochs=10,
    validation_data=(X_test, y_test)
)

# Save model
model.save("cnn_image_classifier.h5")