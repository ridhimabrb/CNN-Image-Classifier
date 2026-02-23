import tensorflow as tf
from tensorflow.keras import datasets

# Load model
model = tf.keras.models.load_model("cnn_image_classifier.h5")

# Load test data
(_, _), (X_test, y_test) = datasets.cifar10.load_data()
X_test = X_test / 255.0

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)