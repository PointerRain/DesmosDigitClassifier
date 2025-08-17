import keras
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

model = keras.saving.load_model("model7-20.keras")
model.summary()

print(model.layers)


# Testing
# data = (np.zeros((28,28,1))+0.5)[np.newaxis,...]
# data = np.flip(np.array(D).reshape((28, 28, 1)), axis=2)[np.newaxis, ...]
# import matplotlib.pyplot as plt

# plt.imshow(data[0], cmap='gray')
# plt.show()
# for layer in model.layers:
#     data = layer(data)
#     print(layer)
#     print(data)

# input()

# Test the model on the MNIST dataset
(train, test), info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label

BATCH_SIZE = 128

test = test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
test = test.batch(BATCH_SIZE)
test = test.cache()
test = test.prefetch(tf.data.AUTOTUNE)

val_loss, val_accuracy = model.evaluate(test)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")


# Get the weights and biases of the model

kernels, biases = model.get_layer(index=0).get_weights()
print(f"Kernels shape: {kernels.shape}")
print(f"Kernels: {kernels}")
L = []
for i in range(kernels.shape[3]):
    L.extend([round(float(w), 8) for w in np.transpose(kernels[:, :, 0, i],(0,1)).flatten()])
    print(f"Kernels {i}: {[float(w) for w in kernels[:, :, 0, i].flatten()]}")
print(L)
print(f"Biases: {biases}")
print(f"Biases: {[float(b) for b in list(biases)]}")

print('\n\n\n')

weights, biases = model.get_layer(index=7).get_weights()
print(f"Weights shape: {weights.shape}")
print(f"Weights: {weights}")
L = []
for i in range(weights.shape[1]):
    L.extend([round(float(w), 8) for w in weights[:, i].flatten()])
    print(f"Weights {i}: {[round(float(w), 8) for w in weights[:, i].flatten()]}")
print(L)
print(f"Biases shape: {biases.shape}")
print(f"Biases: {[float(b) for b in list(biases)]}")
