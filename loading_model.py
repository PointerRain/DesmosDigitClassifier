import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

model = keras.saving.load_model("model5-20.keras")
model.summary()


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

# for i, layer in enumerate(model.layers):
#     print('\n')
#     print(f"Layer {i}: {layer.name}")
#     weights = model.get_layer(index=i).get_weights()
#     for w, weight in enumerate(weights):
#         print(f"Layer {i}, weight {w}: {weight.shape}")
#         print(f"Layer {i}, weight {w}: {weight}")
#     print('\n')


kernels, biases = model.get_layer(index=0).get_weights()
print(f"Kernels shape: {kernels.shape}")
print(f"Kernels: {kernels}")
L = []
for i in range(kernels.shape[3]):
    L.extend([round(float(w),8) for w in kernels[:, :, 0, i].flatten()])
    print(f"Kernels {i}: {[float(w) for w in kernels[:, :, 0, i].flatten()]}")
print(L)
print(f"Biases: {biases}")
print(f"Biases: {[float(b) for b in list(biases)]}")


weights, biases = model.get_layer(index=6).get_weights()
print(f"Weights shape: {weights.shape}")
print(f"Weights: {weights}")
L = []
for i in range(weights.shape[1]):
    L.extend([round(float(w), 8) for w in weights[:, i].flatten()])
    print(f"Weights {i}: {[round(float(w), 8) for w in weights[:, i].flatten()]}")
print(L)
print(f"Biases shape: {biases.shape}")
print(f"Biases: {[float(b) for b in list(biases)]}")