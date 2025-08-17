import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

NUM_CLASSES = 10
IM_SIZE = 28

KERNEL_SIZE = 5

BATCH_SIZE = 32
EPOCHS = 5

DROPOUT = 0.1

TEST_NUM = 10

# Construct a tf.data.Dataset
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


# Normalize and prepare the datasets
train = train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
train = train.cache()
train = train.shuffle(info.splits['train'].num_examples)
train = train.batch(BATCH_SIZE)
train = train.prefetch(tf.data.AUTOTUNE)
test = test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
test = test.batch(BATCH_SIZE)
test = test.cache()
test = test.prefetch(tf.data.AUTOTUNE)

for example in train.take(1):
    image, label = example[0], example[1]

# Visualize some examples
# i = 1
# print(f"Number: {int(label[i])}\n")
#
# plt.imshow(image[i], cmap='gray')
# plt.show()

# Build the model
model = keras.models.Sequential([
    keras.layers.Input(shape=(IM_SIZE, IM_SIZE, 1)),
    keras.layers.Conv2D(4, (KERNEL_SIZE, KERNEL_SIZE), padding='same', activation='relu'),
    keras.layers.MaxPool2D(),
    # keras.layers.Dropout(0.2),
    # keras.layers.Conv2D(4, (KERNEL_SIZE, KERNEL_SIZE), padding='same', activation='relu'),
    keras.layers.MaxPool2D(),
    keras.layers.Dropout(DROPOUT),

    keras.layers.Permute((3, 1, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(epsilon=1e-08),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

model.summary()

history = model.fit(train,
                    epochs=EPOCHS,
                    validation_data=test)

model.save(f"model{TEST_NUM}-{EPOCHS}.keras")

val_loss, val_accuracy = model.evaluate(test)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
print(history.history.keys())
# plt.plot(history.history['loss'], label='Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(history.history['sparse_categorical_accuracy'], label='Training Accuracy')
plt.plot(history.history['val_sparse_categorical_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()
