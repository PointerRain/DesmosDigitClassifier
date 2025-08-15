import keras

model = keras.saving.load_model("model3-5.keras")
model.summary()
print(model.get_layer(index=0).get_config())