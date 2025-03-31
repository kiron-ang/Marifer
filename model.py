"""This module trains a model with data from the "data" directory"""
import tensorflow as tf
import matplotlib.pyplot as plt
def readlines(path):
    """Read file from path and return a list of lines"""
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
    return lines
def writelist(path, list0):
    """Write file at path, where every line is an element from a list"""
    with open(path, "w", encoding="utf-8") as f:
        for l in list0:
            f.write(str(l) + "\n")
train_SMILES = readlines("data/train-SMILES.txt") # Longest string is 28 characters
test_SMILES = readlines("data/test-SMILES.txt") # Longest string is 26 characters
validation_SMILES = readlines("data/validation-SMILES.txt") # Longest string is 27 characters
train_G_atomization = [float(r) for r in readlines("data/train-G_atomization.txt")]
test_G_atomization = [float(r) for r in readlines("data/test-G_atomization.txt")]
validation_G_atomization = [float(r) for r in readlines("data/validation-G_atomization.txt")]
text_vectorization_layer = tf.keras.layers.TextVectorization()
text_vectorization_layer.adapt(train_SMILES)
units = len(text_vectorization_layer.get_vocabulary())
model = tf.keras.models.Sequential([
    text_vectorization_layer,
    tf.keras.layers.Embedding(units, units // 10),
    tf.keras.layers.LSTM(units // 100),
    tf.keras.layers.Dense(1)
])
model.compile(
    loss="huber",
    metrics=["mae", "mape", "mse", "msle"])
history = model.fit(
    tf.data.Dataset.from_tensor_slices((
        tf.constant(train_SMILES), tf.constant(train_G_atomization))).batch(1000),
    epochs=10,
    validation_data=(tf.constant(validation_SMILES), tf.constant(validation_G_atomization))
)
plt.rcParams["font.family"] = "serif"
plt.figure()
plt.plot(history.history["loss"], label="Training Data")
plt.plot(history.history["val_loss"], label="Validation Data")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.savefig("model/loss-epoch.png")
plt.close()
plt.figure()
plt.plot(history.history["mae"], label="Training Data")
plt.plot(history.history["val_mae"], label="Validation Data")
plt.ylabel("MAE")
plt.xlabel("Epoch")
plt.savefig("model/mae-epoch.png")
plt.close()
plt.figure()
plt.plot(history.history["mape"], label="Training Data")
plt.plot(history.history["val_mape"], label="Validation Data")
plt.ylabel("MAPE")
plt.xlabel("Epoch")
plt.savefig("model/mape-epoch.png")
plt.close()
plt.figure()
plt.plot(history.history["mse"], label="Training Data")
plt.plot(history.history["val_mse"], label="Validation Data")
plt.ylabel("MSE")
plt.xlabel("Epoch")
plt.savefig("model/mse-epoch.png")
plt.close()
plt.figure()
plt.plot(history.history["msle"], label="Training Data")
plt.plot(history.history["val_msle"], label="Validation Data")
plt.ylabel("MSLE")
plt.xlabel("Epoch")
plt.savefig("model/msle-epoch.png")
plt.close()
train_predictions = model.predict(tf.constant(train_SMILES))
test_predictions = model.predict(tf.constant(test_SMILES))
validation_predictions = model.predict(tf.constant(validation_SMILES))
writelist("model/train-G_atomization.txt", [p[0] for p in train_predictions])
writelist("model/test-G_atomization.txt", [p[0] for p in test_predictions])
writelist("model/validation-G_atomization.txt", [p[0] for p in validation_predictions])
with open("model/summary.txt", "w", encoding="utf-8") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))
model.save("model/model.keras")
