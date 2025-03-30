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
train_SMILES = readlines("data/train-SMILES.txt") # The longest SMILES string has 28 characters
test_SMILES = readlines("data/test-SMILES.txt") # 26 characters
validation_SMILES = readlines("data/validation-SMILES.txt") # 27 characters
train_G_atomization = [float(r) for r in readlines("data/train-G_atomization.txt")]
test_G_atomization = [float(r) for r in readlines("data/test-G_atomization.txt")]
validation_G_atomization = [float(r) for r in readlines("data/validation-G_atomization.txt")]
def returnmodel(string_list, float_list):
    """Define, compile, fit, and return a new Sequential model"""
    text_vectorization_layer = tf.keras.layers.TextVectorization()
    text_vectorization_layer.adapt(string_list)
    units = len(text_vectorization_layer.get_vocabulary())
    model = tf.keras.models.Sequential([
        text_vectorization_layer,
        tf.keras.layers.Embedding(units, units),
        tf.keras.layers.LSTM(units),
        tf.keras.layers.Dense(1)
    ])
    model.compile(loss="huber")
    print("CONVERTING STRING")
    string_tensor = tf.constant(string_list)
    print("CONVERTING FLOAT")
    float_tensor = tf.constant(float_list)
    print("FIT MODEL")
    return model.fit(string_tensor, float_tensor)
plt.rcParams["font.family"] = "serif"
plt.figure()
# plt.plot(returnmodel(train_SMILES, train_G_atomization).history["loss"], label="Train")
plt.plot(returnmodel(test_SMILES, test_G_atomization).history["loss"], label="Test")
plt.plot(returnmodel(validation_SMILES, validation_G_atomization).history["loss"],
            label="Validation")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend()
plt.savefig("model/loss-epoch.png")
