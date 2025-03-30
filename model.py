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
def returnmodel(string_list, float_list):
    """Define, compile, fit, and return a new Sequential model"""
    text_vectorization_layer = tf.keras.layers.TextVectorization()
    text_vectorization_layer.adapt(string_list)
    units = len(text_vectorization_layer.get_vocabulary())
    model = tf.keras.models.Sequential([
        text_vectorization_layer,
        tf.keras.layers.Embedding(units, units // 10),
        tf.keras.layers.LSTM(units // 100),
        tf.keras.layers.Dense(1)
    ])
    model.compile(loss="huber")
    string_tensor = tf.constant(string_list)
    float_tensor = tf.constant(float_list)
    history = model.fit(tf.data.Dataset.from_tensor_slices((string_tensor,
                            float_tensor)).batch(1000),epochs=10)
    return model, history
trainmodel, trainmodel_history = returnmodel(train_SMILES, train_G_atomization)
testmodel, testmodel_history = returnmodel(test_SMILES, test_G_atomization)
validationmodel, validationmodel_history = returnmodel(validation_SMILES, validation_G_atomization)
plt.rcParams["font.family"] = "serif"
plt.figure()
plt.plot(trainmodel_history.history["loss"], label="Train")
plt.plot(testmodel_history.history["loss"], label="Test")
plt.plot(validationmodel_history.history["loss"], label="Validation")
plt.title("Fitting Model to 3 Different Splits")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend()
plt.savefig("model/loss-epoch.png")
plt.close()
trainmodel.save("model/trainmodel.h5")
train_predictions = trainmodel.predict(tf.constant(train_SMILES))
test_predictions = trainmodel.predict(tf.constant(test_SMILES))
validation_predictions = trainmodel.predict(tf.constant(validation_SMILES))
writelist("model/train-G_atomization.txt", train_predictions)
writelist("model/test-G_atomization.txt", test_predictions)
writelist("model/validation-G_atomization.txt", validation_predictions)
