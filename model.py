"""This module trains a model with data from the "data" directory"""
import matplotlib.pyplot as plt
from rdkit import Chem
from tensorflow.keras import layers, models
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
def compilemodel():
    """Compile a new Sequential model"""
    model = models.Sequential([
        layers.TextVectorization(),
        layers.Embedding(28, 28),
        layers.LSTM(28),
        layers.Dense(28)
    ])
    model.compile()
    return model
def fitmodel(data, labels):
    """Fit an existing model to existing data"""
    vectorizer = layers.TextVectorization()
    vectorizer.adapt(train_SMILES)
    train_features = vectorizer(train_SMILES)
    validation_features = vectorizer(validation_SMILES)
    train_labels = np.array(train_G_atomization)
    validation_labels = np.array(validation_G_atomization)
SMILES = [Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in train_SMILES]
G_atomization = train_G_atomization
writelist("model/SMILES.txt", SMILES)
writelist("model/G_atomization.txt", G_atomization)
plt.rcParams["font.family"] = "serif"
plt.figure()
plt.plot(train_G_atomization, label="Train")
plt.plot(test_G_atomization, label="Test")
plt.plot(validation_G_atomization, label="Validation")
plt.title("THIS IS A TEST")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend()
plt.savefig("model/loss-epoch.png")
