"""This module trains a model with data from the "data" directory."""
import matplotlib.pyplot as plt
from rdkit import Chem
def readlines(path):
    """Read file from path and return a list of lines"""
    with open(path, "r", encoding="utf-8") as f:
        readlines = [line.strip() for line in f.readlines()]
    return readlines
def writelist(path, list):
    """Write file at path, where every line is an element from a list"""
    with open(path, "w", encoding="utf-8") as f:
        for l in list:
            f.write(l, "\n")
train_smiles = readlines("data/train-SMILES.txt")
test_smiles = readlines("data/test-SMILES.txt")
validation_smiles = readlines("data/validation-SMILES.txt")
train_g_atomization = [float(r) for r in readlines("data/train-G_atomization.txt")]
test_g_atomization = [float(r) for r in readlines("data/test-G_atomization.txt")]
validation_g_atomization = [float(r) for r in readlines("data/validation-G_atomization.txt")]
smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in train_smiles]
writelist("model/SMILES.txt", smiles)
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"
plt.rcParams["font.size"] = 20
plt.figure()
plt.plot(train_g_atomization)
plt.plot(test_g_atomization)
plt.plot(validation_g_atomization)
plt.title("THIS IS A TEST")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.savefig("model/loss-epoch.png")
