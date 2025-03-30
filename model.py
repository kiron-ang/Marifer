"""This module trains a model with data from the "data" directory."""
import matplotlib.pyplot as plt
from rdkit import Chem
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
train_SMILES = readlines("data/train-SMILES.txt")
test_SMILES = readlines("data/test-SMILES.txt")
validation_SMILES = readlines("data/validation-SMILES.txt")
train_G_atomization = [float(r) for r in readlines("data/train-G_atomization.txt")]
test_G_atomization = [float(r) for r in readlines("data/test-G_atomization.txt")]
validation_G_atomization = [float(r) for r in readlines("data/validation-G_atomization.txt")]
SMILES = [Chem.MolToSMILES(Chem.MolFromSMILES(s)) for s in train_SMILES]
G_atomization = train_G_atomization
writelist("model/SMILES.txt", SMILES)
writelist("model/G_atomization.txt", G_atomization)
plt.rcParams["font.family"] = "serif"
plt.figure()
plt.plot(train_G_atomization)
plt.plot(test_G_atomization)
plt.plot(validation_G_atomization)
plt.title("THIS IS A TEST")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.savefig("model/loss-epoch.png")
