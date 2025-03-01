number = 0

print(number)
import tensorflow
import tensorflow_datasets
import seaborn

qm9 = tensorflow_datasets.load("qm9/original")

train = qm9["train"]

molecule_graphs = []

for molecule in train:
    number += 1
    smiles = molecule["SMILES"].numpy().decode()

print(number)