"""Train a model!"""
with open("data/train-SMILES.txt", "r", encoding="utf-8") as f:
    train_smiles = f.readlines()
with open("data/train-G_atomization.txt", "r", encoding="utf-8") as f:
    train_g_atomization = f.readlines()
print(train_smiles[0:5])
print(train_g_atomization[0:5])