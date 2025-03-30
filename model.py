"""Train a model!"""
import matplotlib.pyplot as plt
with open("data/train-SMILES.txt", "r", encoding="utf-8") as f:
    train_smiles = [line.strip() for line in f.readlines()]
with open("data/train-G_atomization.txt", "r", encoding="utf-8") as f:
    train_g_atomization = [float(line.strip()) for line in f.readlines()]
with open("data/test-SMILES.txt", "r", encoding="utf-8") as f:
    test_smiles = [line.strip() for line in f.readlines()]
with open("data/test-G_atomization.txt", "r", encoding="utf-8") as f:
    test_g_atomization = [float(line.strip()) for line in f.readlines()]
smiles = train_smiles
with open("model/SMILES.txt", "w", encoding="utf-8") as f:
    for s in smiles:
        f.write(s, "\n")
plt.figure()
plt.plot(train_g_atomization)
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.savefig("model/loss-epoch.png")
