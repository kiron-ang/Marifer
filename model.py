"""
This module defines and trains a generative model to produce novel SMILES strings.
It includes functionality to introduce diversity in the generated molecules
using temperature sampling and saves the molecular weights of the generated molecules.
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Create model directory if it does not exist
os.makedirs("model", exist_ok=True)

# Load the SMILES strings from the training data file
with open("data/train-SMILES.txt", "r", encoding="utf-8") as file:
    smiles_strings = file.readlines()

# Tokenize the SMILES strings
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(smiles_strings)
total_chars = len(tokenizer.word_index) + 1

# Prepare the sequences for training
sequences = []
for smile in smiles_strings:
    encoded = tokenizer.texts_to_sequences([smile])[0]
    for i in range(1, len(encoded)):
        sequence = encoded[:i+1]
        sequences.append(sequence)

max_sequence_len = max(len(seq) for seq in sequences)
sequences = pad_sequences(sequences, maxlen=max_sequence_len, padding="pre")
X, y = sequences[:, :-1], sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=total_chars)

# Define the model architecture
model = Sequential([
    Embedding(total_chars, 50, input_length=max_sequence_len-1),
    LSTM(10, return_sequences=True),
    LSTM(10),
    Dense(total_chars, activation="softmax")
])

model.compile(loss="categorical_crossentropy", optimizer="adam")

# Train the model
plt.figure()
plt.plot(model.fit(X, y, epochs=10).history["loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("model/loss-epoch.png")
plt.close()

def calculate_molecular_weight(smiles):
    """
    Calculate the molecular weight of a given SMILES string using RDKit.

    Args:
        smiles (str): SMILES string of the molecule.

    Returns:
        float: Molecular weight of the molecule. Returns 0 if the molecule is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Descriptors.MolWt(mol)
    return 0

def sample_with_temperature(preds, temp=1.0):
    """
    Sample an index from a probability array using temperature sampling.

    Args:
        preds (numpy.ndarray): Array of prediction probabilities.
        temp (float): Temperature parameter to control diversity. Default is 1.0

    Returns:
        int: Index of the sampled element.
    """
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-10) / temp
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# Generate new SMILES strings with temperature sampling
generated_smiles = []
molecular_weights = []
for i in range(100):
    text = []
    for _ in range(max_sequence_len):
        token_list = tokenizer.texts_to_sequences(["".join(text)])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding="pre")
        predictions = model.predict(token_list, verbose=0)[0]
        next_index = sample_with_temperature(predictions, 0.8)
        next_char = tokenizer.index_word[next_index]
        text.append(next_char)
        if next_char == "\n":
            break
    generated_smiles.append("".join(text))
    molecular_weights.append(calculate_molecular_weight("".join(text)))

# Save the generated SMILES strings to a file
with open("model/SMILES.txt", "w", encoding="utf-8") as output_file:
    for g in generated_smiles:
        output_file.write(g + "\n")

# Save the molecular weights to a file
with open("model/MolWt.txt", "w", encoding="utf-8") as mw_file:
    for mw in molecular_weights:
        mw_file.write(f"{mw}\n")
