"""PLACEHOLDER"""
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
os.makedirs("model", exist_ok=True)
with open("data/train-SMILES.txt", "r", encoding="utf-8") as file:
    smiles_strings = file.readlines()
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(smiles_strings)
total_chars = len(tokenizer.word_index) + 1
sequences = []
for smile in smiles_strings:
    encoded = tokenizer.texts_to_sequences([smile])[0]
    for i in range(1, len(encoded)):
        sequence = encoded[:i+1]
        sequences.append(sequence)
# The longest SMILES string in the training data is 29!
max_sequence_len = max(len(seq) for seq in sequences)
sequences = pad_sequences(sequences, maxlen=max_sequence_len, padding="pre")
X, y = sequences[:, :-1], sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=total_chars)
model = Sequential([
    Embedding(total_chars, 50, input_length=max_sequence_len-1),
    LSTM(100, return_sequences=True),
    LSTM(100),
    Dense(total_chars, activation="softmax")
])
model.compile(loss="categorical_crossentropy", optimizer="adam")
plt.figure()
plt.plot(model.fit(X, y, epochs=100).history["loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("model/loss-epoch.png")
plt.close()
generated_smiles = []
for i in range(1000):
    text = []
    for _ in range(max_sequence_len):
        token_list = tokenizer.texts_to_sequences(["".join(text)])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding="pre")
        predicted = np.argmax(model.predict(token_list), axis=-1)
        next_char = tokenizer.index_word[predicted[0]]
        text.append(next_char)
        if next_char == "\n":
            break
    generated_smiles.append("".join(text))
with open("model/output-SMILES.txt", "w", encoding="utf-8") as output_file:
    for g in generated_smiles:
        output_file.write(g)
