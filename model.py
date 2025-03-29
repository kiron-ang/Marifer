"""Train Long Short-Term Memory Model!"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

file_path = 'data/train-SMILES.txt'
with open(file_path, 'r') as file:
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

max_sequence_len = max([len(seq) for seq in sequences])
sequences = pad_sequences(sequences, maxlen=max_sequence_len, padding='pre')

X, y = sequences[:, :-1], sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=total_chars)

# Step 3: Build the Model
model = Sequential([
    Embedding(total_chars, 50, input_length=max_sequence_len-1),
    LSTM(100, return_sequences=True),
    LSTM(100),
    Dense(total_chars, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, y, epochs=100, batch_size=64)

def generate_smiles(seed_text, max_length):
    """Generate new SMILES strings!"""
    for _ in range(max_length):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = np.argmax(model.predict(token_list), axis=-1)
        next_char = tokenizer.index_word[predicted[0]]
        seed_text += next_char
        if next_char == '\n':
            break
    return seed_text

seed_text = "C"
generated_smiles = generate_smiles(seed_text, max_sequence_len)
print("Generated SMILES:", generated_smiles)
