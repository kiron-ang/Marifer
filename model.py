"""???"""
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import matplotlib.pyplot as plt

def load_smiles(file_path):
    """
    Load SMILES strings from a file.

    Args:
        file_path (str): The path to the SMILES file.

    Returns:
        list: A list of SMILES strings.
    """
    with open(file_path, 'r') as file:
        smiles = file.readlines()
    return [s.strip() for s in smiles]

def save_smiles(file_path, smiles):
    """
    Save SMILES strings to a file.

    Args:
        file_path (str): The path to the output file.
        smiles (list): A list of SMILES strings to save.
    """
    with open(file_path, 'w') as file:
        for s in smiles:
            file.write(f"{s}\n")

def preprocess_smiles(smiles, tokenizer, max_length):
    """
    Preprocess SMILES strings by tokenizing and padding.

    Args:
        smiles (list): A list of SMILES strings.
        tokenizer (Tokenizer): A Keras Tokenizer object.
        max_length (int): The maximum length of the sequences.

    Returns:
        numpy.ndarray: Padded sequences.
    """
    sequences = tokenizer.texts_to_sequences(smiles)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', value=0)
    return padded_sequences

def build_model(vocab_size, max_length):
    """
    Build the generative LSTM model.

    Args:
        vocab_size (int): The size of the vocabulary.
        max_length (int): The maximum length of the sequences.

    Returns:
        Model: A compiled Keras Model object.
    """
    inputs = Input(shape=(max_length,))
    x = Embedding(input_dim=vocab_size, output_dim=256)(inputs)
    x = LSTM(256, return_sequences=True)(x)
    x = LSTM(256, return_sequences=True)(x)
    outputs = TimeDistributed(Dense(vocab_size, activation='softmax'))(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    return model

def generate_smiles(model, tokenizer, max_length, num_smiles):
    """
    Generate new SMILES strings using the trained model.

    Args:
        model (Model): The trained Keras Model.
        tokenizer (Tokenizer): A Keras Tokenizer object.
        max_length (int): The maximum length of the sequences.
        num_smiles (int): The number of SMILES strings to generate.

    Returns:
        list: A list of generated SMILES strings.
    """
    smiles = []
    for _ in range(num_smiles):
        input_seq = np.zeros((1, max_length))
        generated_smiles = []
        for i in range(max_length):
            preds = model.predict(input_seq)[0, i]
            next_token = np.argmax(preds)
            if next_token == 0:
                break
            generated_smiles.append(next_token)
            input_seq[0, i] = next_token
        smiles.append(''.join(tokenizer.sequences_to_texts([generated_smiles])[0]))
    return smiles

def main():
    """
    Main function to load data, train the model, and generate new SMILES strings.
    """
    smiles = load_smiles('data/train-SMILES.txt')
    tokenizer = Tokenizer(char_level=True, oov_token='')
    tokenizer.fit_on_texts(smiles)
    max_length = max([len(s) for s in smiles])
    vocab_size = len(tokenizer.word_index) + 1
    
    x = preprocess_smiles(smiles, tokenizer, max_length)
    y = np.expand_dims(x, -1)
    
    model = build_model(vocab_size, max_length)
    plt.figure()
    plt.plot(model.fit(x, y, epochs=50, batch_size=32).history["loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("model/loss-epoch.png")
    plt.close()
    generated_smiles = generate_smiles(model, tokenizer, max_length, 1000)
        
    save_smiles('model/SMILES.txt', generated_smiles)

if __name__ == '__main__':
    main()
