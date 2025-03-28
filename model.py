"""
This script generates new molecules using a Long Short-Term Memory (LSTM) Model in TensorFlow
and assesses the model's performance using specified chemistry-related criteria.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from rdkit import Chem
from rdkit.Chem import QED, Crippen


def load_smiles(file_path):
    """
    Load the SMILES strings from a file.

    Args:
        file_path (str): Path to the file containing SMILES strings.

    Returns:
        list: List of SMILES strings.
    """
    with open(file_path, 'r') as file:
        smiles = file.readlines()
    return [s.strip() for s in smiles]


def preprocess_smiles(smiles, tokenizer=None, max_length=100):
    """
    Preprocess the SMILES strings.

    Args:
        smiles (list): List of SMILES strings.
        tokenizer (Tokenizer, optional): Tokenizer for the SMILES strings. Defaults to None.
        max_length (int, optional): Maximum length of the SMILES strings. Defaults to 100.

    Returns:
        tuple: Padded sequences and tokenizer.
    """
    if tokenizer is None:
        tokenizer = Tokenizer(char_level=True)
        tokenizer.fit_on_texts(smiles)
    sequences = tokenizer.texts_to_sequences(smiles)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    return padded_sequences, tokenizer


def create_lstm_model(vocab_size, embedding_dim=128, lstm_units=256, max_length=100):
    """
    Define the LSTM model.

    Args:
        vocab_size (int): Vocabulary size.
        embedding_dim (int, optional): Dimension of the embedding layer. Defaults to 128.
        lstm_units (int, optional): Number of LSTM units. Defaults to 256.
        max_length (int, optional): Maximum length of the input sequences. Defaults to 100.

    Returns:
        Sequential: Compiled LSTM model.
    """
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
        LSTM(lstm_units, return_sequences=True),
        LSTM(lstm_units),
        Dense(vocab_size, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def generate_molecules(model, tokenizer, num_molecules=100, max_length=100):
    """
    Generate new molecules using the trained model.

    Args:
        model (Sequential): Trained LSTM model.
        tokenizer (Tokenizer): Tokenizer for the SMILES strings.
        num_molecules (int, optional): Number of molecules to generate. Defaults to 100.
        max_length (int, optional): Maximum length of the generated SMILES strings. Defaults to 100.

    Returns:
        list: List of generated SMILES strings.
    """
    generated_smiles = []
    for _ in range(num_molecules):
        seed = np.random.randint(1, tokenizer.num_words, size=(1, max_length))
        pred = model.predict(seed)
        pred_smiles = tokenizer.sequences_to_texts([np.argmax(pred, axis=-1).tolist()])
        generated_smiles.append(pred_smiles[0])
    return generated_smiles


def assess_performance(generated_smiles, known_smiles):
    """
    Assess the performance of the model.

    Args:
        generated_smiles (list): List of generated SMILES strings.
        known_smiles (list): List of known SMILES strings.

    Returns:
        tuple: Validity, uniqueness, and novelty of the generated molecules.
    """
    valid_smiles = [s for s in generated_smiles if Chem.MolFromSmiles(s) is not None]
    unique_smiles = set(valid_smiles)
    novel_smiles = unique_smiles - set(known_smiles)

    valid = len(valid_smiles) / len(generated_smiles)
    unique = len(unique_smiles) / len(valid_smiles)
    novel = len(novel_smiles) / len(unique_smiles)

    return valid, unique, novel


def calculate_properties(smiles):
    """
    Calculate specific desired properties for a given SMILES string.

    Args:
        smiles (str): SMILES string.

    Returns:
        tuple: logP and QED of the molecule.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    logp = Crippen.MolLogP(mol)
    qed = QED.qed(mol)
    return logp, qed


def main():
    """
    Main function to generate molecules and assess performance.
    """
    # Define paths
    input_file_path = 'data/train-SMILES.txt'
    output_file_path = 'model/output-SMILES.txt'

    # Load and preprocess the SMILES strings
    smiles = load_smiles(input_file_path)
    sequences, tokenizer = preprocess_smiles(smiles)

    # Define and train the LSTM model
    model = create_lstm_model(vocab_size=len(tokenizer.word_index) + 1)
    model.fit(sequences, np.array(sequences), epochs=10, batch_size=64)

    # Generate new molecules
    generated_smiles = generate_molecules(model, tokenizer)

    # Save generated molecules to file
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, 'w') as file:
        for s in generated_smiles:
            file.write(s + '\n')

    # Assess performance
    valid, unique, novel = assess_performance(generated_smiles, smiles)
    print(f'Validity: {valid:.2f}, Uniqueness: {unique:.2f}, Novelty: {novel:.2f}')

    # Calculate properties for each generated molecule
    properties = [calculate_properties(s) for s in generated_smiles if Chem.MolFromSmiles(s) is not None]
    if properties:
        logps, qeds = zip(*[p for p in properties if p is not None])
        print(f'Average logP: {np.mean(logps):.2f}, Average QED: {np.mean(qeds):.2f}')


if __name__ == '__main__':
    main()
