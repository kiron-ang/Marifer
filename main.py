"""
main.py: Train a generative model for molecules using SMILES strings.
Input: A text file ("data/train-smiles.txt") where each line is a SMILES string.
The script uses RDKit and TensorFlow to:
  1. Parse and preprocess the SMILES data.
  2. Train a character-level generative model.
  3. Generate 1000 molecules and assess their chemical validity.
Output:
  - Training loss and validity figures saved in the "output" folder.
  - A text file ("output/output-molecules.txt") containing the 1000 generated SMILES strings.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from rdkit import Chem

# Set random seeds for reproducibility.
np.random.seed(42)
tf.random.set_seed(42)


def load_smiles(filepath):
    """
    Load SMILES strings from a text file.

    Args:
        filepath (str): Path to the input file.

    Returns:
        list: List of SMILES strings.
    """
    with open(filepath, 'r', encoding="utf-8") as file:
        smiles_list = [line.strip() for line in file if line.strip()]
    return smiles_list


def preprocess_smiles(smiles_list, start_token='!', end_token='$'):
    """
    Preprocess SMILES by adding start and end tokens.

    Args:
        smiles_list (list): List of SMILES strings.
        start_token (str): Token indicating start of sequence.
        end_token (str): Token indicating end of sequence.

    Returns:
        list: List of processed SMILES strings.
    """
    return [f"{start_token}{smiles}{end_token}" for smiles in smiles_list]


def create_vocabulary(smiles_list):
    """
    Create a vocabulary mapping from characters to indices.

    Args:
        smiles_list (list): List of SMILES strings.

    Returns:
        tuple: (char_to_idx, idx_to_char)
    """
    vocab = sorted(set(''.join(smiles_list)))
    # Reserve 0 for padding.
    char_to_idx = {char: idx + 1 for idx, char in enumerate(vocab)}
    idx_to_char = {idx + 1: char for idx, char in enumerate(vocab)}
    return char_to_idx, idx_to_char


def tokenize_smiles(smiles_list, char_to_idx):
    """
    Convert SMILES strings to lists of integer tokens.

    Args:
        smiles_list (list): List of SMILES strings.
        char_to_idx (dict): Mapping from character to index.

    Returns:
        list: List of integer sequences.
    """
    sequences = []
    for smiles in smiles_list:
        sequence = [char_to_idx[char] for char in smiles]
        sequences.append(sequence)
    return sequences


def create_training_data(sequences, max_length):
    """
    Create input and target sequences for training.
    For each sequence, the input is all characters except the last and the target is
    the same sequence shifted by one.

    Args:
        sequences (list): List of integer sequences.
        max_length (int): Maximum sequence length for padding.

    Returns:
        tuple: (input_data, target_data)
    """
    input_data, target_data = [], []
    for seq in sequences:
        input_seq = seq[:-1]
        target_seq = seq[1:]
        input_data.append(input_seq)
        target_data.append(target_seq)

    # Pad sequences to a fixed length.
    input_data = pad_sequences(input_data, maxlen=max_length - 1, padding='post')
    target_data = pad_sequences(target_data, maxlen=max_length - 1, padding='post')
    return input_data, target_data


def build_model(vocab_size, embedding_dim=64, lstm_units=128, input_length=None):
    """
    Build and compile the TensorFlow model.

    Args:
        vocab_size (int): Size of the vocabulary.
        embedding_dim (int): Dimension of the embedding layer.
        lstm_units (int): Number of LSTM units.
        input_length (int): Input sequence length.

    Returns:
        tf.keras.Model: Compiled model.
    """
    model = Sequential([
        Embedding(input_dim=vocab_size + 1,
                  output_dim=embedding_dim,
                  mask_zero=True,
                  input_length=input_length),
        LSTM(lstm_units, return_sequences=True),
        Dense(vocab_size + 1, activation='softmax')
    ])
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy')
    return model


def sample_from_model(model, char_to_idx, idx_to_char, max_length, temperature=1.0):
    """
    Generate a SMILES string from the model using a sampling approach.

    Args:
        model (tf.keras.Model): Trained model.
        char_to_idx (dict): Mapping from character to index.
        idx_to_char (dict): Mapping from index to character.
        max_length (int): Maximum length of generated sequence.
        temperature (float): Sampling temperature.

    Returns:
        str: Generated SMILES string (without start and end tokens).
    """
    start_token = '!'
    end_token = '$'
    # Initialize sequence with the start token.
    current_sequence = [char_to_idx[start_token]]
    for _ in range(max_length):
        padded_seq = pad_sequences([current_sequence], maxlen=max_length, padding='post')
        predictions = model.predict(padded_seq, verbose=0)
        # Focus on the prediction for the last character in the current sequence.
        next_token_probs = predictions[0, len(current_sequence) - 1]
        # Adjust probabilities using temperature.
        next_token_probs = np.log(next_token_probs + 1e-8) / temperature
        exp_preds = np.exp(next_token_probs)
        next_token_probs = exp_preds / np.sum(exp_preds)
        # Sample the next token.
        next_index = np.random.choice(len(next_token_probs), p=next_token_probs)
        current_sequence.append(next_index)
        # If the end token is generated, stop generation.
        if idx_to_char.get(next_index, '') == end_token:
            break
    # Convert token indices back to characters.
    generated = ''.join([idx_to_char.get(idx, '') for idx in current_sequence])
    # Remove the start and end tokens.
    if generated.startswith(start_token):
        generated = generated[len(start_token):]
    if generated.endswith(end_token):
        generated = generated[:-len(end_token)]
    return generated


def validate_smiles(smiles):
    """
    Check if a SMILES string represents a chemically valid molecule using RDKit.

    Args:
        smiles (str): SMILES string.

    Returns:
        bool: True if valid, False otherwise.
    """
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


def main():
    """
    Main function to load data, train the model, generate molecules,
    evaluate their validity, and output metrics and figures.
    """
    # Load and preprocess SMILES data.
    char_to_idx, idx_to_char = create_vocabulary(preprocess_smiles(
        load_smiles('data/train-smiles.txt')))
    sequences = tokenize_smiles(processed_smiles, char_to_idx)
    max_length = max(len(seq) for seq in sequences)
    x_train, y_train = create_training_data(sequences, max_length)
    # Expand target dimensions for sparse categorical crossentropy.
    y_train = np.expand_dims(y_train, -1)

    # Build the model.
    model = build_model(len(char_to_idx), input_length=max_length - 1)

    # Train the model.
    history = model.fit(x_train, y_train, epochs=20, batch_size=64, verbose=1)

    # Plot and save training loss.
    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.savefig('output/training_loss.png')
    plt.close()

    # Generate 1000 molecules.
    num_generated = 1000
    generated_smiles = []
    for _ in range(num_generated):
        gen_smiles = sample_from_model(model, char_to_idx, idx_to_char, max_length)
        generated_smiles.append(gen_smiles)

    # Validate generated molecules.
    valid_count = sum(validate_smiles(s) for s in generated_smiles)

    # Plot and save molecule validity results.
    plt.figure()
    plt.bar(['Valid', 'Invalid'], [valid_count, num_generated - valid_count], 
        color=['green', 'red'])
    plt.title('Generated Molecule Validity')
    plt.ylabel('Count')
    plt.savefig('output/validity.png')
    plt.close()

    # Write generated SMILES to the output file.
    with open('data/output-molecules.txt', 'w', encoding="utf-8") as f_out:
        for s in generated_smiles:
            f_out.write(s + '\n')

    # Print evaluation metrics.
    print(f"Generated {num_generated} molecules.")
    print(f"Valid molecules: {valid_count} ({(valid_count / num_generated) * 100:.2f}%)")
    print(f"Invalid molecules: {num_generated - valid_count} ({
        (num_generated - valid_count / num_generated) * 100:.2f}%)")


if __name__ == '__main__':
    main()
