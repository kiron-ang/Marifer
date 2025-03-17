"""
Script for training and evaluating Variational Autoencoders (VAEs) for molecular generation.
"""

import os
import sys
import argparse
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

# -------------------- Configuration -------------------- #
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, "output.txt")),
        logging.StreamHandler()
    ]
)

# -------------------- Utility Functions -------------------- #
def read_smiles(file_path):
    """Reads SMILES strings from a text file."""
    with open(file_path, "r", encoding="utf-8") as file_in:
        return [line.strip() for line in file_in if line.strip()]

def create_smiles_tokenizer(smiles_list):
    """Creates mappings for tokenizing SMILES strings."""
    chars = sorted(set("".join(smiles_list)))
    char_to_index = {char: idx + 1 for idx, char in enumerate(chars)}
    index_to_char = {idx + 1: char for idx, char in enumerate(chars)}
    return char_to_index, index_to_char

def encode_smiles(smiles, char_to_index, max_length):
    """Encodes a SMILES string into a fixed-length sequence of integers."""
    seq = [char_to_index.get(char, 0) for char in smiles]
    return seq + [0] * (max_length - len(seq)) if len(seq) < max_length else seq[:max_length]

def decode_sequence(seq, index_to_char):
    """Decodes a sequence of integers back into a SMILES string."""
    return "".join(index_to_char.get(idx, "") for idx in seq if idx != 0)

def encode_graph(smiles, fixed_dim):
    """Encodes a SMILES string into a fixed-length Morgan fingerprint vector."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(fixed_dim, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=fixed_dim)
    arr = np.zeros((fixed_dim,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def decode_graph(vector, fingerprint_database):
    """Decodes a graph representation back into a SMILES string using nearest-neighbor search."""
    binary = (np.array(vector) > 0.5).astype(int)
    bit_string = "".join(str(b) for b in binary)
    fp_pred = DataStructs.CreateFromBitString(bit_string)

    best_smiles = max(
        fingerprint_database, key=lambda x: DataStructs.TanimotoSimilarity(fp_pred, x[0]), 
        default=(None, "")
    )[1]

    return best_smiles

def is_valid_smiles(smiles):
    """Checks whether a SMILES string represents a valid molecule."""
    return Chem.MolFromSmiles(smiles) is not None

def evaluate_generation(generated_smiles):
    """Evaluates the percentage of valid molecules in generated SMILES strings."""
    valid_count = sum(1 for smi in generated_smiles if is_valid_smiles(smi))
    return 100.0 * valid_count / len(generated_smiles) if generated_smiles else 0.0

def process_data(input_path):
    """Reads and preprocesses SMILES data."""
    logging.info("Reading SMILES data from: %s", input_path)
    smiles_list = read_smiles(input_path)
    if not smiles_list:
        logging.error("No SMILES strings found in the input file.")
        sys.exit(1)

    char_to_index, index_to_char = create_smiles_tokenizer(smiles_list)
    max_length = max(len(smi) for smi in smiles_list)
    encoded_smiles = np.array([encode_smiles(smi, char_to_index, 
                                max_length) for smi in smiles_list], dtype=np.int32)

    graph_dim = 100
    graph_data = np.array([encode_graph(smi, graph_dim) for smi in smiles_list], dtype=np.float32)

    fingerprint_database = [
        (AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), radius=2, 
                                                nBits=graph_dim), smi)
        for smi in smiles_list if Chem.MolFromSmiles(smi) is not None
    ]

    logging.info("Data preprocessing complete.")
    return encoded_smiles, char_to_index, index_to_char, max_length,graph_data,fingerprint_database

# -------------------- Custom Model Layers -------------------- #
class Sampling(keras.layers.Layer):
    """Reparameterization trick for variational autoencoder sampling."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# -------------------- Model Definitions -------------------- #
class StringVariations(keras.Model):
    """Variational Autoencoder for SMILES string encoding."""

    def __init__(self, vocab_size, max_length, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim

        encoder_inputs = keras.Input(shape=(max_length,))
        x = keras.layers.Embedding(input_dim=vocab_size + 1, output_dim=64, mask_zero=True)(encoder_inputs)
        x = keras.layers.LSTM(64)(x)
        z_mean = keras.layers.Dense(latent_dim)(x)
        z_log_var = keras.layers.Dense(latent_dim)(x)
        z = Sampling()([z_mean, z_log_var])
        self.encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z])

        latent_inputs = keras.Input(shape=(latent_dim,))
        x = keras.layers.RepeatVector(max_length)(latent_inputs)
        x = keras.layers.LSTM(64, return_sequences=True)(x)
        decoder_outputs = keras.layers.TimeDistributed(keras.layers.Dense(vocab_size + 1, 
                                                        activation="softmax"))(x)
        self.decoder = keras.Model(latent_inputs, decoder_outputs)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def call(self, x):
        _, _, z = self.encode(x)
        return self.decode(z)

class GraphVariations(keras.Model):
    """Variational Autoencoder for graph-based encoding."""

    def __init__(self, input_dim, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim

        encoder_inputs = keras.Input(shape=(input_dim,))
        x = keras.layers.Dense(128, activation="relu")(encoder_inputs)
        z_mean = keras.layers.Dense(latent_dim)(x)
        z_log_var = keras.layers.Dense(latent_dim)(x)
        z = Sampling()([z_mean, z_log_var])
        self.encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z])

        latent_inputs = keras.Input(shape=(latent_dim,))
        x = keras.layers.Dense(128, activation="relu")(latent_inputs)
        decoder_outputs = keras.layers.Dense(input_dim, activation="sigmoid")(x)
        self.decoder = keras.Model(latent_inputs, decoder_outputs)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def call(self, x):
        _, _, z = self.encode(x)
        return self.decode(z)

# -------------------- Main Script -------------------- #
def main():
    """Parses arguments, trains models, and evaluates outputs."""
    parser = argparse.ArgumentParser(description="Train two VAEs for molecule generation.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input SMILES.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")

    args = parser.parse_args()

    data = process_data(args.input)
    encoded_smiles, char_to_index, index_to_char, max_length,graph_data,fingerprint_database = data

    string_model = StringVariations(len(char_to_index), max_length)
    graph_model = GraphVariations(graph_data.shape[1])

    string_model.compile(optimizer="adam")
    graph_model.compile(optimizer="adam")

    string_model.fit(encoded_smiles, epochs=args.epochs, batch_size=32)
    graph_model.fit(graph_data, epochs=args.epochs, batch_size=32)

if __name__ == "__main__":
    main()
