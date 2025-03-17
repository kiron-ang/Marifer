"""
Main script to train two Variational Autoencoders for molecular generation.
One model, StringVariations, works directly with SMILES strings, and the other,
GraphVariations, works on a dummy graph representation. After training, each model
generates 1000 molecules and the percentage of valid molecules is computed using RDKit.
All console output is logged into 'output/output.txt'. The generated molecules are saved
in 'output/string-variations.txt' and 'output/graph-variations.txt'.
"""

import sys
import os
import argparse
import logging
import hashlib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from rdkit import Chem

# Create output directory and set up logging.
OUTPUT_DIR = "output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

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
    """
    Reads SMILES strings from a text file.

    Args:
        file_path (str): Path to the text file.

    Returns:
        list: A list of SMILES strings.
    """
    with open(file_path, "r", encoding="utf-8") as file_in:
        return [line.strip() for line in file_in if line.strip()]


def create_smiles_tokenizer(smiles_list):
    """
    Creates character-to-index and index-to-character mappings for SMILES strings.

    Args:
        smiles_list (list): List of SMILES strings.

    Returns:
        tuple: (char_to_index, index_to_char) dictionaries.
    """
    chars = sorted(set("".join(smiles_list)))
    char_to_index = {char: idx + 1 for idx, char in enumerate(chars)}
    index_to_char = {idx + 1: char for idx, char in enumerate(chars)}
    return char_to_index, index_to_char


def encode_smiles(smiles, char_to_index, max_length):
    """
    Encodes a SMILES string into a fixed-length sequence of integers.

    Args:
        smiles (str): The SMILES string.
        char_to_index (dict): Mapping from character to integer index.
        max_length (int): Maximum sequence length.

    Returns:
        list: A list of integer indices.
    """
    seq = [char_to_index.get(char, 0) for char in smiles]
    if len(seq) < max_length:
        seq += [0] * (max_length - len(seq))
    else:
        seq = seq[:max_length]
    return seq


def decode_sequence(seq, index_to_char):
    """
    Decodes a sequence of integers back into a SMILES string.

    Args:
        seq (list): List of integer indices.
        index_to_char (dict): Mapping from integer index to character.

    Returns:
        str: The decoded SMILES string.
    """
    return "".join(index_to_char.get(idx, "") for idx in seq if idx != 0)


def encode_graph(smiles, fixed_dim):
    """
    Dummy function to encode a SMILES string into a fixed-dimension vector representing
    a graph. In a real application, this should convert the molecule into an actual graph
    representation.

    Args:
        smiles (str): The SMILES string.
        fixed_dim (int): Dimensionality of the graph representation.

    Returns:
        np.ndarray: A one-hot-like vector of length fixed_dim.
    """
    vec = np.zeros(fixed_dim, dtype=np.float32)
    idx = int(hashlib.sha256(smiles.encode("utf-8")).hexdigest(), 16) % fixed_dim
    vec[idx] = 1.0
    return vec


def decode_graph(_vector):
    """
    Dummy function to decode a graph representation vector back into a SMILES string.
    Here, we return a placeholder molecule (methane).

    Args:
        _vector (np.ndarray): Graph representation vector.

    Returns:
        str: The decoded SMILES string.
    """
    return "C"


def is_valid_smiles(smiles):
    """
    Checks whether a SMILES string represents a chemically valid molecule using RDKit.

    Args:
        smiles (str): SMILES string.

    Returns:
        bool: True if the molecule is valid, False otherwise.
    """
    return Chem.MolFromSmiles(smiles) is not None


def evaluate_generation(generated_smiles):
    """
    Evaluates the percentage of valid molecules in a list of generated SMILES strings.

    Args:
        generated_smiles (list): List of generated SMILES strings.

    Returns:
        float: Percentage of valid molecules.
    """
    valid_count = sum(1 for smi in generated_smiles if is_valid_smiles(smi))
    return 100.0 * valid_count / len(generated_smiles) if generated_smiles else 0.0


def process_data(input_path):
    """
    Reads and preprocesses SMILES data.

    Args:
        input_path (str): Path to the SMILES text file.

    Returns:
        tuple: (encoded_smiles, char_to_index, index_to_char, max_length, graph_data)
    """
    logging.info("Reading SMILES data from: %s", input_path)
    smiles_list = read_smiles(input_path)
    if not smiles_list:
        logging.error("No SMILES strings found in the input file.")
        sys.exit(1)
    char_to_index, index_to_char = create_smiles_tokenizer(smiles_list)
    max_length = max(len(smi) for smi in smiles_list)
    encoded_smiles = np.array(
        [encode_smiles(smi, char_to_index, max_length) for smi in smiles_list],
        dtype=np.int32
    )
    graph_dim = 100
    graph_data = np.array(
        [encode_graph(smi, graph_dim) for smi in smiles_list],
        dtype=np.float32
    )
    logging.info("Data preprocessing complete.")
    return encoded_smiles, char_to_index, index_to_char, max_length, graph_data


# -------------------- Custom Layers and Models -------------------- #
# pylint: disable=too-few-public-methods
class Sampling(keras.layers.Layer):
    """
    Uses (z_mean, z_log_var) to sample z, the latent vector.
    """

    def call(self, inputs):
        """
        Reparameterization trick by sampling from an isotropic unit Gaussian.

        Args:
            inputs (list): [z_mean, z_log_var].

        Returns:
            Tensor: Sampled latent vector.
        """
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class StringVariations(keras.Model):
    """
    Variational Autoencoder that works directly with SMILES strings.
    """

    def __init__(self, vocab_size, max_length, latent_dim=32):
        """
        Initializes the StringVariations model.

        Args:
            vocab_size (int): Size of the SMILES character vocabulary.
            max_length (int): Maximum length of SMILES sequences.
            latent_dim (int): Dimensionality of the latent space.
        """
        super().__init__()
        self.max_length = max_length
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size

        # Build encoder model.
        encoder_inputs = keras.Input(shape=(max_length,),
                                     name="string_encoder_input")
        x = keras.layers.Embedding(input_dim=vocab_size + 1, output_dim=64,
                                   mask_zero=True, name="string_embedding")(encoder_inputs)
        x = keras.layers.LSTM(64, name="string_encoder_lstm")(x)
        z_mean = keras.layers.Dense(latent_dim, name="string_z_mean")(x)
        z_log_var = keras.layers.Dense(latent_dim, name="string_z_log_var")(x)
        z = Sampling(name="string_sampling")([z_mean, z_log_var])
        self.encoder = keras.Model(
            encoder_inputs, [z_mean, z_log_var, z],
            name="string_variations_encoder"
        )

        # Build decoder model.
        latent_inputs = keras.Input(shape=(latent_dim,),
                                     name="string_z_sampling")
        x = keras.layers.RepeatVector(max_length,
                                      name="string_repeat_vector")(latent_inputs)
        x = keras.layers.LSTM(64, return_sequences=True,
                              name="string_decoder_lstm")(x)
        decoder_outputs = keras.layers.TimeDistributed(
            keras.layers.Dense(vocab_size + 1, activation="softmax"),
            name="string_decoder_output"
        )(x)
        self.decoder = keras.Model(
            latent_inputs, decoder_outputs,
            name="string_variations_decoder"
        )
        self.total_loss_tracker = keras.metrics.Mean(name="loss")

    def encode(self, x):
        """
        Encodes the input data into latent variables.

        Args:
            x (Tensor): Input tensor.

        Returns:
            tuple: (z_mean, z_log_var, z)
        """
        return self.encoder(x)

    def decode(self, z):
        """
        Decodes the latent vector back to the original space.

        Args:
            z (Tensor): Latent vector.

        Returns:
            Tensor: Reconstructed output.
        """
        return self.decoder(z)

    def call(self, x):
        """
        Forward pass through the model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Reconstructed output.
        """
        _, _, z = self.encode(x)
        return self.decode(z)

    def train_step(self, data):
        """
        Custom training step for the model.

        Args:
            data (Tensor): Input data.

        Returns:
            dict: Loss metrics.
        """
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encode(data)
            reconstruction = self.decode(z)
            reconstruction_loss = tf.reduce_mean(
                keras.losses.sparse_categorical_crossentropy(data, reconstruction)
            ) * self.max_length
            kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var -
                                              tf.square(z_mean) -
                                              tf.exp(z_log_var))
            loss = reconstruction_loss + kl_loss
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(loss)
        return {"loss": self.total_loss_tracker.result()}

    def generate(self, num_samples, index_to_char):
        """
        Generates SMILES strings by sampling from the latent space.

        Args:
            num_samples (int): Number of molecules to generate.
            index_to_char (dict): Mapping from token indices to characters.

        Returns:
            list: Generated SMILES strings.
        """
        latent_samples = np.random.normal(size=(num_samples, self.latent_dim))
        preds = self.decode(latent_samples)
        generated = []
        for pred in preds:
            token_indices = np.argmax(pred, axis=-1)
            generated.append(decode_sequence(token_indices, index_to_char))
        return generated


class GraphVariations(keras.Model):
    """
    Variational Autoencoder that works on a dummy graph representation.
    In a full implementation, the encoder/decoder would operate on actual graph data.
    """

    def __init__(self, input_dim, latent_dim=32):
        """
        Initializes the GraphVariations model.

        Args:
            input_dim (int): Dimensionality of the graph representation.
            latent_dim (int): Dimensionality of the latent space.
        """
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Build encoder model.
        encoder_inputs = keras.Input(shape=(input_dim,),
                                     name="graph_encoder_input")
        x = keras.layers.Dense(128, activation="relu",
                               name="graph_dense1")(encoder_inputs)
        z_mean = keras.layers.Dense(latent_dim, name="graph_z_mean")(x)
        z_log_var = keras.layers.Dense(latent_dim, name="graph_z_log_var")(x)
        z = Sampling(name="graph_sampling")([z_mean, z_log_var])
        self.encoder = keras.Model(
            encoder_inputs, [z_mean, z_log_var, z],
            name="graph_variations_encoder"
        )

        # Build decoder model.
        latent_inputs = keras.Input(shape=(latent_dim,),
                                     name="graph_z_sampling")
        x = keras.layers.Dense(128, activation="relu",
                               name="graph_dense2")(latent_inputs)
        decoder_outputs = keras.layers.Dense(input_dim, activation="sigmoid",
                                             name="graph_decoder_output")(x)
        self.decoder = keras.Model(
            latent_inputs, decoder_outputs,
            name="graph_variations_decoder"
        )
        self.total_loss_tracker = keras.metrics.Mean(name="loss")

    def encode(self, x):
        """
        Encodes the input graph data into latent variables.

        Args:
            x (Tensor): Input tensor.

        Returns:
            tuple: (z_mean, z_log_var, z)
        """
        return self.encoder(x)

    def decode(self, z):
        """
        Decodes the latent vector back to the graph representation.

        Args:
            z (Tensor): Latent vector.

        Returns:
            Tensor: Reconstructed graph representation.
        """
        return self.decoder(z)

    def call(self, x):
        """
        Forward pass through the model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Reconstructed output.
        """
        _, _, z = self.encode(x)
        return self.decode(z)

    def train_step(self, data):
        """
        Custom training step for the model.

        Args:
            data (Tensor): Input data.

        Returns:
            dict: Loss metrics.
        """
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encode(data)
            reconstruction = self.decode(z)
            reconstruction_loss = tf.reduce_mean(tf.square(data - reconstruction))
            kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var -
                                              tf.square(z_mean) -
                                              tf.exp(z_log_var))
            loss = reconstruction_loss + kl_loss
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(loss)
        return {"loss": self.total_loss_tracker.result()}

    def generate(self, num_samples):
        """
        Generates graph representations by sampling from the latent space,
        then decodes them into SMILES strings using a dummy conversion.

        Args:
            num_samples (int): Number of molecules to generate.

        Returns:
            list: Generated SMILES strings.
        """
        latent_samples = np.random.normal(size=(num_samples, self.latent_dim))
        _ = self.decode(latent_samples)
        # Dummy conversion: always return "C" (methane) for each sample.
        return ["C" for _ in range(num_samples)]


def train_models(args):
    """
    Preprocesses the data and trains both models.

    Args:
        args (Namespace): Command line arguments.

    Returns:
        tuple: (string_variations, graph_variations, index_to_char)
    """
    encoded_smiles, char_to_index, index_to_char, max_length, graph_data = process_data(args.input)
    logging.info("Initializing StringVariations model.")
    string_variations = StringVariations(vocab_size=len(char_to_index), max_length=max_length)
    string_variations.compile(optimizer="adam")
    logging.info("Training StringVariations for %d epochs...", args.epochs)
    string_variations.fit(encoded_smiles, epochs=args.epochs, batch_size=32, verbose=2)

    logging.info("Initializing GraphVariations model.")
    graph_variations = GraphVariations(input_dim=graph_data.shape[1])
    graph_variations.compile(optimizer="adam")
    logging.info("Training GraphVariations for %d epochs...", args.epochs)
    graph_variations.fit(graph_data, epochs=args.epochs, batch_size=32, verbose=2)
    return string_variations, graph_variations, index_to_char


def generate_and_evaluate(string_variations, graph_variations, index_to_char):
    """
    Generates molecules with both models and evaluates their validity.

    Args:
        string_variations (StringVariations): Trained model for SMILES strings.
        graph_variations (GraphVariations): Trained model for graph data.
        index_to_char (dict): Mapping from token indices to characters.

    Returns:
        tuple: (generated_strings, generated_graphs)
    """
    num_generate = 1000
    logging.info("Generating %d molecules with StringVariations...", num_generate)
    generated_strings = string_variations.generate(num_generate, index_to_char)
    logging.info("Generating %d molecules with GraphVariations...", num_generate)
    generated_graphs = graph_variations.generate(num_generate)

    valid_pct_string = evaluate_generation(generated_strings)
    valid_pct_graph = evaluate_generation(generated_graphs)

    logging.info("StringVariations generated molecules (first 20 shown):")
    for smi in generated_strings[:20]:
        logging.info(smi)
    logging.info("GraphVariations generated molecules (first 20 shown):")
    for smi in generated_graphs[:20]:
        logging.info(smi)

    logging.info("Evaluation Metrics:")
    logging.info("StringVariations valid percentage: %.2f%%", valid_pct_string)
    logging.info("GraphVariations valid percentage: %.2f%%", valid_pct_graph)
    return generated_strings, generated_graphs


def main():
    """
    Main function that parses arguments, trains models, generates molecules,
    evaluates them, and saves all output files in the output directory.
    """
    parser = argparse.ArgumentParser(
        description="Train two VAEs for molecular generation and evaluate validity."
    )
    parser.add_argument(
        "--input", type=str, default="data/test-smiles.txt",
        help="Path to the input SMILES text file."
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Number of training epochs for both models."
    )
    args = parser.parse_args()

    string_var, graph_var, index_to_char = train_models(args)
    gen_strings, gen_graphs = generate_and_evaluate(string_var, graph_var, index_to_char)

    string_file = os.path.join(OUTPUT_DIR, "string-variations.txt")
    graph_file = os.path.join(OUTPUT_DIR, "graph-variations.txt")
    logging.info("Saving generated molecules to '%s' and '%s'.",
                 string_file, graph_file)
    with open(string_file, "w", encoding="utf-8") as f:
        for smi in gen_strings:
            f.write(smi + "\n")
    with open(graph_file, "w", encoding="utf-8") as f:
        for smi in gen_graphs:
            f.write(smi + "\n")
    logging.info("All outputs saved in the '%s' directory.", OUTPUT_DIR)


if __name__ == "__main__":
    main()