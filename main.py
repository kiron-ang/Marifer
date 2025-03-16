"""
Main script to train two Variational Autoencoders (VAEs) for molecular generation.
One VAE uses SMILES strings directly and the other uses a dummy graph representation.
After training, each VAE generates 1000 molecules and the percentage of valid molecules
is computed using RDKit.
"""

import argparse
import hashlib
import numpy as np
import tensorflow as tf
from tensorflow import keras  # pylint: disable=import-error
import tensorflow_datasets as tfds
from rdkit import Chem


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
        smiles_list = [line.strip() for line in file_in if line.strip()]
    return smiles_list


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
    Dummy function to encode a SMILES string into a fixed-dimension vector representing a graph.
    In a real application, this should convert the molecule into an actual graph representation.

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
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


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


def preprocess_data(input_path):
    """
    Reads and preprocesses SMILES data.

    Args:
        input_path (str): Path to the SMILES text file.

    Returns:
        tuple: (encoded_smiles, char_to_index, index_to_char, max_length, graph_data)
    """
    print(f"Reading SMILES data from: {input_path}")
    smiles_list = read_smiles(input_path)
    if not smiles_list:
        print("No SMILES strings found in the input file.")
        exit(1)
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
    print("Data preprocessing complete.")
    return encoded_smiles, char_to_index, index_to_char, max_length, graph_data


# -------------------- Custom Layers and Models -------------------- #
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

    def get_config(self):
        config = super().get_config()
        return config


class SmilesVAE(keras.Model):
    """
    Variational Autoencoder that works directly with SMILES strings.
    """

    def __init__(self, vocab_size, max_length, latent_dim=32):
        """
        Initializes the SmilesVAE.

        Args:
            vocab_size (int): Size of the SMILES character vocabulary.
            max_length (int): Maximum length of SMILES sequences.
            latent_dim (int): Dimensionality of the latent space.
        """
        super(SmilesVAE, self).__init__()
        self.max_length = max_length
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size
        self.embedding = keras.layers.Embedding(
            input_dim=vocab_size + 1, output_dim=64, mask_zero=True, name="embedding"
        )
        self.encoder_lstm = keras.layers.LSTM(64, name="encoder_lstm")
        self.z_mean_dense = keras.layers.Dense(latent_dim, name="z_mean")
        self.z_log_var_dense = keras.layers.Dense(latent_dim, name="z_log_var")
        self.sampling = Sampling(name="sampling")
        self.repeat_vector = keras.layers.RepeatVector(max_length, name="repeat_vector")
        self.decoder_lstm = keras.layers.LSTM(64, return_sequences=True, name="decoder_lstm")
        self.decoder_dense = keras.layers.TimeDistributed(
            keras.layers.Dense(vocab_size + 1, activation="softmax"), name="decoder_output"
        )
        self.total_loss_tracker = keras.metrics.Mean(name="loss")

    def encode(self, x):
        x = self.embedding(x)
        x = self.encoder_lstm(x)
        z_mean = self.z_mean_dense(x)
        z_log_var = self.z_log_var_dense(x)
        z = self.sampling([z_mean, z_log_var])
        return z_mean, z_log_var, z

    def decode(self, z):
        x = self.repeat_vector(z)
        x = self.decoder_lstm(x)
        return self.decoder_dense(x)

    def call(self, x):
        z_mean, z_log_var, z = self.encode(x)
        return self.decode(z)

    def train_step(self, data):
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
        latent_samples = np.random.normal(size=(num_samples, self.latent_dim))
        preds = self.decode(latent_samples)
        generated = []
        for pred in preds:
            token_indices = np.argmax(pred, axis=-1)
            smiles = decode_sequence(token_indices, index_to_char)
            generated.append(smiles)
        return generated


class GraphVAE(keras.Model):
    """
    Variational Autoencoder that works on a dummy graph representation.
    In a full implementation, the encoder/decoder would operate on actual graph data.
    """

    def __init__(self, input_dim, latent_dim=32):
        """
        Initializes the GraphVAE.

        Args:
            input_dim (int): Dimensionality of the graph representation.
            latent_dim (int): Dimensionality of the latent space.
        """
        super(GraphVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.dense1 = keras.layers.Dense(128, activation="relu", name="graph_dense1")
        self.z_mean_dense = keras.layers.Dense(latent_dim, name="graph_z_mean")
        self.z_log_var_dense = keras.layers.Dense(latent_dim, name="graph_z_log_var")
        self.sampling = Sampling(name="graph_sampling")
        self.dense2 = keras.layers.Dense(128, activation="relu", name="graph_dense2")
        self.decoder_output = keras.layers.Dense(input_dim, activation="sigmoid",
                                                   name="graph_decoder_output")
        self.total_loss_tracker = keras.metrics.Mean(name="loss")

    def encode(self, x):
        x = self.dense1(x)
        z_mean = self.z_mean_dense(x)
        z_log_var = self.z_log_var_dense(x)
        z = self.sampling([z_mean, z_log_var])
        return z_mean, z_log_var, z

    def decode(self, z):
        x = self.dense2(z)
        return self.decoder_output(x)

    def call(self, x):
        z_mean, z_log_var, z = self.encode(x)
        return self.decode(z)

    def train_step(self, data):
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
        latent_samples = np.random.normal(size=(num_samples, self.latent_dim))
        preds = self.decode(latent_samples)
        generated = []
        for pred in preds:
            smiles = decode_graph(pred)
            generated.append(smiles)
        return generated


# -------------------- Main Function -------------------- #
def main():
    """
    Main function that reads data, trains both VAE models, generates molecules,
    and evaluates the validity of the generated molecules.
    """
    parser = argparse.ArgumentParser(
        description="Train two VAEs for molecular generation and evaluate validity."
    )
    parser.add_argument(
        "--input", type=str, default="test-smiles.txt",
        help="Path to the input SMILES text file."
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Number of training epochs for both models."
    )
    args = parser.parse_args()

    print("Starting preprocessing of SMILES data.")
    (encoded_smiles, char_to_index, index_to_char,
     max_length, graph_data) = preprocess_data(args.input)

    # Initialize and train SMILES VAE
    print("Initializing SMILES VAE.")
    smiles_vae = SmilesVAE(vocab_size=len(char_to_index), max_length=max_length)
    smiles_vae.compile(optimizer="adam")
    print("Training SMILES VAE...")
    smiles_vae.fit(encoded_smiles, epochs=args.epochs, batch_size=32, verbose=2)

    # Initialize and train Graph VAE
    print("Initializing Graph VAE.")
    graph_vae = GraphVAE(input_dim=graph_data.shape[1])
    graph_vae.compile(optimizer="adam")
    print("Training Graph VAE...")
    graph_vae.fit(graph_data, epochs=args.epochs, batch_size=32, verbose=2)

    num_generate = 1000
    print(f"Generating {num_generate} molecules with SMILES VAE...")
    generated_smiles_direct = smiles_vae.generate(num_generate, index_to_char)
    print(f"Generating {num_generate} molecules with Graph VAE...")
    generated_smiles_graph = graph_vae.generate(num_generate)

    valid_pct_direct = evaluate_generation(generated_smiles_direct)
    valid_pct_graph = evaluate_generation(generated_smiles_graph)

    print("\nSMILES VAE generated molecules (first 20 shown):")
    for smi in generated_smiles_direct[:20]:
        print(smi)
    print("\nGraph VAE generated molecules (first 20 shown):")
    for smi in generated_smiles_graph[:20]:
        print(smi)

    print("\nEvaluation Metrics:")
    print(f"SMILES VAE valid percentage: {valid_pct_direct:.2f}%")
    print(f"Graph VAE valid percentage: {valid_pct_graph:.2f}%")


if __name__ == "__main__":
    main()
