"""
Main script to train two Variational Autoencoders (VAEs) for molecular generation.
One VAE uses SMILES strings directly and the other uses a dummy graph representation.
After training, each VAE generates 100 molecules and the percentage of valid molecules
is computed using RDKit.
"""

import argparse
import hashlib
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
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
    # Reserve 0 for padding.
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
    # Use a hash of the SMILES to pick an index (for demonstration purposes).
    idx = int(hashlib.sha256(smiles.encode("utf-8")).hexdigest(), 16) % fixed_dim
    vec[idx] = 1.0
    return vec


def decode_graph(vector):
    """
    Dummy function to decode a graph representation vector back into a SMILES string.
    Here, we return a placeholder molecule (methane).

    Args:
        vector (np.ndarray): Graph representation vector.

    Returns:
        str: A SMILES string.
    """
    # Placeholder: always return methane
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


# -------------------- Sampling Layer -------------------- #
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


# -------------------- SMILES VAE -------------------- #
class SmilesVAE:
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
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.latent_dim = latent_dim
        self._build_model()

    def _build_model(self):
        """
        Builds the encoder, decoder, and VAE models.
        """
        # Encoder
        encoder_inputs = keras.Input(shape=(self.max_length,), name="encoder_input")
        x = keras.layers.Embedding(
            input_dim=self.vocab_size + 1, output_dim=64, mask_zero=True,
            name="embedding")(encoder_inputs)
        x = keras.layers.LSTM(64, name="encoder_lstm")(x)
        z_mean = keras.layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = keras.layers.Dense(self.latent_dim, name="z_log_var")(x)
        z = Sampling(name="sampling")([z_mean, z_log_var])
        self.encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z],
                                    name="smiles_encoder")

        # Decoder
        latent_inputs = keras.Input(shape=(self.latent_dim,), name="z_sampling")
        x = keras.layers.RepeatVector(self.max_length, name="repeat_vector")(latent_inputs)
        x = keras.layers.LSTM(64, return_sequences=True, name="decoder_lstm")(x)
        decoder_outputs = keras.layers.TimeDistributed(
            keras.layers.Dense(self.vocab_size + 1, activation="softmax"),
            name="decoder_output")(x)
        self.decoder = keras.Model(latent_inputs, decoder_outputs,
                                   name="smiles_decoder")

        # VAE Model
        outputs = self.decoder(z)
        self.vae = keras.Model(encoder_inputs, outputs, name="smiles_vae")

        # Loss calculation
        reconstruction_loss = keras.losses.sparse_categorical_crossentropy(
            encoder_inputs, outputs)
        reconstruction_loss = tf.reduce_mean(reconstruction_loss) * self.max_length
        kl_loss = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        self.vae.add_loss(reconstruction_loss + kl_loss)
        self.vae.compile(optimizer="adam")

    def train(self, x_train, epochs=1, batch_size=32):
        """
        Trains the SMILES VAE.

        Args:
            x_train (np.ndarray): Training data.
            epochs (int): Number of epochs.
            batch_size (int): Batch size.
        """
        self.vae.fit(x_train, x_train, epochs=epochs, batch_size=batch_size)

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
        preds = self.decoder.predict(latent_samples)
        generated = []
        for pred in preds:
            token_indices = np.argmax(pred, axis=-1)
            smiles = decode_sequence(token_indices, index_to_char)
            generated.append(smiles)
        return generated


# -------------------- Graph VAE -------------------- #
class GraphVAE:
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
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self._build_model()

    def _build_model(self):
        """
        Builds the encoder, decoder, and VAE models.
        """
        # Encoder
        encoder_inputs = keras.Input(shape=(self.input_dim,), name="graph_encoder_input")
        x = keras.layers.Dense(128, activation="relu", name="graph_dense1")(encoder_inputs)
        z_mean = keras.layers.Dense(self.latent_dim, name="graph_z_mean")(x)
        z_log_var = keras.layers.Dense(self.latent_dim, name="graph_z_log_var")(x)
        z = Sampling(name="graph_sampling")([z_mean, z_log_var])
        self.encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z],
                                    name="graph_encoder")

        # Decoder
        latent_inputs = keras.Input(shape=(self.latent_dim,), name="graph_z_sampling")
        x = keras.layers.Dense(128, activation="relu", name="graph_dense2")(latent_inputs)
        decoder_outputs = keras.layers.Dense(self.input_dim, activation="sigmoid",
                                             name="graph_decoder_output")(x)
        self.decoder = keras.Model(latent_inputs, decoder_outputs, name="graph_decoder")

        # VAE Model
        outputs = self.decoder(z)
        self.vae = keras.Model(encoder_inputs, outputs, name="graph_vae")

        reconstruction_loss = tf.reduce_mean(
            tf.square(encoder_inputs - outputs))
        kl_loss = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        self.vae.add_loss(reconstruction_loss + kl_loss)
        self.vae.compile(optimizer="adam")

    def train(self, x_train, epochs=1, batch_size=32):
        """
        Trains the Graph VAE.

        Args:
            x_train (np.ndarray): Training data.
            epochs (int): Number of epochs.
            batch_size (int): Batch size.
        """
        self.vae.fit(x_train, x_train, epochs=epochs, batch_size=batch_size)

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
        preds = self.decoder.predict(latent_samples)
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
        description="Train two VAEs for molecular generation and evaluate validity.")
    parser.add_argument(
        "--input", type=str, default="test-smiles.txt",
        help="Path to the input SMILES text file.")
    parser.add_argument(
        "--epochs", type=int, default=1,
        help="Number of training epochs for both models.")
    args = parser.parse_args()

    # Read and preprocess SMILES data
    smiles_list = read_smiles(args.input)
    if not smiles_list:
        print("No SMILES strings found in the input file.")
        return

    char_to_index, index_to_char = create_smiles_tokenizer(smiles_list)
    max_length = max(len(smi) for smi in smiles_list)
    encoded_smiles = np.array(
        [encode_smiles(smi, char_to_index, max_length) for smi in smiles_list],
        dtype=np.int32
    )

    # For the graph VAE, we use a dummy fixed dimension
    graph_dim = 100
    graph_data = np.array(
        [encode_graph(smi, graph_dim) for smi in smiles_list],
        dtype=np.float32
    )

    # Initialize and train SMILES VAE
    smiles_vae = SmilesVAE(vocab_size=len(char_to_index), max_length=max_length)
    smiles_vae.train(encoded_smiles, epochs=args.epochs)

    # Initialize and train Graph VAE
    graph_vae = GraphVAE(input_dim=graph_dim)
    graph_vae.train(graph_data, epochs=args.epochs)

    # Generate molecules and evaluate
    num_generate = 100
    generated_smiles_direct = smiles_vae.generate(num_generate, index_to_char)
    generated_smiles_graph = graph_vae.generate(num_generate)

    valid_pct_direct = evaluate_generation(generated_smiles_direct)
    valid_pct_graph = evaluate_generation(generated_smiles_graph)

    print("SMILES VAE generated molecules:")
    for smi in generated_smiles_direct:
        print(smi)
    print("\nGraph VAE generated molecules:")
    for smi in generated_smiles_graph:
        print(smi)

    print("\nEvaluation Metrics:")
    print("SMILES VAE valid percentage: {:.2f}%".format(valid_pct_direct))
    print("Graph VAE valid percentage: {:.2f}%".format(valid_pct_graph))


if __name__ == "__main__":
    main()
