import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Input, Dense, Conv1D, Flatten, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy

# Load the dataset
qm9_dimenet = tfds.load("qm9/dimenet")
qm9_train = tfds.as_dataframe(qm9_dimenet["train"])
tokens = qm9_train["SMILES"].unique().tolist()

def one_hot_encode(smiles, max_length=120):
    """
    One-hot encodes a SMILES string.
    """
    encoded_smiles = [tokens.index(char) if char in tokens else len(tokens) for char in smiles]
    encoded_smiles += [len(tokens)] * (max_length - len(encoded_smiles))
    one_hot = np.zeros((max_length, len(tokens) + 1))
    for i, idx in enumerate(encoded_smiles):
        one_hot[i, idx] = 1
    return one_hot

def one_hot_to_smiles(one_hot_smiles):
    """
    Converts one-hot encoded SMILES back to a SMILES string.
    """
    return ''.join([tokens[np.argmax(vec)] for vec in one_hot_smiles])

def encoder_model(input_dim, latent_dim):
    """
    Creates the encoder part of the VAE model.
    """
    inputs = Input(shape=(input_dim, len(tokens) + 1))
    x = Conv1D(64, kernel_size=3, activation='relu')(inputs)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    z_mean = Dense(latent_dim)(x)
    z_log_var = Dense(latent_dim)(x)
    return Model(inputs=inputs, outputs=[z_mean, z_log_var])

def decoder_model(latent_dim, output_dim):
    """
    Creates the decoder part of the VAE model.
    """
    latent_inputs = Input(shape=(latent_dim,))
    x = Dense(128, activation='relu')(latent_inputs)
    x = Dense(output_dim * (len(tokens) + 1), activation='softmax')(x)
    outputs = Reshape((output_dim, len(tokens) + 1))(x)
    return Model(inputs=latent_inputs, outputs=outputs)

def sampling(args):
    """
    Sampling function to get latent vectors from the encoder output.
    """
    z_mean, z_log_var = args
    epsilon = tf.random.normal(tf.shape(z_mean), 0, 1, dtype=tf.float32)
    return z_mean + tf.exp(z_log_var / 2) * epsilon

class VAE(Model):
    """
    Variational Autoencoder model.
    """
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def compile(self, optimizer, loss_fn):
        super().compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encoder(data)
            z = sampling([z_mean, z_log_var])
            reconstructed = self.decoder(z)
            reconstruction_loss = self.loss_fn(data, reconstructed)
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss) * -0.5
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {"loss": total_loss, "reconstruction_loss": reconstruction_loss, "kl_loss": kl_loss}

def data_preprocessing(smiles_dataset):
    """
    Efficient data preprocessing using tf.data API.
    """
    def process_smiles(smiles):
        return one_hot_encode(smiles)
    
    # Create a dataset pipeline
    dataset = tf.data.Dataset.from_tensor_slices(smiles_dataset)
    dataset = dataset.map(lambda smiles: tf.py_function(process_smiles, [smiles], tf.float64))
    dataset = dataset.batch(32)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)  # Prefetch for better performance
    return dataset

if __name__ == "__main__":
    INPUT_DIM = 120
    LATENT_DIM = 128
    BATCH_SIZE = 32
    EPOCHS = 10

    smiles_dataset = qm9_train["SMILES"].tolist()

    # Optimized data preprocessing using tf.data pipeline
    dataset = data_preprocessing(smiles_dataset)

    # Model initialization
    encoder = encoder_model(INPUT_DIM, LATENT_DIM)
    decoder = decoder_model(LATENT_DIM, INPUT_DIM)
    vae = VAE(encoder, decoder)
    vae.compile(
        optimizer=Adam(learning_rate=0.001),
        loss_fn=CategoricalCrossentropy(from_logits=False)
    )

    # Training the model
    vae.fit(dataset, epochs=EPOCHS)

    # Generate molecules
    z_samples = tf.random.normal([10, LATENT_DIM])
    generated_smiles = vae.decoder(z_samples)
    for i, smiles in enumerate(generated_smiles):
        smiles_str = one_hot_to_smiles(smiles.numpy())
        print(f"Generated SMILES {i + 1}: {smiles_str}")
