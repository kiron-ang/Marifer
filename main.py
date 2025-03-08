import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv1D, Flatten, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy

# Define tokens for SMILES strings
tokens = ['C', 'N', 'O', '(', ')', '[', ']']

# Function to one-hot encode a SMILES string
def one_hot_encode(smiles, max_length=120):
    encoded_smiles = []
    for char in smiles:
        if char in tokens:
            encoded_smiles.append(tokens.index(char))
        else:
            # Handle unknown tokens
            encoded_smiles.append(len(tokens))
    
    # Pad to max_length if necessary
    encoded_smiles += [len(tokens)] * (max_length - len(encoded_smiles))
    
    # Convert to one-hot
    one_hot = np.zeros((max_length, len(tokens) + 1))
    for i, idx in enumerate(encoded_smiles):
        one_hot[i, idx] = 1
    
    return one_hot

# Function to convert one-hot encoded SMILES back to string
def one_hot_to_smiles(one_hot_smiles):
    smiles_str = ""
    for vec in one_hot_smiles:
        idx = np.argmax(vec)
        if idx < len(tokens):
            smiles_str += tokens[idx]
    return smiles_str

# Define the encoder model
def encoder_model(input_dim, latent_dim):
    inputs = Input(shape=(input_dim, len(tokens) + 1))
    x = Conv1D(64, kernel_size=3, activation='relu')(inputs)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    z_mean = Dense(latent_dim)(x)
    z_log_var = Dense(latent_dim)(x)
    return Model(inputs=inputs, outputs=[z_mean, z_log_var])

# Define the decoder model
def decoder_model(latent_dim, output_dim):
    latent_inputs = Input(shape=(latent_dim,))
    x = Dense(128, activation='relu')(latent_inputs)
    x = Dense(output_dim * (len(tokens) + 1), activation='softmax')(x)
    outputs = Reshape((output_dim, len(tokens) + 1))(x)
    return Model(inputs=latent_inputs, outputs=outputs)

# Sampling function for latent space
def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(tf.shape(z_mean), 0, 1, dtype=tf.float32)
    return z_mean + tf.exp(z_log_var / 2) * epsilon

# Define the VAE model
class VAE(Model):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def compile(self, optimizer, loss_fn):
        super(VAE, self).compile()
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

# Example usage
if __name__ == "__main__":
    # Parameters
    input_dim = 120  # Max length of SMILES
    latent_dim = 128
    batch_size = 32
    epochs = 10

    # Example SMILES dataset
    smiles_dataset = ["CCO", "CCN", "CCC", "CCCN", "CCCCN"]

    # One-hot encode SMILES dataset
    encoded_smiles = np.array([one_hot_encode(smiles) for smiles in smiles_dataset])

    # Define models
    encoder = encoder_model(input_dim, latent_dim)
    decoder = decoder_model(latent_dim, input_dim)
    vae = VAE(encoder, decoder)

    # Compile VAE
    vae.compile(optimizer=Adam(learning_rate=0.001), loss_fn=CategoricalCrossentropy(from_logits=False))

    # Train VAE
    vae.fit(encoded_smiles, epochs=epochs, batch_size=batch_size)

    # Generate new molecules
    def generate_molecule(vae, num_molecules):
        z_samples = tf.random.normal([num_molecules, latent_dim])
        generated_smiles = vae.decoder(z_samples)
        return generated_smiles

    generated_smiles = generate_molecule(vae, 10)

    # Convert generated SMILES back to strings
    for i, smiles in enumerate(generated_smiles):
        smiles_str = one_hot_to_smiles(smiles.numpy())
        print(f"Generated SMILES {i+1}: {smiles_str}")
