import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# Print the name of the script and the Python version
print(f"Running script: {sys.argv[0]}")
print(f"Python version: {sys.version}")

# Load the QM9 dataset
print("Loading QM9 dataset...")
dataset, info = tfds.load('qm9/dimenet', with_info=True, as_supervised=False)

# Split the dataset into train, test, and validation datasets
train_data = dataset['train']
test_data = dataset['test']
validation_data = dataset['validation']

def preprocess_data(data):
    """
    Preprocess the dataset by selecting scalar features.

    Parameters:
    - data: The dataset to preprocess.

    Returns:
    - features: Selected scalar input features.
    - labels: Same as features for autoencoding.
    """
    print("Preprocessing data...")
    features = []
    labels = []

    # Iterate through the dataset
    for example in data:
        # Extract scalar features
        input_data = [
            example['A'],
            example['B'],
            example['C'],
            example['Cv'],
            example['G'],
            example['G_atomization'],
            example['H'],
            example['H_atomization'],
            example['U'],
            example['U0'],
            example['U0_atomization'],
            example['U_atomization'],
            example['alpha'],
            example['gap'],
            example['homo'],
            example['index'],
            example['lumo'],
            example['mu'],
            example['num_atoms'],
            example['r2'],
            example['zpve'],
        ]

        # For autoencoding, the output is the same as input
        features.append(input_data)
        labels.append(input_data)

    features = np.array(features)
    labels = np.array(labels)

    print(f"Processed {len(features)} samples")
    return features, labels

# Preprocess the train, test, and validation data
print("Preprocessing train data...")
train_features, train_labels = preprocess_data(train_data)

print("Preprocessing validation data...")
validation_features, validation_labels = preprocess_data(validation_data)

print("Preprocessing test data...")
test_features, test_labels = preprocess_data(test_data)

def build_autoencoder(input_size):
    """
    Build an autoencoder model with the specified input dimension.

    Parameters:
    - input_size: The dimension of the input data.

    Returns:
    - autoencoder: The autoencoder model.
    - encoder: The encoder part of the autoencoder.
    """
    # Encoder network
    input_layer = tf.keras.layers.Input(shape=(input_size,))
    encoded = tf.keras.layers.Dense(128, activation='relu')(input_layer)
    encoded = tf.keras.layers.Dense(64, activation='relu')(encoded)
    encoded = tf.keras.layers.Dense(32, activation='relu')(encoded)

    # Decoder network
    decoded = tf.keras.layers.Dense(64, activation='relu')(encoded)
    decoded = tf.keras.layers.Dense(128, activation='relu')(decoded)
    decoded = tf.keras.layers.Dense(input_size, activation='linear')(decoded)

    # Autoencoder model
    model = tf.keras.models.Model(input_layer, decoded)

    # Encoder model (to extract the encoding)
    encoder = tf.keras.models.Model(input_layer, encoded)

    # Compile the autoencoder
    model.compile(optimizer='adam', loss='mse')

    print("Autoencoder model built successfully")
    return model, encoder

# Determine the new input size based on the number of scalar features
new_input_size = len([
    'A',
    'B',
    'C',
    'Cv',
    'G',
    'G_atomization',
    'H',
    'H_atomization',
    'U',
    'U0',
    'U0_atomization',
    'U_atomization',
    'alpha',
    'gap',
    'homo',
    'index',
    'lumo',
    'mu',
    'num_atoms',
    'r2',
    'zpve',
])

# Build the autoencoder with the new input size
qm9_autoencoder_model, qm9_encoder_model = build_autoencoder(new_input_size)

# Train the autoencoder model
print("Training the autoencoder...")

history = qm9_autoencoder_model.fit(
    train_features,
    train_labels,
    epochs=50,
    batch_size=64,
    validation_data=(validation_features, validation_labels),
    verbose=1
)

# Print the training history to monitor the performance
print("Training complete. Here's the training history:")
print(history.history)

# Evaluate the model on the test set
print("Evaluating the model on the test set...")
test_loss = qm9_autoencoder_model.evaluate(test_features, test_labels, verbose=1)
print(f"Test loss: {test_loss}")

# Save the trained model
MODEL_SAVE_PATH = './autoencoder_model'
print(f"Saving the model to {MODEL_SAVE_PATH}...")
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
qm9_autoencoder_model.save(os.path.join(MODEL_SAVE_PATH, 'autoencoder.h5'))
qm9_encoder_model.save(os.path.join(MODEL_SAVE_PATH, 'encoder.h5'))

# Generate new latent space samples and reconstruct the molecules
print("Generating new samples from the latent space...")

# Sample random points in the latent space
# 10 new random samples with 32 dimensions
latent_space_samples = np.random.normal(size=(10, 32))

# Note: Since the decoder expects input from the encoded space (32 dimensions),
# you should first encode your input data to match this dimension.
# However, for demonstration, we'll directly use random latent space samples.
generated_samples = qm9_autoencoder_model.predict(qm9_encoder_model.predict(train_features[:10]))

print("Generated samples from latent space:")
print(generated_samples)
