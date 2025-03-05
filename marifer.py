import sys
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os

# Print the name of the script and the Python version
print(f"Running script: {sys.argv[0]}")
print(f"Python version: {sys.version}")

# Load the QM9 dataset
print("Loading QM9 dataset...")
dataset, info = tfds.load('qm9/original', with_info=True, as_supervised=False)

# Split the dataset into train, test, and validation datasets
train_data = dataset['train']
test_data = dataset['test']
validation_data = dataset['validation']

# Prepare the dataset: Convert it to a format we can use in the autoencoder
def preprocess_data(data):
    print("Preprocessing data...")
    features = []
    labels = []
    
    # Iterate through the dataset
    for example in data:
        # Extract the 'positions' (3D coordinates of atoms) and 'charges' (atomic charges) features
        positions = example['positions']  # Shape: (29, 3)
        charges = example['charges']  # Shape: (29,)
        
        # Concatenate both 'positions' and 'charges' as input features for the autoencoder
        input_data = np.concatenate([positions.flatten(), charges.flatten()])
        
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

# Define the autoencoder model
print("Building autoencoder model...")

def build_autoencoder(input_dim):
    # Encoder network
    input_layer = tf.keras.layers.Input(shape=(input_dim,))
    encoded = tf.keras.layers.Dense(128, activation='relu')(input_layer)
    encoded = tf.keras.layers.Dense(64, activation='relu')(encoded)
    encoded = tf.keras.layers.Dense(32, activation='relu')(encoded)
    
    # Decoder network
    decoded = tf.keras.layers.Dense(64, activation='relu')(encoded)
    decoded = tf.keras.layers.Dense(128, activation='relu')(decoded)
    decoded = tf.keras.layers.Dense(input_dim, activation='sigmoid')(decoded)
    
    # Autoencoder model
    autoencoder = tf.keras.models.Model(input_layer, decoded)
    
    # Encoder model (to extract the encoding)
    encoder = tf.keras.models.Model(input_layer, encoded)
    
    # Compile the autoencoder
    autoencoder.compile(optimizer='adam', loss='mse')
    
    print("Autoencoder model built successfully")
    return autoencoder, encoder

# Build the autoencoder
input_dim = train_features.shape[1]
autoencoder, encoder = build_autoencoder(input_dim)

# Train the autoencoder model
print("Training the autoencoder...")

history = autoencoder.fit(
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
test_loss = autoencoder.evaluate(test_features, test_labels, verbose=1)
print(f"Test loss: {test_loss}")

# Save the trained model
model_save_path = './autoencoder_model'
print(f"Saving the model to {model_save_path}...")
os.makedirs(model_save_path, exist_ok=True)
autoencoder.save(os.path.join(model_save_path, 'autoencoder.h5'))
encoder.save(os.path.join(model_save_path, 'encoder.h5'))

# Generate new latent space samples and reconstruct the molecules
print("Generating new samples from the latent space...")

# Sample random points in the latent space
latent_space_samples = np.random.normal(size=(10, 32))  # 10 new random samples with 32 dimensions

# Decode the random latent space samples back to the original data
generated_samples = autoencoder.layers[-2](latent_space_samples)  # Decode using the second-to-last layer (decoder)

print("Generated samples from latent space:")
print(generated_samples)

# Here we would continue by using these generated samples to generate new molecules and optimize them
# based on certain properties (such as predicting the 'gap', 'HOMO', 'LUMO', etc.).
