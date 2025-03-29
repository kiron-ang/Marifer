"""
This module defines and trains a variational autoencoder (VAE) to produce novel SMILES strings.
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Lambda, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K

def model(num_samples=1000, temperature=0.8):
    """
    Train a variational autoencoder (VAE) to produce novel SMILES strings.

    Args:
        num_samples (int): Number of SMILES strings to generate. Default is 1000.
        temperature (float): Temperature parameter to control diversity. Default is 0.8.

    Returns:
        None
    """

    # Create model directory if it does not exist
    os.makedirs("model", exist_ok=True)

    # Load the SMILES strings from the training data file
    with open("data/train-SMILES.txt", "r", encoding="utf-8") as file:
        smiles_strings = file.readlines()

    # Tokenize the SMILES strings
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(smiles_strings)
    total_chars = len(tokenizer.word_index) + 1

    # Prepare the sequences for training
    sequences = []
    for smile in smiles_strings:
        encoded = tokenizer.texts_to_sequences([smile])[0]
        for i in range(1, len(encoded)):
            sequence = encoded[:i+1]
            sequences.append(sequence)

    max_sequence_len = max(len(seq) for seq in sequences)
    sequences = pad_sequences(sequences, maxlen=max_sequence_len, padding="pre")
    X = sequences[:, :-1]
    y = sequences[:, -1]
    y = tf.keras.utils.to_categorical(y, num_classes=total_chars)

    # Define VAE model architecture
    latent_dim = 100

    # Encoder
    inputs = Input(shape=(max_sequence_len-1,))
    embedding = Embedding(total_chars, 50, input_length=max_sequence_len-1)(inputs)
    lstm = LSTM(100, return_sequences=False)(embedding)
    z_mean = Dense(latent_dim)(lstm)
    z_log_var = Dense(latent_dim)(lstm)

    z = Lambda(lambda args: args[0] + K.exp(0.5 * args[1]) * K.random_normal(K.shape(args[0])),
                output_shape=(latent_dim,))([z_mean, z_log_var])

    # Decoder
    decoder_inputs = Input(shape=(latent_dim,))
    decoder_lstm = Dense(max_sequence_len-1)(decoder_inputs)
    decoder_lstm = LSTM(100, return_sequences=True)(decoder_lstm)
    outputs = Dense(total_chars, activation='softmax')(decoder_lstm)

    # VAE model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    decoder = Model(decoder_inputs, outputs, name='decoder')
    vae_outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, vae_outputs, name='vae')

    # Loss functions
    reconstruction_loss = tf.keras.losses.categorical_crossentropy(y, vae_outputs)
    reconstruction_loss *= max_sequence_len-1
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')

    # Train the model
    plt.figure()
    plt.plot(vae.fit(X, y, epochs=100).history["loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("model/loss-epoch.png")
    plt.close()

    generated_smiles = []
    molecular_weights = []
    for _ in range(num_samples):
        z_sample = np.random.normal(size=(1, latent_dim))
        decoded_sequence = decoder.predict(z_sample)
        text = []
        for char_prob in decoded_sequence[0]:
            preds = np.asarray(char_prob).astype('float64')
            preds = np.log(preds + 1e-10) / temperature
            exp_preds = np.exp(preds)
            preds = exp_preds / np.sum(exp_preds)
            probas = np.random.multinomial(1, preds, 1)
            next_index = np.argmax(probas)
            next_char = tokenizer.index_word[next_index]
            text.append(next_char)
            if next_char == "\n":
                break
        smiles = "".join(text)
        generated_smiles.append(smiles)
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            molecular_weights.append(Descriptors.MolWt(mol))
        else:
            molecular_weights.append(0)

    # Save the generated SMILES strings to a file
    with open("model/SMILES.txt", "w", encoding="utf-8") as output_file:
        for g in generated_smiles:
            output_file.write(g + "\n")

    # Save the molecular weights to a file
    with open("model/MolWt.txt", "w", encoding="utf-8") as mw_file:
        for mw in molecular_weights:
            mw_file.write(f"{mw}\n")

# Call the function
model(num_samples=1000, temperature=0.8)
