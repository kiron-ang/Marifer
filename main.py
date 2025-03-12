import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
import tensorflow as tf
import tensorflow_datasets as tfds

# Load the QM9 dataset using the "dimenet" configuration.
# Using a dictionary for the split returns a dict with keys 'train', 'validation', and 'test'.
qm9_ds, qm9_info = tfds.load(
    "qm9/dimenet",
    split={'train': 'train', 'validation': 'validation', 'test': 'test'},
    with_info=True
)

# Extract each split into its own variable.
train_data = qm9_ds['train']
validation_data = qm9_ds['validation']
test_data = qm9_ds['test']

print(qm9_info)
