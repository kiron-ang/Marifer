"""
This script loads the QM9/DimeNet dataset and creates .txt files from its features.

The script performs the following operations:
1. Loads the specified TensorFlow dataset.
2. Iterates over each split in the dataset.
3. For each split, iterates over each feature.
4. Creates a .txt file for each feature, containing the feature values for that split.

The generated .txt files are stored in the 'data' directory in the format '{split}-{feature}.txt'.

Usage:
    Simply run the script with the desired TensorFlow dataset configuration name.
    Example: txt("qm9/dimenet")
"""
import os
import tensorflow_datasets as tfds
def txt(config):
    """
    Creates .txt files from a TensorFlow Dataset.

    This function takes a TensorFlow dataset configuration, loads the dataset,
    and then iterates over each split and feature to create .txt files containing
    the feature values.

    Args:
        config (str): The configuration name for the TensorFlow dataset to load.

    Creates:
        .txt files: For each split and feature in the dataset, a corresponding .txt file
                    is created in the 'data' directory. The files are named in the format
                    '{split}-{feature}.txt' and contain the feature values, one per line.
    """
    os.makedirs("data", exist_ok=True)
    dataset = tfds.load(config)
    for split in dataset:
        d = next(iter(dataset[split]))
        features = list(d.keys())
        for f in features:
            with open("data/" + split + "-" + f + ".txt", "w", encoding="utf-8") as txtfile:
                for d in dataset[split]:
                    value = d[f].numpy()
                    if isinstance(value, bytes):
                        value = value.decode("utf-8")
                    txtfile.write(str(value) + "\n")
txt("qm9/dimenet")
