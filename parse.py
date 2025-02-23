# Import required libraries
import tensorflow
import tensorflow_datasets
import matplotlib.pyplot as plt
import json

# Load the QM9 dataset
# QM9 is a quantum chemistry dataset of ~130k small organic molecules
qm9 = tensorflow_datasets.load("qm9/original")

# Get the training set
# This dataset doesn't have a specific train/test split, so we use the entire dataset
train = qm9["train"]

# Initialize a list to eventually store a dictionary for every molecule
# This will allow us to create a more structured representation of our data
molecule_graphs = []

# Iterate through each molecule in the training set
for molecule in train:
    # Decode the SMILES string from bytes to a regular string
    # SMILES (Simplified Molecular Input Line Entry System) is a specification for unambiguously describing molecular structure
    smiles = molecule["SMILES"].numpy().decode()
    
    # TODO: Add more processing here
    # 1. Parse the SMILES string to extract atoms and bonds
    # 2. Create a graph representation of the molecule
    # 3. Extract other relevant features from the molecule object
    
    # Create a dictionary to store molecule information
    molecule_info = {
        "smiles": smiles,
        # Add other molecule properties here
    }
    
    # Append the molecule info to our list
    molecule_graphs.append(molecule_info)

# Write dataset into a file
# This allows us to save our processed data for future use without reprocessing
with open('qm9_processed.json', 'w') as f:
    json.dump(molecule_graphs, f)

print(f"Processed {len(molecule_graphs)} molecules and saved to qm9_processed.json!")