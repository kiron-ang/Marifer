# Import required libraries
import tensorflow
import tensorflow_datasets
import matplotlib.pyplot as plt
import csv

# Load the QM9 dataset
qm9 = tensorflow_datasets.load("qm9/original")

# Get the training set
train = qm9["train"]

# Initialize a list to store the number of nodes (atoms) for each molecule
number_of_nodes_each_molecule_has = []

# Iterate through each molecule in the training set
for molecule in train:
    # Decode the SMILES string from bytes to a regular string
    smiles = molecule["SMILES"].numpy().decode()
    
    # Count the number of atoms (alphabetic characters) in the SMILES string
    # This simple heuristic only works because the only atoms are C, H, O, N, and F.
    atoms = sum([1 for character in smiles if character.isalpha()])
    
    # Append the atom count to our list
    number_of_nodes_each_molecule_has.append(atoms)

# Create a histogram of the number of nodes (atoms) per molecule
plt.hist(number_of_nodes_each_molecule_has)

# Add labels and title to the plot
total_molecules = len(number_of_nodes_each_molecule_has)

plt.xlabel("Number of Atoms (Number of Nodes)")
plt.ylabel("Frequency")
plt.title(f"Distribution of Molecule Sizes in QM9 Dataset ({total_molecules} molecules)")

# Save the figure
plt.savefig("qm9_nodes_distribution.png", dpi = 800)


number_of_nodes_each_molecule_has_dictionary = {}

# Iterate over the list of atom counts
for number in number_of_nodes_each_molecule_has:
    if number in number_of_nodes_each_molecule_has_dictionary.keys():
        number_of_nodes_each_molecule_has_dictionary[number] += 1
    else:
        number_of_nodes_each_molecule_has_dictionary[number] = 1

with open("qm9_nodes_distribution.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Number of atoms", "Number of molecules with that number of atoms", "Percentage (%)"])
    for number, count in number_of_nodes_each_molecule_has_dictionary.items():
        percentage = (count / total_molecules) * 100
        writer.writerow([number, count, percentage])