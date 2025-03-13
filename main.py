"""
Extracts the SMILES feature from the QM9/DimeNet dataset using TensorFlow and saves it to a text file.
"""
import tensorflow_datasets as tfds

def extract_smiles_from_qm9_dimenet(output_file: str, split: str) -> None:
    """
    Extracts the SMILES feature from the QM9/DimeNet dataset and writes it to a text file.

    Args:
        output_file (str): Path to the output text file.
        split (str): Dataset split to load ("train", "test", or "validation").
    """
    dataset = tfds.load("qm9/dimenet", split=split)

    with open(output_file, "w", encoding="utf-8") as file:
        for sample in dataset:
            smiles = sample["SMILES"].numpy().decode("utf-8")
            file.write(smiles + "\n")

    print(f"SMILES data for {split} split successfully written to {output_file}")

if __name__ == "__main__":
    for split in ["train", "test", "validation"]:
        output_path = f"smiles_{split}.txt"
        extract_smiles_from_qm9_dimenet(output_path, split)
