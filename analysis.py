"""analysis.py"""
import rdkit as rd

def read_smiles(file_path):
    """Read SMILES from file and return a list of SMILES strings"""
    with open(file_path, "r", encoding="utf-8") as f:
        smiles = [line.strip() for line in f.readlines()]
    return smiles

def calculate_fingerprints(smiles_list):
    """Calculate molecular fingerprints for a list of SMILES strings"""
    fingerprints = []
    for smiles in smiles_list:
        mol = rd.Chem.MolFromSmiles(smiles)
        if mol is not None:
            fingerprint = rd.Chem.AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            fingerprints.append((smiles, fingerprint))
    return fingerprints

def calculate_similarity(fp1, fp2):
    """Calculate Tanimoto similarity between two fingerprints"""
    return rd.DataStructs.FingerprintSimilarity(fp1, fp2)

def find_most_similar_molecules(fingerprints):
    """Find the most similar molecules based on Tanimoto similarity"""
    most_similar_pairs = []
    num_fps = len(fingerprints)
    for i in range(num_fps):
        smiles1, fp1 = fingerprints[i]
        max_similarity = 0
        most_similar_smiles = None
        for j in range(num_fps):
            if i != j:
                smiles2, fp2 = fingerprints[j]
                similarity = calculate_similarity(fp1, fp2)
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_smiles = smiles2
        most_similar_pairs.append((smiles1, most_similar_smiles, max_similarity))
    return most_similar_pairs

def analyze_similarity(smiles_file, output_file):
    """Analyze molecular similarity and save results to file"""
    smiles_list = read_smiles(smiles_file)
    fingerprints = calculate_fingerprints(smiles_list)
    most_similar_pairs = find_most_similar_molecules(fingerprints)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("SMILES1,SMILES2,Similarity\n")
        for smiles1, smiles2, similarity in most_similar_pairs:
            f.write(f"{smiles1},{smiles2},{similarity:.4f}\n")

    print("Similarity analysis complete. Results saved to", output_file)

if __name__ == "__main__":
    # Define file paths
    smiles_file = "data/test-SMILES.txt"
    output_file = "analysis/similarity_results.csv"

    # Perform similarity analysis
    analyze_similarity(smiles_file, output_file)
