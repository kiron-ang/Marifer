print(0)
import tensorflow
import tensorflow_datasets
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.preprocessing import StandardScaler
qm9 = tensorflow_datasets.load("qm9/original")
qm9 = qm9["train"]
qm9 = tensorflow_datasets.as_dataframe(qm9)
smiles = qm9['SMILES'].values
properties = qm9[['gap', 'homo', 'lumo', 'mu', 'zpve']].values
def smiles_to_mol(smiles):
    return Chem.MolFromSmiles(smiles)
def get_molecular_features(mol):
    features = []
    if mol is not None:
        features.append(Chem.Descriptors.MolWt(mol))
        features.append(Chem.Descriptors.MolLogP(mol))
    return np.array(features)
molecular_features = []
for smi in smiles:
    mol = smiles_to_mol(smi)
    features = get_molecular_features(mol)
    molecular_features.append(features)
X = np.array(molecular_features)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y = properties[:, 0]
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential([
    Dense(64, input_dim=X_scaled.shape[1], activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_scaled, y, epochs=10, batch_size=32, validation_split=0.2)
loss = model.evaluate(X_scaled, y)
print(f'Model Loss: {loss}')
print(1)