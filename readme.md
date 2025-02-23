# Marifer
Molecular Generation with Deep Learning, Graph Representations, and the QM9 Dataset!

## Conda Environment
Launch Anaconda Prompt to begin! Then, enter these commands:
```
conda create -y -n=Marifer python=3.11
conda activate Marifer
pip install tensorflow
pip install tfds-nightly
pip install mlcroissant
pip install apache_beam
pip install matplotlib
tfds build qm9/original
```