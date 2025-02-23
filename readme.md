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

## Analyze and Parse the QM9 Dataset
The last command from the previous section instructs the computer to build the QM9 dataset hosted by Tensorflow. [You can learn more about it on their website.](https://www.tensorflow.org/datasets/catalog/qm9). To analyze the dataset for basic information and patterns, please use the Python programming language as demonstrated in the ``analyze.py`` file in this repository. It illustrates how to save several PNG figures into the ``analyze`` directory without needing to convert the QM9 molecules into graphs.

Additionally, the dataset must be processed to obtain molecular graph representations. [The QM9 dataset uses the "SMILES" system to represent molecular structures as strings.](https://pubs.acs.org/doi/abs/10.1021/ci00057a005) To parse these strings into adjacency lists, please see the ``parse.py`` file in this repository. It creates a new data file stored in the ``parse`` directory.

## Train the Marifer Model


## Scientific Literature
Honestly, I am not well-versed in molecular generation. So, I have to read academic articles to supplement my current knowledge. Below, I summarize some articles I have found online:

First, I read a 2025 article describing how some researchers experimentally tested several architectures for de novo molecular generation [1]. Their work is valuable when considering reproducibility in the molecular generation realm, but they focus solely on natural language processing architectures. I wanted to utilize graph representations as input and output, so I continued to search.

Then, I found a 2024 review article that focused more broadly on generative molecular design methodologies [2].

I also read the 1988 article that describes the "SMILES" system for representing chemical structures [3]. This article was referenced in the original 2014 paper for the QM9 dataset [4].

[1]: Wang, Y., Guo, M., Chen, X. et al. Screening of multi deep learning-based de novo molecular generation models and their application for specific target molecular generation. Sci Rep 15, 4419 (2025). https://doi.org/10.1038/s41598-025-86840-z

[2]: Du, Y., Jamasb, A.R., Guo, J. et al. Machine learning-aided generative molecular design. Nat Mach Intell 6, 589–604 (2024). https://doi.org/10.1038/s42256-024-00843-5

[3]: Weininger, D. SMILES, a chemical language and information system. 1.Introduction to methodology and encoding rules. J. Chem. Inf. Comp. Sci. 28, 31–36 (1988).

[4]: Ramakrishnan, R., Dral, P., Rupp, M. et al. Quantum chemistry structures and properties of 134 kilo molecules. Sci Data 1, 140022 (2014). https://doi.org/10.1038/sdata.2014.22
