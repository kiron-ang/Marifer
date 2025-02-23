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

## Scientific Literature
Honestly, I am not well-versed in molecular generation as a deep learning focus. So, I have to read academic articles to supplement my current knowledge. Below, I summarize some articles I have found online:

First, I read a 2025 article describing how some researchers experimentally tested several architectures for de novo molecular generation [1]. Their work is valuable when considering reproducibility in the molecular generation realm, but they focus solely on natural language processing architectures. I wanted to utilize graph representations as input and output, so I continued to search.

Then, I found a 2024 review article that focused more broadly on generative molecular design methodologies [2].

[1]: Wang, Y., Guo, M., Chen, X. et al. Screening of multi deep learning-based de novo molecular generation models and their application for specific target molecular generation. Sci Rep 15, 4419 (2025). https://doi.org/10.1038/s41598-025-86840-z

[2]: Du, Y., Jamasb, A.R., Guo, J. et al. Machine learning-aided generative molecular design. Nat Mach Intell 6, 589–604 (2024). https://doi.org/10.1038/s42256-024-00843-5

