# Marifer

## Getting started
1. Install git: ``sudo dnf install -y git``
2. Install tar: ``sudo dnf install -y tar``
3. Install uv: ``curl -LsSf https://astral.sh/uv/install.sh | sh``
4. Clone this repository: ``git clone https://github.com/kiron-ang/Marifer``
5. Install dependencies: ``uv sync``
6. Build the QM9 dataset: ``uv run tfds build qm9/dimenet``
7. Run the code: ``uv run python main.py``
