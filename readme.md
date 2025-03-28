# Marifer Instructions
1. Install git: ``sudo dnf install -y git``
2. Install tar: ``sudo dnf install -y tar``
3. Install uv: ``curl -LsSf https://astral.sh/uv/install.sh | sh``
4. Clone this repository: ``git clone git@github.com:kiron-ang/Marifer``
5. Prepare the data: ``uv run data.py``
6. Train the model: ``uv run model.py``
7. Generate molecules: ``uv run marifer.py``