"""marifer.py"""
import os
import subprocess
os.makedirs("data", exist_ok=True)
os.makedirs("model", exist_ok=True)
os.makedirs("analysis", exist_ok=True)
subprocess.run(["uv", "run", "tfds", "build", "qm9/dimenet"], check=True)
subprocess.run(["uv", "run", "data.py"], check=True)
subprocess.run(["uv", "run", "model.py"], check=True)
subprocess.run(["uv", "run", "analysis.py"], check=True)
