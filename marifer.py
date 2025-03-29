"""Generate molecules!"""
import subprocess
import os
subprocess.run("uv run tfds build qm9/dimenet", check=True, cwd=os.getcwd())
subprocess.run("uv run data.py", check=True, cwd=os.getcwd())
subprocess.run("uv run model.py", check=True, cwd=os.getcwd())
