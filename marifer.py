"""Generate molecules!"""
import subprocess
subprocess.run("uv run tfds build qm9/dimenet")
subprocess.run("uv run data.py")
subprocess.run("uv run model.py")
