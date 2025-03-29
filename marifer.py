"""Generate molecules!"""
import subprocess
subprocess.run("uv run tfds build qm9/dimenet", check = True)
subprocess.run("uv run data.py", check = True)
subprocess.run("uv run model.py", check = True)
