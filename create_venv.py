import subprocess
import re
import venv

def get_latest_pyenv_python():
    command = "pyenv versions --bare"
    result = subprocess.run(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    versions = result.stdout.split()
    versions.sort(key=lambda s: list(map(int, s.split('.'))), reverse=True)
    return versions[0] if versions else None

def create_venv_with_latest_python():
    latest_python = get_latest_pyenv_python()
    
    if not latest_python:
        print("No Python versions found via pyenv.")
        return
    
    command = f"pyenv exec {latest_python} -m venv .venv"
    result = subprocess.run(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    if result.returncode == 0:
        print(f"Virtual environment .venv created in the current directory using Python {latest_python}.")
    else:
        print(f"Failed to create virtual environment using Python {latest_python}.")
        print(result.stderr)

# Example usage
create_venv_with_latest_python()
