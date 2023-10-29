import subprocess
import re

def get_pyenv_versions():
    command = "pyenv install --list"
    result = subprocess.run(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.stdout

def get_latest_python_version(pyenv_list_output):
    versions = re.findall(r'\s+(\d+\.\d+\.\d+)', pyenv_list_output)
    versions.sort(key=lambda s: list(map(int, s.split('.'))), reverse=True)
    return versions[1]

def install_latest_python():
    pyenv_list_output = get_pyenv_versions()
    latest_version = get_latest_python_version(pyenv_list_output)
    
    print(f"Installing Python {latest_version}...")
    
    command = f"pyenv install {latest_version}"
    result = subprocess.run(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    if result.returncode == 0:
        print(f"Successfully installed Python {latest_version}")
    else:
        print(f"Failed to install Python {latest_version}")
        print(result.stderr)

# Example usage
install_latest_python()
