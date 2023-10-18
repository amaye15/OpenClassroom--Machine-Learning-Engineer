import os
import subprocess
import json
from typing import List

def get_latest_python_version() -> str:
    """
    Fetches the second latest Python version available through Conda.
    
    Returns:
        str: The latest Python version as a string.
    """
    # Fetch available Python versions from Conda
    result = subprocess.run(["conda", "search", "python", "--json"], capture_output=True, text=True)
    python_versions = json.loads(result.stdout)
    
    # Extract and sort the versions
    versions = []
    for package_info in python_versions.values():
        for package in package_info:
            versions.append(package['version'])
    
    latest_version = sorted(versions, key=lambda v: tuple(map(int, v.split('.'))))[-2]
    return latest_version

def create_conda_env(env_name: str, python_version: str, packages: List[str]) -> None:
    """
    Creates a new Conda environment based on the given parameters and saves it to an environment.yml file.
    
    Parameters:
        env_name (str): The name of the new Conda environment.
        python_version (str): The Python version to use in the new environment.
        packages (List[str]): A list of package names to install in the new environment.
    """
    # Create environment.yml file content
    env_content = f"""name: {env_name}
channels:
  - defaults
dependencies:
  - python={python_version}
"""
    for package in packages:
        env_content += f"  - {package}\n"

    # Write to environment.yml file
    with open("environment.yml", "w") as f:
        f.write(env_content)

    # Create Conda environment
    subprocess.run(["conda", "env", "create", "-f", "environment.yml"])

    print(f"Conda environment '{env_name}' has been created with Python {python_version}.")

if __name__ == "__main__":
    env_name = "p4_env"
    python_version = get_latest_python_version()
    packages = ["numpy", "pandas", "scikit-learn"]

    create_conda_env(env_name, python_version, packages)

    # Activate the environment (This won't affect the current shell, only a subshell)
    os.system(f"conda activate {env_name}")
