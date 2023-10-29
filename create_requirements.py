import subprocess
import os

def find_venv():
    for item in os.listdir('.'):
        if os.path.isdir(item) and item == '.venv':
            return True
    return False

def generate_requirements():
    if find_venv():
        venv_path = "./.venv/bin/activate"  # Adjust this path as needed
        command = f"source {venv_path} && pip freeze > requirements.txt"
        
        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if result.returncode == 0:
            print("Successfully generated requirements.txt based on the current .venv.")
        else:
            print("Failed to generate requirements.txt.")
            print(result.stderr)
    else:
        print("No .venv directory found in the current working directory.")

# Example usage
generate_requirements()
