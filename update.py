import subprocess
from typing import Union

def update_current_conda_env(yaml_file_path: str) -> Union[None, str]:
    """
    Updates the current Conda environment using a specified YAML file.
    
    Parameters:
        yaml_file_path (str): The path to the YAML file that contains the environment specifications.

    Returns:
        Union[None, str]: None if successful, otherwise returns the error message.
    """
    # Update the current Conda environment using the specified YAML file
    result = subprocess.run(["conda", "env", "update", "--file", yaml_file_path, "--prune"], capture_output=True, text=True)

    if result.returncode == 0:
        print(f"Successfully updated the current Conda environment using {yaml_file_path}.")
        return None
    else:
        error_message = f"Failed to update the Conda environment. Error:\n{result.stderr}"
        print(error_message)
        return error_message

if __name__ == "__main__":
    # Uncomment the line below to actually run the function (Make sure the environment.yml file path is correct)
    update_result = update_current_conda_env("environment.yml")

    if update_result is not None:
        print("Update failed.")
