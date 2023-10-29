import platform
import subprocess

def get_operating_system():
    os_name = platform.system()
    os_version = platform.release()
    return f"Operating System: {os_name}, Version: {os_version}"

def install_pyenv():
    os_info = get_operating_system()
    os_name = os_info.split(",")[0].split(":")[1].strip()
    
    if os_name == "Windows":
        # For Windows, you might install pyenv-win
        install_cmd = "pip install pyenv-win --target %USERPROFILE%\\.pyenv"
        subprocess.run(install_cmd, shell=True)
        
    elif os_name == "Linux":
        # For Linux, you might use a series of commands to install Pyenv
        install_cmds = [
            "sudo apt-get update",
            "sudo apt-get install -y build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python3-openssl git",
            "curl https://pyenv.run | bash"
        ]
        for cmd in install_cmds:
            subprocess.run(cmd, shell=True)
        
    elif os_name == "Darwin":
        # For macOS, you might use Homebrew to install Pyenv
        install_cmds = [
            "brew update",
            "brew install pyenv"
        ]
        for cmd in install_cmds:
            subprocess.run(cmd, shell=True)
        
    else:
        print("Unsupported OS")
        
    print(f"Pyenv installed on {os_name}")

# To use the function:
# install_pyenv()
