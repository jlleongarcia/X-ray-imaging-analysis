import subprocess
import sys


def install_packages():
    """Install packages from requirements.txt."""
    print("Checking and installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Packages installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {e}")
        sys.exit(1)

def run_app(port=8502):
    """Run the Streamlit app."""
    print(f"Starting the Streamlit app on port {port}...")
    try:
        command = [
            sys.executable, "-m", "streamlit", "run", "menu_analyzer.py",
            "--server.port", str(port)
        ]
        subprocess.check_call(command)
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("Starting setup script...")

    # Install dependencies
    install_packages()

    # Define a default port and check for a command-line argument
    port_number = 8502
    if len(sys.argv) > 1:
        try:
            port_number = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port number '{sys.argv[1]}'. Using default port {port_number}.")

    # Run the Streamlit app
    run_app(port=port_number)