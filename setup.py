import os
import subprocess
from setuptools import setup, find_packages

def init_rlbotpack():
    """Initialize RLBotPack submodule if not already initialized"""
    if os.path.exists("RLBotPack") and not os.listdir("RLBotPack"):
        print("Initializing RLBotPack submodule...")
        subprocess.check_call(["git", "submodule", "update", "--init", "--recursive"])
        print("RLBotPack submodule initialized successfully")
    else:
        print("RLBotPack submodule already initialized")

if __name__ == "__main__":
    # Initialize RLBotPack submodule during setup
    try:
        init_rlbotpack()
    except Exception as e:
        print(f"Warning: Could not initialize RLBotPack submodule: {str(e)}")
        print("You may need to run: git submodule update --init --recursive")

    setup(
        name="rlbot-training",
        version="0.1.0",
        description="RLBot training with curriculum learning and RLBotPack opponents",
        author="Your Name",
        packages=find_packages(),
        install_requires=[
            line.strip()
            for line in open("requirements.txt").readlines()
            if not line.startswith("#") and line.strip()
        ],
        python_requires=">=3.8",
    )