# filemanager.py
import time
from pathlib import Path

def create_folder(custom_name=None):
    """
    Creates a folder in the output directory.
    :param custom_name: str: Custom name for the folder
    :return: str: path
    """
    # Define the folder name
    if custom_name:
        folder_name = custom_name
    else:
        time_started = time.strftime("%d.%m.%Y--%H-%M-%S", time.localtime())
        folder_name = f"PD output from {time_started}"
    # Define the path
    path = Path(f"{str(Path(__file__).parent.parent.parent.resolve())}/output/{folder_name}")
    # Create the folder
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print("Folder creation failed")

    return path
