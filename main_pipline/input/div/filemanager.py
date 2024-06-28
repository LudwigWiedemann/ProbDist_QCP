# filemanager.py
import time
from pathlib import Path

def create_folder(custom_name=None):
    """
    Creates folder if not exists, if failed prints error.
    """
    if custom_name:
        folder_name = custom_name
    else:
        time_started = time.strftime("%d.%m.%Y--%H-%M-%S", time.localtime())
        folder_name = f"PD output from {time_started}"

    path = Path(f"{str(Path(__file__).parent.parent.parent.resolve())}/output/{folder_name}")

    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print("Folder creation failed")

    return path
