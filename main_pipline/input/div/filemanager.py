import time
from pathlib import Path

time_started = time.strftime("%d.%m.%Y--%H-%M-%S", time.localtime())  # adds time to name to insure diffent names
folder_name = f"PD output from {time_started}"
path = f"{str(Path(__file__).parent.parent.parent.resolve())}\output\{folder_name}"


def create_folder():
    """
        creates folder if not exists, if failed prints error

    """

    try:
        Path(path).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print("Folder creation failed")
